import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """Fuses per-item embeddings from multiple modalities into one representation.

    Strategies
    ----------
    concat        — concatenate along feature dim; no params; output_dim = sum(modality_dims).
    weighted_sum  — project each modality to output_dim, L2-normalise, weighted sum.
                    Weights are fixed scalars (pass fixed_weights) for zero-shot, or a
                    learned nn.Parameter (fixed_weights=None) for fine-tuning.
    cross_attention — SASRec (modality 0) is the query; remaining modalities are
                    keys/values in a single multi-head attention layer.
    film          — remaining modalities produce per-feature scale + shift applied to
                    the projected SASRec embedding (Feature-wise Linear Modulation).

    The forward() method accepts inputs shaped [..., dim_i] and returns [..., output_dim],
    so it works identically for:
      - zero-shot precompute  : [n_items, dim_i]
      - fine-tuning per-batch : [batch, seq_len, dim_i]
    """

    def __init__(
        self,
        modality_dims: List[int],
        strategy: str,
        output_dim: int = None,
        fixed_weights: Optional[List[float]] = None,
        num_heads: int = 4,
    ):
        super().__init__()
        assert len(modality_dims) >= 1
        self.strategy = strategy
        self.modality_dims = modality_dims
        n = len(modality_dims)

        if strategy == "concat":
            self.output_dim = sum(modality_dims)

        elif strategy == "weighted_sum":
            self.output_dim = output_dim or modality_dims[0]
            self.projs = nn.ModuleList([
                nn.Linear(dim, self.output_dim) for dim in modality_dims
            ])
            if fixed_weights is not None:
                total = sum(fixed_weights)
                w = torch.tensor([wi / total for wi in fixed_weights], dtype=torch.float32)
                self.register_buffer("_weights", w)
                self._learned = False
            else:
                self.log_weights = nn.Parameter(torch.zeros(n))
                self._learned = True

        elif strategy == "cross_attention":
            assert n >= 2, "cross_attention needs at least 2 modalities"
            self.output_dim = output_dim or modality_dims[0]
            self.query_proj = nn.Linear(modality_dims[0], self.output_dim)
            self.kv_projs = nn.ModuleList([
                nn.Linear(dim, self.output_dim) for dim in modality_dims[1:]
            ])
            # reduce num_heads until it divides output_dim
            nh = num_heads
            while self.output_dim % nh != 0 and nh > 1:
                nh -= 1
            self.attn = nn.MultiheadAttention(self.output_dim, num_heads=nh, batch_first=True)
            self.out_proj = nn.Linear(self.output_dim, self.output_dim)

        elif strategy == "film":
            assert n >= 2, "film needs at least 2 modalities"
            self.output_dim = output_dim or modality_dims[0]
            cond_dim = sum(modality_dims[1:])
            self.base_proj  = nn.Linear(modality_dims[0], self.output_dim)
            self.scale_proj = nn.Linear(cond_dim, self.output_dim)
            self.shift_proj = nn.Linear(cond_dim, self.output_dim)

        else:
            raise ValueError(
                f"Unknown fusion strategy {strategy!r}. "
                "Choose from 'concat', 'weighted_sum', 'cross_attention', 'film'."
            )

    # ------------------------------------------------------------------
    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        """
        embeds : list of tensors each shaped [..., dim_i].
        Returns a tensor shaped [..., output_dim].
        """
        assert len(embeds) == len(self.modality_dims), (
            f"Expected {len(self.modality_dims)} modality tensors, got {len(embeds)}"
        )

        if self.strategy == "concat":
            return torch.cat(embeds, dim=-1)

        elif self.strategy == "weighted_sum":
            projected = [proj(e) for proj, e in zip(self.projs, embeds)]
            normed    = [F.normalize(p, p=2, dim=-1) for p in projected]
            if self._learned:
                w = torch.softmax(self.log_weights, dim=0)
            else:
                w = self._weights.to(normed[0].device)
            return sum(wi * e for wi, e in zip(w, normed))

        elif self.strategy == "cross_attention":
            lead   = embeds[0].shape[:-1]          # e.g. (n_items,) or (bs, T)
            flat_n = embeds[0].reshape(-1, embeds[0].shape[-1]).shape[0]

            q  = self.query_proj(embeds[0]).reshape(flat_n, 1, self.output_dim)
            kv = torch.cat(
                [proj(e).reshape(flat_n, 1, self.output_dim)
                 for proj, e in zip(self.kv_projs, embeds[1:])],
                dim=1,
            )                                       # [flat_n, M, D]

            out, _ = self.attn(q, kv, kv)          # [flat_n, 1, D]
            out    = self.out_proj(out.squeeze(1))  # [flat_n, D]
            return out.reshape(*lead, self.output_dim)

        elif self.strategy == "film":
            base  = self.base_proj(embeds[0])           # [..., D]
            cond  = torch.cat(embeds[1:], dim=-1)       # [..., cond_dim]
            scale = self.scale_proj(cond)               # [..., D]
            shift = self.shift_proj(cond)               # [..., D]
            return scale * base + shift


# ---------------------------------------------------------------------------
# Shared helper used by both zero-shot scripts and finetune.py
# ---------------------------------------------------------------------------

def load_node_embeddings(
    node_path: str,
    mapping_path: str,
    n_items: int,
    item_map: dict,
    model_name: str,
) -> Optional[torch.Tensor]:
    """Load per-item embeddings from a node directory into a remapped tensor.

    Reads  <node_path>/embeddings.csv  which lists (artist_name, track_name,
    embedding_path) rows, joins on the master map to get raw item_id, then
    remaps through item_map to produce a [n_items, emb_dim] tensor whose row i
    holds the embedding for the item whose remapped index is i.

    Returns None if node_path is missing or no embeddings matched.
    """
    if not node_path or not os.path.isdir(node_path):
        print(f"Skipping {model_name}: node_path not provided or invalid.")
        return None

    print(f"\nLoading {model_name} embeddings from {node_path}...")
    emb_csv = pd.read_csv(os.path.join(node_path, "embeddings.csv"))
    master  = pd.read_csv(mapping_path)

    emb_csv["_key"] = (emb_csv["artist_name"].str.lower().str.strip() + "||" +
                       emb_csv["track_name"].str.lower().str.strip())
    master["_key"]  = (master["artist_name"].str.lower().str.strip() + "||" +
                       master["track_name"].str.lower().str.strip())

    merged = (emb_csv.merge(master[["item_id", "_key"]], on="_key", how="inner")
                     .drop_duplicates(subset=["track_index"]))
    print(f"   {model_name} files: {len(emb_csv)} | "
          f"master items: {len(master)} | matched: {len(merged)}")

    if len(merged) == 0:
        print(f"   WARNING: no matched {model_name} embeddings. Returning None.")
        return None

    sample_file = os.path.basename(merged["embedding_path"].iloc[0])
    sample = torch.load(os.path.join(node_path, sample_file),
                        map_location="cpu", weights_only=False)
    emb_dim = sample.shape[0]
    print(f"   {model_name} embedding dim: {emb_dim}")

    aligned = torch.zeros(n_items, emb_dim, dtype=torch.float32)

    def _load_one(row):
        pt = os.path.join(node_path, os.path.basename(row["embedding_path"]))
        new_idx = item_map.get(int(row["item_id"]), 0)
        if os.path.exists(pt) and new_idx > 0:
            return new_idx, torch.load(pt, map_location="cpu", weights_only=False).float()
        return None

    rows  = [r for _, r in merged.iterrows()]
    found = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(_load_one, rows):
            if result is not None:
                idx, emb = result
                aligned[idx] = emb
                found += 1

    print(f"   Loaded {found} / {len(merged)} matched {model_name} embeddings.")
    return aligned
