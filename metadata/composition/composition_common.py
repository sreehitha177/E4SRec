#!/usr/bin/env python3
"""
Shared helpers for LLM-based composition feature annotation.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import AzureOpenAI

COMPOSITION_FEATURES = [
    "Focus on Lead Vocal",
    "Focus on Lyrics",
    "Focus on Melody",
    "Focus on Vocal Accompaniment",
    "Focus on Rhythmic Groove",
    "Focus on Musical Arrangements",
    "Focus on Form",
    "Focus on Riffs",
    "Focus on Performance",
]

FEATURE_DEFINITIONS = {
    "Focus on Lead Vocal": (
        "How compositionally dominant the lead vocal is in the overall track experience. "
        "Most vocal-driven songs should score high."
    ),
    "Focus on Lyrics": (
        "How compositionally dominant the lyrical content is in the overall track experience."
    ),
    "Focus on Melody": (
        "How compositionally dominant the melody is in the overall track experience."
    ),
    "Focus on Vocal Accompaniment": (
        "How compositionally dominant backing or harmony vocals are in the overall track experience."
    ),
    "Focus on Rhythmic Groove": (
        "How compositionally dominant the rhythmic feel or groove is in the overall track experience."
    ),
    "Focus on Musical Arrangements": (
        "How compositionally dominant the arrangement is, including instrument count and "
        "quality/novelty of part-writing and orchestration."
    ),
    "Focus on Form": (
        "How compositionally dominant the form is. More complex or non-traditional forms may score higher."
    ),
    "Focus on Riffs": (
        "How compositionally dominant repeated instrumental melodic motifs (riffs) are "
        "in the overall track experience."
    ),
    "Focus on Performance": (
        "How compositionally dominant instrumental performance skill is in the overall track experience. "
        "Instrumental jazz and classical often score high."
    ),
}

COMPOSITION_FEATURE_INDICES = {
    "Focus on Lead Vocal": 49,
    "Focus on Lyrics": 50,
    "Focus on Melody": 51,
    "Focus on Vocal Accompaniment": 52,
    "Focus on Rhythmic Groove": 53,
    "Focus on Musical Arrangements": 54,
    "Focus on Form": 55,
    "Focus on Riffs": 56,
    "Focus on Performance": 57,
}

AZURE_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "azure/gpt-5")
AZURE_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "https://dolby-metadata.openai.azure.com/")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")


def _get_azure_deployment_name(model_name: str) -> str:
    if model_name.startswith("azure/"):
        return model_name.split("/", 1)[1]
    return model_name


def normalize_feature_subset(selected_features: list[str] | None) -> list[str]:
    if not selected_features:
        return COMPOSITION_FEATURES.copy()

    normalized = []
    for feature in selected_features:
        feature_name = feature.strip()
        if not feature_name:
            continue
        if feature_name not in FEATURE_DEFINITIONS:
            raise ValueError(f"Unknown feature '{feature_name}'. Allowed: {COMPOSITION_FEATURES}")
        normalized.append(feature_name)

    if not normalized:
        raise ValueError("No valid features provided after normalization.")
    return normalized


def build_system_prompt(selected_features: list[str]) -> str:
    return (
        "You are an expert musicologist annotating compositional characteristics of songs. "
        "Rate each requested feature on a 0-5 scale using the provided definitions. "
        "Use at most 2 decimal places. "
        f"Return only a JSON object with exactly these keys: {selected_features}. "
        "No explanation, no markdown, no extra keys."
    )


def build_user_prompt(artist: str, title: str, selected_features: list[str]) -> str:
    feature_block = "\n".join(
        f'- "{feat}" (0-5): {FEATURE_DEFINITIONS[feat]}'
        for feat in selected_features
    )
    return (
        f'Track: "{title}" by {artist}\n\n'
        "Rate each requested feature from 0 (lowest) to 5 (highest).\n"
        "If composition cannot be inferred confidently, prefer moderate scores (2.0-3.0)\n"
        "instead of inventing details.\n\n"
        f"{feature_block}\n\n"
        f"Return a JSON object with exactly these keys: {selected_features}"
    )


def load_song_list(song_list_path: str) -> pd.DataFrame:
    return pd.read_csv(song_list_path, usecols=["artist_name", "track_name"])


def annotate_track(
    artist: str,
    title: str,
    api_key: str,
    selected_features: list[str],
    retries: int = 3,
) -> dict | None:
    deployment_name = _get_azure_deployment_name(AZURE_MODEL)
    for attempt in range(retries):
        try:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=AZURE_API_VERSION,
                azure_endpoint=AZURE_API_BASE,
            )
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": build_system_prompt(selected_features)},
                    {"role": "user", "content": build_user_prompt(artist, title, selected_features)},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            scores = json.loads(content if content is not None else "{}")
            normalized = {}
            for feature in selected_features:
                value = scores.get(feature)
                if value is None:
                    return None
                normalized[feature] = float(value) / 5.0
            return normalized
        except Exception as exc:  # pragma: no cover - network/runtime behavior
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed for {artist} - {title}: {exc}")
                return None
    return None


def annotate_df_parallel(
    df: pd.DataFrame,
    api_key: str,
    selected_features: list[str],
    artist_col: str = "artist_name",
    title_col: str = "track_name",
    max_workers: int = 10,
) -> pd.DataFrame:
    unique_tracks = df[[artist_col, title_col]].drop_duplicates()
    results = []

    def annotate_row(row: pd.Series) -> dict:
        scores = annotate_track(row[artist_col], row[title_col], api_key, selected_features)
        entry = {artist_col: row[artist_col], title_col: row[title_col]}
        entry.update(scores if scores else {feature: None for feature in selected_features})
        return entry

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(annotate_row, row): row for _, row in unique_tracks.iterrows()}
        for idx, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if idx % 100 == 0:
                print(f"{idx}/{len(unique_tracks)} done")

    return pd.DataFrame(results)
