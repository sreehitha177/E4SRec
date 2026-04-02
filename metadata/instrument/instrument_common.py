#!/usr/bin/env python3
"""
Shared helpers for LLM-based instrument feature annotation.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import AzureOpenAI

INSTRUMENTAL_FEATURES = [
    "Drum Set",
    "Drum Aggressiveness",
    "Synthetic Drums",
    "Percussion",
    "Electric Guitar",
    "Electric Guitar Distortion",
    "Acoustic Guitar",
    "String Ensemble",
    "Horn Ensemble",
    "Piano",
    "Organ",
    "Rhodes",
    "Synthesizer",
    "Synth Timbre",
    "Bass Guitar",
    "Reed Instrument",
]

FEATURE_DEFINITIONS = {
    "Drum Set": (
        "Presence and dominance of a drum set in the instrumentation. "
        "0 = no drum set, 5 = drum set is the dominant element."
    ),
    "Drum Aggressiveness": (
        "Aggressiveness of the drum set performance. "
        "0 = extremely light (e.g. brushes, feather-light touch), "
        "5 = extreme aggression common in hard rock and metal."
    ),
    "Synthetic Drums": (
        "Presence and dominance of synthetic drums (MIDI pads, drum machines, or programmed beats). "
        "0 = no synthetic drums, 5 = fully synthetic/programmed drums dominant. "
        "Note: Synthetic Drums is a subset of Drum Set — score 0 if Drum Set is absent."
    ),
    "Percussion": (
        "Presence and dominance of percussion instruments excluding drum set "
        "(e.g. congas, bongos, tambourine, shakers, marimba). "
        "0 = no percussion, 5 = percussion is the dominant element."
    ),
    "Electric Guitar": (
        "Presence and dominance of electric guitar(s) in the instrumentation. "
        "0 = no electric guitar, 5 = electric guitar is the dominant element."
    ),
    "Electric Guitar Distortion": (
        "Overall degree and impact of guitar distortion. "
        "0 = very clean/unmodified tone, 5 = extremely dirty tone common in extreme metal."
    ),
    "Acoustic Guitar": (
        "Presence and dominance of acoustic guitar(s) in the instrumentation. "
        "0 = no acoustic guitar, 5 = acoustic guitar is the dominant element."
    ),
    "String Ensemble": (
        "Presence and dominance of a string ensemble (from two violins to a full orchestra). "
        "0 = no strings, 5 = strings are the dominant element."
    ),
    "Horn Ensemble": (
        "Presence and dominance of a horn ensemble (from two trumpets to a full concert band). "
        "0 = no horns, 5 = horns are the dominant element."
    ),
    "Piano": (
        "Presence and dominance of piano in the instrumentation. "
        "0 = no piano, 5 = piano is the dominant element."
    ),
    "Organ": (
        "Presence and dominance of organ in the instrumentation. "
        "0 = no organ, 5 = organ is the dominant element."
    ),
    "Rhodes": (
        "Presence and dominance of a Fender Rhodes or other electric piano. "
        "0 = no Rhodes/electric piano, 5 = Rhodes is the dominant element."
    ),
    "Synthesizer": (
        "Presence and dominance of synthesizer(s), excluding synths mimicking other instruments "
        "(horns, flutes, electric pianos, strings, etc.). "
        "0 = no synthesizer, 5 = synthesizer is the dominant element."
    ),
    "Synth Timbre": (
        "Timbral character of synthesizers present in the track. "
        "0 = ambient/atmospheric pads, 5 = industrial/robotic timbres common in techno and electronic music. "
        "Score 0 if no synthesizer is present."
    ),
    "Bass Guitar": (
        "Presence and dominance of bass guitar in the instrumentation. "
        "0 = no bass guitar, 5 = bass guitar is the dominant element."
    ),
    "Reed Instrument": (
        "Presence and dominance of reed instruments (saxophone, clarinet, oboe, english horn, etc.). "
        "0 = no reed instruments, 5 = reed instruments are the dominant element."
    ),
}

INSTRUMENTAL_FEATURE_INDICES = {
    "Drum Set": 19,
    "Drum Aggressiveness": 20,
    "Synthetic Drums": 21,
    "Percussion": 22,
    "Electric Guitar": 23,
    "Electric Guitar Distortion": 24,
    "Acoustic Guitar": 25,
    "String Ensemble": 26,
    "Horn Ensemble": 27,
    "Piano": 28,
    "Organ": 29,
    "Rhodes": 30,
    "Synthesizer": 31,
    "Synth Timbre": 32,
    "Bass Guitar": 33,
    "Reed Instrument": 34,
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
        return INSTRUMENTAL_FEATURES.copy()

    normalized = []
    for feature in selected_features:
        feature_name = feature.strip()
        if not feature_name:
            continue
        if feature_name not in FEATURE_DEFINITIONS:
            raise ValueError(f"Unknown feature '{feature_name}'. Allowed: {INSTRUMENTAL_FEATURES}")
        normalized.append(feature_name)

    if not normalized:
        raise ValueError("No valid features provided after normalization.")
    return normalized


def build_system_prompt(selected_features: list[str]) -> str:
    return (
        "You are an expert musicologist annotating instrumental feature characteristics of songs. "
        "Rate each requested feature on a 0-5 integer scale using the provided definitions and examples. "
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
        "Rate each requested feature from 0 (lowest) to 5 (highest):\n"
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
