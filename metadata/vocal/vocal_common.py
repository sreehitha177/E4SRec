#!/usr/bin/env python3
"""
Shared helpers for LLM-based vocal feature annotation.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import AzureOpenAI

VOCAL_FEATURES = [
    "Vocal Register",
    "Vocal Timbre Thin to Full",
    "Vocal Breathiness",
    "Vocal Smoothness",
    "Vocal Grittiness",
    "Vocal Nasality",
    "Vocal Accompaniment",
]

FEATURE_DEFINITIONS = {
    "Vocal Register": (
        "The pitch range of the lead vocalist. "
        "0 = very low/bass, 5 = very high/soprano. "
        "Reference: Male midrange is A below middle C to E above middle C; "
        "female is approximately a fourth higher."
    ),
    "Vocal Timbre Thin to Full": (
        "The thickness/richness of vocal tone. "
        "0 = thin/wispy/weak fundamental, 5 = full/resonant/strong fundamental. "
        "Examples: 0 = Olivia Newton-John, Margot Timmins; 5 = Aretha Franklin, Josh Groban."
    ),
    "Vocal Breathiness": (
        "Presence of aspiration (breath) with lightness or airiness. "
        "0 = not breathy/clean tone, 5 = very breathy/airy. "
        "Examples of high breathiness: Hope Sandoval (Mazzy Star), Mariah Carey, Nick Drake."
    ),
    "Vocal Smoothness": (
        "Absence of roughness/raspiness, yielding roundness or sweetness. "
        "0 = very rough/raspy, 5 = very smooth/polished. "
        "Examples of high smoothness: Sade, Natalie Cole, James Taylor, Mel Torme."
    ),
    "Vocal Grittiness": (
        "Roughness/raspiness yielding harshness or dirtiness. "
        "0 = completely clean, 5 = extremely gritty/distorted. "
        "Examples of high grittiness: Tom Waits, Joe Cocker, Janis Joplin."
    ),
    "Vocal Nasality": (
        "Pinched or plugged-up quality, especially on nasal consonants (n, m, d). "
        "0 = not nasal, 5 = very nasal. "
        "Examples of high nasality: Bob Dylan, Joe Walsh, Reba McEntire."
    ),
    "Vocal Accompaniment": (
        "Level of dominance of accompaniment vocals - any vocal activity not in a lead role. "
        "0 = no backing vocals, 5 = very prominent/dominant backing vocals."
    ),
}

VOCAL_FEATURE_INDICES = {
    "Vocal Register": 0,
    "Vocal Timbre Thin to Full": 1,
    "Vocal Breathiness": 2,
    "Vocal Smoothness": 3,
    "Vocal Grittiness": 4,
    "Vocal Nasality": 5,
    "Vocal Accompaniment": 6,
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
        return VOCAL_FEATURES.copy()

    normalized = []
    for feature in selected_features:
        feature_name = feature.strip()
        if not feature_name:
            continue
        if feature_name not in FEATURE_DEFINITIONS:
            raise ValueError(f"Unknown feature '{feature_name}'. Allowed: {VOCAL_FEATURES}")
        normalized.append(feature_name)

    if not normalized:
        raise ValueError("No valid features provided after normalization.")
    return normalized


def build_system_prompt(selected_features: list[str]) -> str:
    return (
        "You are an expert musicologist annotating vocal characteristics of songs. "
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
