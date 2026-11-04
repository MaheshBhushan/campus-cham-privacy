"""Synthetic twin with SDV: fit a generative model on a table, sample new rows that keep the
statistical patterns but belong to nobody."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

NUMERIC = {"age", "enrolment_year", "semester", "ects", "avg_grade"}
BOOLEAN = {"disability_support", "at_risk"}


def _metadata(df: pd.DataFrame) -> Metadata:
    md = Metadata()
    md.add_table("students")
    for col in df.columns:
        if col in BOOLEAN:
            md.add_column(col, "students", sdtype="boolean")
        elif col in NUMERIC and pd.api.types.is_numeric_dtype(df[col]):
            md.add_column(col, "students", sdtype="numerical")
        else:
            # Generalised values ("20-24", "93***") and codes are categories, never PII to re-fake.
            md.add_column(col, "students", sdtype="categorical")
    return md


def synthesise(df: pd.DataFrame, model: str = "gaussian_copula", n: int | None = None, seed: int = 0, epochs: int = 300) -> pd.DataFrame:
    md = _metadata(df)
    # SDV draws from numpy/torch global RNGs during fit and sample; fix them for reproducible runs.
    np.random.seed(seed)
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "gaussian_copula":
            synth = GaussianCopulaSynthesizer(md, enforce_min_max_values=True, enforce_rounding=True)
        elif model == "ctgan":
            synth = CTGANSynthesizer(md, epochs=epochs, enforce_min_max_values=True, enforce_rounding=True)
        else:
            raise ValueError(f"unknown model {model!r}")
        synth.fit(df)
        out = synth.sample(num_rows=n or len(df))
    for col in BOOLEAN & set(out.columns):
        out[col] = out[col].astype(bool)
    return out[df.columns]
