"""Module-level singletons loaded once at app startup.

Accessing these before calling `initialize()` will raise RuntimeError.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path for all ML imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

_model = None
_vocabs = None
_samples = None
_device = None

_initialized = False


def initialize() -> None:
    global _model, _vocabs, _samples, _device, _initialized
    if _initialized:
        return

    import torch
    from model_training import build_model_from_checkpoint, select_device
    from data_analysis.routes_cluster import _load_samples_and_vocabs

    checkpoint_path = str(_PROJECT_ROOT / "data" / "models" / "analysis_model.pt")
    cache_path = str(_PROJECT_ROOT / "data" / "preprocessed_routes_cache.pt")
    db_path = str(_PROJECT_ROOT / "data" / "raw" / "kilter_database.sqlite")

    print("Loading route samples and vocabs...")
    samples, vocabs = _load_samples_and_vocabs(
        db_path=db_path,
        cache_path=cache_path,
        checkpoint_path=checkpoint_path,
        metadata_source="kilter",
        metadata_product_id=1,
    )
    _samples = samples
    _vocabs = vocabs
    print(f"  {len(samples)} routes loaded.")

    device = select_device()
    print(f"Loading model from {checkpoint_path} on {device}...")
    model, _ = build_model_from_checkpoint(checkpoint_path, vocabs, device)
    model.eval()
    _model = model
    _device = device
    print("  Model ready.")

    _initialized = True


def get_model():
    if not _initialized:
        raise RuntimeError("startup.initialize() has not been called")
    return _model


def get_vocabs():
    if not _initialized:
        raise RuntimeError("startup.initialize() has not been called")
    return _vocabs


def get_samples():
    if not _initialized:
        raise RuntimeError("startup.initialize() has not been called")
    return _samples


def get_device():
    if not _initialized:
        raise RuntimeError("startup.initialize() has not been called")
    return _device


def project_root() -> Path:
    return _PROJECT_ROOT
