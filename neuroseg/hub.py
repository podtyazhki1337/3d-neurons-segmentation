"""Download released model files from the Hugging Face Hub (only the files a model needs).

Set NEUROSEG_MODEL_DIR to a local copy of the Hub repository to work offline.
"""
from __future__ import annotations

import os

from .registry import HF_REPO


def fetch(spec: dict, cache_dir: str | None = None) -> str | None:
    """Local folder with the model files of `spec` (registry.model_spec); downloads them on first use."""
    if not spec["files"]:
        return None
    local = os.environ.get("NEUROSEG_MODEL_DIR")
    if local:
        path = os.path.join(local, spec["subfolder"])
        missing = [f for f in spec["files"] if not os.path.exists(os.path.join(local, f))]
        if missing:
            raise FileNotFoundError(f"NEUROSEG_MODEL_DIR={local} lacks {missing}")
        return path
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(HF_REPO, f, cache_dir=cache_dir) for f in spec["files"]]
    root = paths[0][: -len(spec["files"][0])]  # snapshot root of the Hub cache
    return os.path.join(root, spec["subfolder"])
