"""Optional, rank-zero CSV decision logs (independent of the text log level)."""

import csv
import os
import sys
from pathlib import Path


def log_cache_decision(context, prefix, can_cache, reason, diff=None, threshold=None):
    path = os.environ.get("CACHE_DIT_DECISION_CSV")
    if not path:
        return

    # Use SGLang's actual loop index when available, without importing SGLang
    # into standalone Cache-DiT applications.
    forward_context = None
    module = sys.modules.get("sglang.multimodal_gen.runtime.managers.forward_context")
    if module is not None:
        try:
            forward_context = module.get_forward_context()
        except AssertionError:
            pass  # No active model forward.
    batch = getattr(forward_context, "forward_batch", None)
    row = {
        "run_label": os.environ.get("CACHE_DIT_RUN_LABEL", ""),
        "run_id": os.environ.get("CACHE_DIT_RUN_ID", ""),
        "request_id": getattr(batch, "request_id", ""),
        "is_warmup": getattr(batch, "is_warmup", ""),
        "denoising_step": getattr(forward_context, "current_timestep", ""),
        "cache_step": context.get_current_step(),
        "transformer_step": context.get_current_transformer_step(),
        "cache_context": context.name,
        "prefix": prefix,
        "cfg_branch": (
            ("uncond" if batch.is_cfg_negative else "cond")
            if batch is not None
            else ("cfg" if context.is_separate_cfg_step() else "main")
        ),
        "result": "HIT" if can_cache else "MISS",
        "reason": reason,
        "diff": diff,
        "threshold": threshold,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Close after each decision to retain partial logs if a worker is stopped.
    # Only rank zero writes; sequential runs can append to the same CSV.
    with destination.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
