"""Latency of each local router: one query at a time (batch size 1), warm model, CPU only.

Run from the project root, with nothing else heavy running:
    uv run python scripts/measure_latency.py

Uses a fixed random sample of 200 format-controlled validation queries (the
same input as E6) and router.cost.measure_latency, which runs 10 warm-up
queries first. CPU only, so the numbers describe a machine without a GPU.
S4's latency comes from its recorded API times instead (scripts/run_s4.py).

Writes results/latency.json.
"""

import json
import platform
import subprocess

import numpy as np
import torch

from router import data
from router.cost import measure_latency
from router.preprocess import format_control
from router.provenance import run_metadata
from router.systems.registry import load_predictor

SYSTEMS = {  # result tag -> (system, seed) loaded with the chosen checkpoint from each config
    "s0_tfidf": ("s0", 0),
    "s1_distilroberta-base": ("s1", 0),
    "s2_all-MiniLM-L6-v2": ("s2", 0),
    "s3_quora-distilroberta-base": ("s3", 0),
}
N_QUERIES = 200
SAMPLE_SEED = 0


def machine_name() -> str:
    """CPU model (macOS reports it through sysctl), plus the thread count torch uses."""
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor() or platform.machine()
    return f"{cpu}, CPU only, {torch.get_num_threads()} torch threads"


def main() -> None:
    texts = data.load_eval_set("val", with_oos=True)["text"].map(format_control).tolist()
    sample = list(np.random.default_rng(SAMPLE_SEED).choice(texts, size=N_QUERIES, replace=False))
    device = torch.device("cpu")
    results = {}
    for tag, (system, seed) in SYSTEMS.items():
        predict = load_predictor(system, seed=seed, device=device)
        results[tag] = measure_latency(lambda text: predict([text]), sample, warmup=10)
        print(f"{tag}: p50 {results[tag]['p50_ms']:.1f} ms, p95 {results[tag]['p95_ms']:.1f} ms")
        del predict
    out = {"metadata": run_metadata(data.load_config("data")), "machine": machine_name(), "batch_size": 1,
           "n_queries": N_QUERIES, "sample_seed": SAMPLE_SEED, "input": "val, format-controlled", "systems": results}
    with open(data.RESULTS_DIR / "latency.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote results/latency.json ({out['machine']})")


if __name__ == "__main__":
    main()
