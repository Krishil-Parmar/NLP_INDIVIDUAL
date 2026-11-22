"""Cost and latency model (README section 8.1).

cost_per_query = router_cost + (1 - coverage) * llm_cost, where llm_cost is
S4's measured cost per call and coverage is the share the router decides itself.
"""

import time
from collections.abc import Callable

import numpy as np


def cost_per_query(coverage: float, llm_cost: float, router_cost: float = 0.0) -> float:
    """Average cost of one query when the router handles `coverage` of them and escalates the rest.

    router_cost defaults to 0: the local model runs on hardware the assistant
    already has (an assumption stated in README section 8.1). Its time cost is
    reported separately as latency.
    """
    return router_cost + (1.0 - coverage) * llm_cost


def measure_latency(
    predict_one: Callable[[str], object], texts: list[str], warmup: int = 10
) -> dict[str, float]:
    """Wall-clock time per query at batch size 1, with a warm model.

    The first `warmup` calls are run but not timed, so one-off start-up costs
    (memory allocation, caches) do not count. Returns milliseconds.
    """
    for text in texts[:warmup]:
        predict_one(text)
    times_ms = []
    for text in texts:
        start = time.perf_counter()
        predict_one(text)
        times_ms.append((time.perf_counter() - start) * 1000)
    times = np.array(times_ms)
    return {
        "n_queries": len(times),
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }
