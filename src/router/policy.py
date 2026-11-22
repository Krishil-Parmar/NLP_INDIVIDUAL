"""Router policy: what to do with each query (README section 7.1).

    HANDLE    the router is confident; dispatch the intent directly (near-zero cost)
    REJECT    clearly out of scope; return a canned reply (near-zero cost)
    ESCALATE  unsure; send the query to the LLM (full cost)

Policy v1 (required): HANDLE if confidence >= t_high, otherwise ESCALATE.
Policy v2 (stretch, M4+): also REJECT when the out-of-scope score >= t_oos.

Thresholds are chosen on validation data only (M4). The dangerous error is a
confident wrong HANDLE, above all on an out-of-scope query.
"""

import numpy as np

HANDLE = "HANDLE"
REJECT = "REJECT"
ESCALATE = "ESCALATE"


def decide(confidence: float, t_high: float) -> str:
    """Policy v1 for one query."""
    return HANDLE if confidence >= t_high else ESCALATE


def decide_all(confidence: np.ndarray, t_high: float) -> np.ndarray:
    """Policy v1 for many queries at once. Returns an array of HANDLE / ESCALATE strings."""
    return np.where(np.asarray(confidence) >= t_high, HANDLE, ESCALATE)
