"""Temperature scaling (Guo et al., 2017), README section 7.1.

A softmax classifier is often over-confident. Temperature scaling divides the
logits by one number T before the softmax: T > 1 softens the probabilities,
T < 1 sharpens them. T is chosen on validation data to minimise the negative
log-likelihood of the true intents. It is used for S0 and S1, the two systems
with logits. (S2's cosine and S3's single sigmoid are not rescaled: a
monotone rescaling cannot change which queries they route.)
"""

import numpy as np

from router.systems.base import softmax

# 400 candidate temperatures, evenly spaced on a log scale from 0.05 to 10.
TEMPERATURE_GRID = np.logspace(np.log10(0.05), np.log10(10.0), 400)


def negative_log_likelihood(logits: np.ndarray, true_ids: np.ndarray) -> float:
    """Mean of -log(probability of the true class), computed stably from logits."""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)  # subtracting the max avoids overflow in exp
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(true_ids)), true_ids].mean())


def fit_temperature(logits: np.ndarray, true_ids: np.ndarray, grid: np.ndarray = TEMPERATURE_GRID) -> float:
    """The temperature from the grid with the lowest negative log-likelihood. A grid search: simple and deterministic."""
    losses = [negative_log_likelihood(np.asarray(logits) / t, true_ids) for t in grid]
    return float(grid[int(np.argmin(losses))])


def calibrated_confidence(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Max softmax probability after dividing the logits by the temperature."""
    return softmax(np.asarray(logits, dtype=float), temperature=temperature).max(axis=1)
