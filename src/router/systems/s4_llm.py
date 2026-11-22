"""S4: LLM zero-shot baseline (README section 7), Claude Sonnet 4.6 on Amazon Bedrock.

S4 is what the router is meant to replace: the LLM Jeff calls today. It sees
the list of 77 intent names and picks one, or out_of_scope. Its error rate is
the "matched risk" in E6, and its measured token cost is the price of every
query the router escalates.

Rigour rule 7: the prompt lives in the repo (prompts/), temperature is 0, and
every response is cached to disk with its model ID, token usage and timestamp.
A cached query is never sent again, and a hard spend cap stops a run before it
can cost more than configs/s4_llm.yaml allows.

The Anthropic SDK is imported only inside make_client(), so the notebook and
the evaluation scripts can read cached S4 predictions without it installed.
"""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from router.data import OOS_LABEL
from router.preprocess import format_control
from router.systems.base import Predictions

INVALID = "invalid"  # label for a reply that is not one of the allowed answers (scored as an error)


# ---------------------------------------------------------------------------
# Prompt and answer format
# ---------------------------------------------------------------------------

def build_system_prompt(
    template: str, label_names: list[str], oos_answer: str, examples: dict[str, list[str]] | None = None,
) -> str:
    """Fill the prompt template with the intent list, one name per line.

    With examples (the few-shot variant), each name is followed by its example messages.
    """
    if examples is None:
        lines = [f"- {name}" for name in label_names]
    else:
        lines = [f"- {name} (examples: " + "; ".join(f'"{text}"' for text in examples[name]) + ")" for name in label_names]
    return template.format(intent_list="\n".join(lines), oos_answer=oos_answer)


def pick_examples(train_df: pd.DataFrame, label_names: list[str], per_intent: int, seed: int) -> dict[str, list[str]]:
    """per_intent train queries for every intent, picked at random with a fixed seed.

    They are format-controlled like the queries S4 is asked about, so the
    examples do not look different from the questions.
    """
    rng = np.random.default_rng(seed)
    examples = {}
    for name in label_names:
        texts = sorted(train_df.loc[train_df["label"] == name, "text"].map(format_control).unique())
        chosen = sorted(rng.choice(len(texts), size=per_intent, replace=False))
        examples[name] = [texts[i] for i in chosen]
    return examples


def output_schema(label_names: list[str], oos_answer: str) -> dict:
    """JSON schema for the structured answer: {"intent": <one of the 77 names or out_of_scope>}."""
    return {
        "type": "object",
        "properties": {"intent": {"type": "string", "enum": list(label_names) + [oos_answer]}},
        "required": ["intent"],
        "additionalProperties": False,
    }


def parse_answer(raw_text: str, label_names: list[str], oos_answer: str) -> str:
    """Turn the model's reply into a label: an intent name, "oos", or "invalid".

    Accepts the structured JSON reply ({"intent": "..."}) and, as a fallback,
    a plain reply that is exactly one allowed name. Anything else is invalid.
    """
    answer = raw_text.strip()
    try:
        value = json.loads(answer)
        if isinstance(value, dict):
            answer = str(value.get("intent", "")).strip()
    except json.JSONDecodeError:
        pass  # not JSON: treat the whole reply as the answer
    if answer == oos_answer:
        return OOS_LABEL
    return answer if answer in set(label_names) else INVALID


# ---------------------------------------------------------------------------
# Cost, cache and spend cap
# ---------------------------------------------------------------------------

def call_cost(usage: dict, prices: dict) -> float:
    """US dollars for one call: each kind of token times its price per million tokens."""
    return (
        usage.get("input_tokens", 0) * prices["input"]
        + usage.get("output_tokens", 0) * prices["output"]
        + usage.get("cache_creation_input_tokens", 0) * prices["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * prices["cache_read"]
    ) / 1_000_000


def upper_bound_call_cost(system_prompt: str, max_tokens: int, prices: dict) -> float:
    """A deliberately high guess at one call's cost, used before any call has been measured:
    2 characters per token (real text averages about 4) plus 200 tokens for the query, and a full-length reply."""
    input_tokens = len(system_prompt) / 2 + 200
    return (input_tokens * prices["input"] + max_tokens * prices["output"]) / 1_000_000


def cache_key(model_id: str, system_prompt: str, text: str) -> str:
    """One key per (model, exact prompt, query). Any change to the prompt gives new keys,
    so a stale cached answer can never be reused for a different prompt."""
    return hashlib.sha256(f"{model_id}\n{system_prompt}\n{text}".encode("utf-8")).hexdigest()


class ResponseCache:
    """Append-only JSON Lines file with one record per API call, keyed by cache_key."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["key"]] = record
        self._lock = threading.Lock()  # several threads may add records at once

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def add(self, record: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self.records[record["key"]] = record

    def total_cost(self) -> float:
        return float(sum(record["cost_usd"] for record in self.records.values()))


class SpendCapReached(RuntimeError):
    """Raised before a call that could push total spend over the cap."""


class SpendTracker:
    """Running spend total that refuses any call which could cross the cap.

    Before each call, reserve() sets aside 1.5 times the most expensive call so
    far (or the upper-bound guess before the first call). After the call,
    settle() swaps that reservation for the real cost. Thread-safe.
    """

    def __init__(self, cap_usd: float, spent_usd: float, first_call_estimate_usd: float):
        self.cap = cap_usd
        self.spent = spent_usd
        self.largest_call = first_call_estimate_usd
        self.reserved = 0.0
        self._lock = threading.Lock()

    def reserve(self) -> float:
        with self._lock:
            estimate = 1.5 * self.largest_call
            if self.spent + self.reserved + estimate > self.cap:
                raise SpendCapReached(
                    f"Stopping: spent ${self.spent:.4f}, and the next call could take the total over the ${self.cap:.2f} cap."
                )
            self.reserved += estimate
            return estimate

    def settle(self, estimate: float, actual: float) -> None:
        with self._lock:
            self.reserved -= estimate
            self.spent += actual
            self.largest_call = max(self.largest_call, actual)


class RateLimiter:
    """Spaces out call start times so no more than `per_minute` calls start in any minute."""

    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute
        self.next_start = time.monotonic()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            start = max(self.next_start, time.monotonic())
            self.next_start = start + self.interval
        time.sleep(max(0.0, start - time.monotonic()))


# ---------------------------------------------------------------------------
# Calling Bedrock
# ---------------------------------------------------------------------------

def make_client(cfg: dict):
    """The Bedrock client, using the configured AWS profile and region."""
    from anthropic import AnthropicBedrock  # imported here: reading cached results must not need the SDK

    return AnthropicBedrock(aws_profile=cfg["aws_profile"], aws_region=cfg["aws_region"], max_retries=cfg["sdk_max_retries"])


def classify_one(client, cfg: dict, system_prompt: str, schema: dict, text: str) -> dict:
    """One API call for one query. Returns the raw reply, stop reason, token usage and latency."""
    request = {
        "model": cfg["model_id"],
        "max_tokens": cfg["max_tokens"],
        # Explicit cache breakpoint on the fixed instructions (Bedrock has no automatic caching).
        # Below the model's minimum cacheable length it silently does nothing.
        "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": text}],
        # SDK 1.x has no temperature argument; extra_body adds it to the request as-is.
        "extra_body": {"temperature": cfg["temperature"]},
    }
    if cfg["structured_output"]:
        request["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    start = time.perf_counter()
    response = client.messages.create(**request)
    latency_ms = (time.perf_counter() - start) * 1000
    usage = response.usage
    return {
        "raw": "".join(block.text for block in response.content if block.type == "text"),
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        },
        "latency_ms": latency_ms,
    }


def fill_cache(
    texts: list[str], cfg: dict, label_names: list[str], system_prompt: str,
    cache: ResponseCache, tracker: SpendTracker, client,
) -> dict:
    """Call the API for every text not already in the cache, within the rate limit and spend cap.

    Returns counts: calls made, and whether the spend cap stopped the run early.
    """
    schema = output_schema(label_names, cfg["oos_answer"])
    missing = list(dict.fromkeys(t for t in texts if cache.get(cache_key(cfg["model_id"], system_prompt, t)) is None))
    limiter = RateLimiter(cfg["requests_per_minute"])
    stop = threading.Event()  # set once the cap is reached, so no new calls start
    counts = {"calls": 0, "stopped_by_cap": False}
    count_lock = threading.Lock()

    def work(text: str) -> None:
        if stop.is_set():
            return
        try:
            estimate = tracker.reserve()
        except SpendCapReached as error:
            print(error)
            stop.set()
            counts["stopped_by_cap"] = True
            return
        limiter.wait()
        try:
            result = classify_one(client, cfg, system_prompt, schema, text)
        except Exception:
            tracker.settle(estimate, 0.0)  # failed call: release the reservation, then fail loudly
            stop.set()
            raise
        cost = call_cost(result["usage"], cfg["prices_usd_per_million_tokens"])
        tracker.settle(estimate, cost)
        cache.add({
            "key": cache_key(cfg["model_id"], system_prompt, text),
            "model_id": cfg["model_id"],
            "prompt_version": cfg["prompt_version"],
            "text": text,
            **result,
            "cost_usd": cost,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        with count_lock:
            counts["calls"] += 1
            if counts["calls"] % 50 == 0:
                print(f"  {counts['calls']} / {len(missing)} calls, spend so far ${tracker.spent:.4f}", flush=True)

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as pool:
        for future in [pool.submit(work, text) for text in missing]:
            future.result()  # re-raises any API error from a worker thread
    return counts


def predictions_from_cache(
    texts: list[str], cfg: dict, label_names: list[str], system_prompt: str, cache: ResponseCache,
) -> Predictions:
    """S4 predictions from cached replies only (no API call). Fails if any text is missing.

    S4 always answers itself, so confidence is 1.0. Its OOS score is 1.0 when it
    said out_of_scope and 0.0 otherwise, which lets E1 and E3 score it like the other systems.
    """
    records = [cache.get(cache_key(cfg["model_id"], system_prompt, t)) for t in texts]
    n_missing = sum(record is None for record in records)
    if n_missing:
        raise KeyError(f"{n_missing} of {len(texts)} queries have no cached S4 reply. Run scripts/run_s4.py first.")
    labels = [parse_answer(record["raw"], label_names, cfg["oos_answer"]) for record in records]
    is_oos = np.array([label == OOS_LABEL for label in labels])
    return Predictions(
        pred_label=labels,
        confidence=np.ones(len(labels)),
        oos_score=is_oos.astype(float),
    )
