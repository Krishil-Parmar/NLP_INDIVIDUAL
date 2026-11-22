"""Local router server for the demo (README section 12.2).

The demo front end is the real Jeff chat screen from the Velm app, run as a
dev-only harness (apps/mobile/lib/jeff_router_demo_main.dart in the Velm repo).
Every message Jeff receives is sent here first. This server runs the router
(the same src/router code as the notebook) and answers with its decision:

    HANDLE    confident: handled locally, no LLM call
    ESCALATE  unsure: in production this query would be sent to the LLM

It runs offline on a laptop CPU, uses only the Python standard library for the
web part, listens on 127.0.0.1 only, and never calls an LLM. Session counters
(queries, handled, escalated, LLM calls saved) live in memory.

Run from the project root:
    uv run python demo/app.py                       # S3, default threshold
    uv run python demo/app.py --system s2 --threshold 0.9

Endpoints:
    GET  /health   -> {"ok": true, "system": ...}
    POST /route    {"text": "...", "threshold": 0.8}  -> decision, evidence, session counts
    POST /reset    -> clears the session counters
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from router.policy import ESCALATE, HANDLE, decide
from router.seed import pick_device
from router.systems.registry import load_predictor

SYSTEM_NAMES = {
    "s0": "S0 TF-IDF + logistic regression",
    "s1": "S1 fine-tuned DistilRoBERTa",
    "s2": "S2 embedding nearest neighbour",
    "s3": "S3 pairwise paraphrase matcher",
}


class RouterState:
    """The loaded system plus the running session counts. One per server."""

    def __init__(self, system: str, default_threshold: float):
        self.system = system
        self.default_threshold = default_threshold
        self.predict = load_predictor(system, seed=0, device=pick_device())
        self.lock = threading.Lock()  # one prediction at a time: the models are not thread-safe
        self.reset()

    def reset(self) -> None:
        self.queries = 0
        self.handled = 0
        self.escalated = 0

    def route(self, text: str, threshold: float) -> dict:
        with self.lock:
            start = time.perf_counter()
            preds = self.predict([text])
            latency_ms = (time.perf_counter() - start) * 1000
            confidence = float(preds.confidence[0])
            decision = decide(confidence, threshold)
            self.queries += 1
            if decision == HANDLE:
                self.handled += 1
            elif decision == ESCALATE:
                self.escalated += 1
            return {
                "decision": decision,
                "intent": preds.pred_label[0],
                "confidence": confidence,
                "threshold": threshold,
                "matched_example": preds.matched_example[0] if preds.matched_example else None,
                "system": SYSTEM_NAMES[self.system],
                "latency_ms": round(latency_ms, 1),
                "session": {
                    "queries": self.queries,
                    "handled": self.handled,
                    "escalated": self.escalated,
                    "llm_calls_saved": self.handled,  # every HANDLE is one LLM call not made
                },
            }


def make_handler(state: RouterState):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            # The Flutter web harness is served from another localhost port, so the
            # browser needs these CORS headers to let it read the reply.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self) -> None:  # browser CORS pre-flight check
            self._send(204, {})

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "system": SYSTEM_NAMES[state.system], "default_threshold": state.default_threshold})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "body must be JSON"})
                return
            if self.path == "/reset":
                state.reset()
                self._send(200, {"ok": True})
            elif self.path == "/route":
                text = str(body.get("text", "")).strip()
                if not text:
                    self._send(400, {"error": "text is required"})
                    return
                threshold = float(body.get("threshold", state.default_threshold))
                self._send(200, state.route(text, threshold))
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, fmt: str, *args) -> None:  # one short line per request
            print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=list(SYSTEM_NAMES), default="s3")
    # Thin-demo default. M4 replaces it with the t_high chosen on validation.
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Loading {SYSTEM_NAMES[args.system]} ...")
    state = RouterState(args.system, args.threshold)
    state.route("warm up", args.threshold)  # first call is slow; do it before the audience sees it
    state.reset()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"Router ready on http://127.0.0.1:{args.port} (threshold {args.threshold}). Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
