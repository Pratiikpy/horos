"""Run the recorder: issue on the candle clock, score when windows close, anchor on X Layer.

    python -m scripts.recorder_daemon --device cuda:0
    python -m scripts.recorder_daemon --once            # one cycle, for checking a deployment

**Exactly one of these may run against a ledger at a time.** Two writers would interleave appends
into one hash chain and corrupt it — not lose data, but produce a chain that fails verification
permanently, which for this product is worse. There is a lock file to make that hard to do by
accident, and it is checked rather than advisory.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.commit import AnchorError, Anchorer                       # noqa: E402
from core.config import get_settings                                # noqa: E402
from core.crypto import Signer                                      # noqa: E402
from core.forecaster import scheduled_forecaster                    # noqa: E402
from core.kronos_runner import KronosRunner                         # noqa: E402
from core.ledger import Ledger                                      # noqa: E402
from core.market_data import MarketData                             # noqa: E402
from core.recorder import Recorder                                  # noqa: E402

log = logging.getLogger("horos.daemon")


class SingleWriter:
    """A lock file holding the owning pid, so a second recorder refuses to start."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        if self.path.exists():
            try:
                pid = int(self.path.read_text().split()[0])
            except (ValueError, IndexError):
                pid = -1
            if pid > 0 and _alive(pid):
                raise SystemExit(
                    f"another recorder (pid {pid}) already owns this ledger. Only one writer may "
                    f"append to a hash chain; a second would corrupt it. Stop that process, or "
                    f"delete {self.path} if it is stale.")
            log.warning("clearing a stale lock from pid %s", pid)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{os.getpid()} {time.time():.0f}")
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def build(device: str) -> Recorder:
    s = get_settings()
    ledger = Ledger(s.ledger_path, Signer(s.signing_key_path))
    data = MarketData(cache_ttl=s.cache_ttl_seconds,
                      timeout_ms=int(s.http_timeout_seconds * 1000))
    runner = KronosRunner(device=device)
    anchorer = None
    if s.anchor_enabled:
        try:
            anchorer = Anchorer(s.xlayer_rpc, s.xlayer_chain_id, s.anchor_key, s.xlayer_explorer)
            log.info("anchoring from %s, balance %.9f OKB", anchorer.address,
                     anchorer.balance_wei() / 1e18)
        except AnchorError as e:
            # Reported, never silently disabled. An operator who thinks anchoring is on when it is
            # not would publish a record claiming an integrity property it does not have.
            log.error("anchoring is unavailable: %s", e)
    else:
        log.warning("HOROS_ANCHOR_KEY is not set — the ledger will chain but will not be committed "
                    "on X Layer, and the scorecard will say so")
    return Recorder(ledger, data, forecaster=scheduled_forecaster(data, runner), anchorer=anchorer)


def main() -> int:
    p = argparse.ArgumentParser(description="Horos recorder")
    p.add_argument("--device", default=os.environ.get("HOROS_DEVICE", "cpu"))
    p.add_argument("--once", action="store_true", help="one issue+score+anchor cycle, then exit")
    p.add_argument("--score-only", action="store_true", help="score closed windows and exit")
    p.add_argument("--score-every", type=int, default=300)
    p.add_argument("--anchor-every", type=int, default=900)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    s = get_settings()
    with SingleWriter(Path(s.data_dir) / "recorder.lock"):
        rec = build(args.device)
        log.info("ledger: %d entries, head %s", rec.ledger.count(), rec.ledger.head()[:24])

        if args.score_only:
            for row in rec.score_round():
                log.info("%s", row)
            return 0

        if args.once:
            log.info("issue: %s", rec.issue_round())
            log.info("score: %s", rec.score_round())
            log.info("anchor: %s", rec.anchor_now())
            return 0

        stopping = {"now": False}

        def handle(signum, _frame):
            log.info("signal %s — finishing the current step and stopping", signum)
            stopping["now"] = True

        signal.signal(signal.SIGINT, handle)
        signal.signal(signal.SIGTERM, handle)

        rec.run_forever(score_every=args.score_every, anchor_every=args.anchor_every,
                        stop=lambda: stopping["now"])
        ok, problems = rec.ledger.verify()
        log.info("stopped. ledger verifies: %s %s", ok, problems or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
