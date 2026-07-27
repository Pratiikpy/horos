"""The service catalogue: what Horos sells, what each one costs, and what it promises.

Every service is one bounded request/response with a declared schema. That is what per-call x402
pricing is for, it is what an agent mid-task can consume, and it is what makes each answer
independently checkable.

Three rules are enforced structurally rather than by review, because each one has a rejected listing
behind it:

**Descriptions are two-part, name their parameters, carry a usage example, and fit in 500
characters.** OKX rejects a one-part description outright, and "the description claims more than the
output delivers" is the rejection they name as most common. `validate()` fails at import if any
service breaks this, so a bad description cannot reach a listing.

**Every response says what it could not see.** A forecast on a symbol with no public record says so.
A band that has not been conformally calibrated says so. A source that was unreachable is reported as
unreachable, never as an absence.

**Nothing returns a bare 500.** Every failure is a typed code with a sentence a caller can act on.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_DESCRIPTION = 500


class ServiceError(Exception):
    """A failure with a code and an explanation. The only kind of failure a handler may raise."""

    def __init__(self, message: str, code: str = "service_error", status: int = 422, **extra):
        super().__init__(message)
        self.code = code
        self.status = status
        self.extra = extra

    def as_dict(self) -> dict:
        return {"error": str(self), "code": self.code, **self.extra}


@dataclass(frozen=True)
class Param:
    name: str
    kind: str
    description: str
    default: Any = None
    required: bool = False


@dataclass
class Service:
    endpoint: str
    group: str
    price_usdt: float
    title: str
    summary: str            # what it does
    returns: str            # what you get back
    params: tuple[Param, ...]
    example: dict
    handler: Callable[[dict, Any], dict]
    depth: str = ""         # the honest statement of how far it actually goes
    # The marketplace caps serviceName at 30 characters. `title` is the readable one used on our own
    # pages; this is what gets listed. Set explicitly rather than truncated, because a name chopped
    # mid-word is how a listing ends up reading "Probability of reaching a le".
    short: str = ""

    @property
    def description(self) -> str:
        """The two-part description the listing carries. Kept under the 500-character limit."""
        text = f"{self.summary.strip()} {self.returns.strip()}"
        example = f" Example: {self.example_line()}"
        if len(text) + len(example) <= MAX_DESCRIPTION:
            return text + example
        return text[:MAX_DESCRIPTION - len(example)].rstrip() + example

    def example_line(self) -> str:
        import json
        if not self.example and not self.params:
            return "{} (no parameters)"
        return json.dumps(self.example, separators=(",", ":"), ensure_ascii=False)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params if p.required)

    def input_contract(self) -> dict:
        """Advertised in the 402 challenge, so a buying agent knows what to send before it pays."""
        return {
            "type": "object",
            "required": list(self.required),
            "properties": {
                p.name: {"type": p.kind, "description": p.description,
                         **({"default": p.default} if p.default is not None else {})}
                for p in self.params
            },
            "example": self.example,
        }

    def price_str(self) -> str:
        return f"{self.price_usdt:.6f}".rstrip("0").rstrip(".") or "0"

    def listing(self) -> dict:
        return {"endpoint": self.endpoint, "group": self.group, "title": self.title,
                "price_usdt": self.price_str(), "description": self.description,
                "depth": self.depth, "input": self.input_contract()}


class Registry:
    def __init__(self) -> None:
        self._by_endpoint: dict[str, Service] = {}

    def add(self, service: Service) -> Service:
        if service.endpoint in self._by_endpoint:
            raise ValueError(f"duplicate service endpoint {service.endpoint!r}")
        self._by_endpoint[service.endpoint] = service
        return service

    def get(self, endpoint: str) -> Service | None:
        return self._by_endpoint.get(endpoint)

    def all(self) -> list[Service]:
        return list(self._by_endpoint.values())

    def endpoints(self) -> list[str]:
        return list(self._by_endpoint)

    def by_group(self) -> dict[str, list[Service]]:
        out: dict[str, list[Service]] = {}
        for s in self._by_endpoint.values():
            out.setdefault(s.group, []).append(s)
        return out

    def validate(self) -> None:
        """Fail at import rather than at review.

        A listing whose description is a stub, or which advertises a parameter its example does not
        use, is rejected by OKX. Catching it here means it cannot be deployed at all.
        """
        problems: list[str] = []
        for s in self._by_endpoint.values():
            if len(s.description) > MAX_DESCRIPTION:
                problems.append(f"{s.endpoint}: description is {len(s.description)} chars "
                                f"(limit {MAX_DESCRIPTION})")
            if len(s.summary.split()) < 8:
                problems.append(f"{s.endpoint}: summary is too thin to describe the service")
            if not s.returns.strip():
                problems.append(f"{s.endpoint}: no statement of what the caller gets back")
            # A service with no parameters at all is correctly exemplified by `{}` — that really is
            # how you call it. A service *with* parameters and an empty example is a stub.
            if not s.example and s.params:
                problems.append(f"{s.endpoint}: no usage example, but it declares "
                                f"{len(s.params)} parameter(s)")
            for name in s.required:
                if name not in s.example:
                    problems.append(f"{s.endpoint}: required parameter {name!r} is missing from "
                                    f"its own usage example")
            for key in s.example:
                if key not in {p.name for p in s.params}:
                    problems.append(f"{s.endpoint}: the example passes {key!r}, which is not a "
                                    f"declared parameter")
            if not s.depth:
                problems.append(f"{s.endpoint}: no depth statement — every service must say how far "
                                f"it actually goes")
            name = s.short or s.title
            if len(name) > 30:
                problems.append(f"{s.endpoint}: listed name is {len(name)} chars, the marketplace "
                                f"caps serviceName at 30: {name!r}")
            # A zero fee makes the marketplace's own task-402-pay flow fail with an amount
            # mismatch (OKX builder chat, 2026-07-25). The public record stays genuinely free on
            # the web surface — /scorecard, /ledger and /ledger/verify need no payment at all — but
            # a *listed* service priced at zero is a broken purchase path, not a gift.
            if s.price_usdt == 0:
                problems.append(f"{s.endpoint}: listed at zero. A $0 x402 gate breaks "
                                f"task-402-pay; price it at least 0.001 and keep the free access "
                                f"on the public web pages.")
        if problems:
            raise RuntimeError("the service catalogue is not fit to list:\n  "
                               + "\n  ".join(problems))


# ── the response envelope ────────────────────────────────────────────────────────────────────────

def envelope(service: Service, request_input: dict, output: dict, signer,
             notes: list[str] | None = None, started: float | None = None) -> dict:
    """The signed wrapper every paid answer travels in.

    The signature covers a manifest of six fields, all of which are echoed in the envelope, so a
    buyer can rebuild the manifest from the response alone and verify it without asking us anything.
    """
    from core.crypto import digest

    job_id = uuid.uuid4().hex
    input_hash = digest(request_input)
    output_hash = digest(output)
    manifest = {
        "endpoint": service.endpoint,
        "input_sha256": input_hash,
        "output_sha256": output_hash,
        "tool": "horos",
        "price_usdt": service.price_str(),
        "job_id": job_id,
    }
    return {
        "service": "Horos",
        "endpoint": service.endpoint,
        "title": service.title,
        "job_id": job_id,
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3) if started else None,
        "price_usdt": service.price_str(),
        "input": request_input,
        "input_sha256": input_hash,
        "output": output,
        "output_sha256": output_hash,
        # Never omitted. An empty list is a claim that nothing was unmeasurable; a populated one is
        # the honest alternative to a confident answer with a hole in it.
        "notes": notes or [],
        "receipt": signer.sign(manifest),
        "verify": "GET /verify for the exact steps to check this receipt offline",
    }


REGISTRY = Registry()


def service(endpoint: str, group: str, price: float, title: str, summary: str, returns: str,
            depth: str, params: tuple[Param, ...], example: dict, short: str = ""):
    """Decorator registering one paid service."""
    def wrap(fn: Callable[[dict, Any], dict]) -> Callable[[dict, Any], dict]:
        REGISTRY.add(Service(endpoint=endpoint, group=group, price_usdt=price, title=title,
                             short=short, summary=summary, returns=returns, depth=depth,
                             params=params, example=example, handler=fn))
        return fn
    return wrap
