

def test_every_prose_field_is_ascii_not_only_the_brief():
    """A delivered A2A artifact reached a customer reading `periods â€" described as ordinary`.

    The em-dash came from `models.role` — a fixed string in a2a.py — which sat outside the
    normalisation. It was encoded to UTF-8, decoded again as cp1252 by the marketplace's
    compose-and-write step, and re-encoded: the bytes in the delivered file are
    C3 A2 E2 82 AC E2 80 9D and no real em-dash survives.

    That step is not ours to fix; handing it no non-ASCII is.
    """
    from services.a2a import to_ascii_punctuation

    payload = {
        "brief": {"summary": "Funding is ordinary — the 43rd percentile."},
        "models": {"role": "it produced no numbers — every figure comes from a named service"},
        "not_advice": "Horos measures and forecasts — it does not tell you whether to trade.",
        "understanding": "you want 24h risk “and” funding",
        "out_of_scope": ["anything needing a directional view …"],
    }
    cleaned = to_ascii_punctuation(payload)
    flat = repr(cleaned)
    for ch in ("—", "–", "‘", "’", "“", "”", "…"):
        assert ch not in flat, f"{ch!r} survived normalisation"
    assert "--" in cleaned["models"]["role"]
    assert "..." in cleaned["out_of_scope"][0]


def test_raw_results_keep_the_characters_the_services_produced():
    """The brief is checked against raw_results, so rewriting characters inside it would break that
    correspondence. That exclusion was deliberate and must survive the wider normalisation."""
    import services.a2a as a2a

    src = a2a.__dict__
    assert "to_ascii_punctuation" in src
    import inspect
    body = inspect.getsource(a2a)
    assert 'raw = payload.pop("raw_results")' in body
    assert 'payload["raw_results"] = raw' in body
