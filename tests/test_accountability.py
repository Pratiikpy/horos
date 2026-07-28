

def test_a_window_off_the_candle_grid_is_moved_forward_not_back():
    """A forecaster committing "BTC in one hour" at 14:23 asks for a window nothing can score.

    Measured against the live service: `commit` accepted 14:23→15:23 without a word, and `judge`
    after the close returned `no 1h candles between …` — a dead end reached only once both fees were
    spent and the window could no longer be changed. Outcomes come from hourly candles, so the close
    has to be on the hour.

    Forward, never back: snapping back could put the close in the past and turn an honest commitment
    into a backdated one, which is the single thing this ledger exists to make impossible.
    """
    from services.accountability import _snap_to_candle_grid

    assert _snap_to_candle_grid("2026-07-28T14:23:00Z") == ("2026-07-28T15:00:00Z", True)
    assert _snap_to_candle_grid("2026-07-28T23:45:00Z") == ("2026-07-29T00:00:00Z", True)
    assert _snap_to_candle_grid("2026-07-28T00:00:01Z") == ("2026-07-28T01:00:00Z", True)
    # already on the grid: untouched, and reported as untouched so no note is added
    assert _snap_to_candle_grid("2026-07-28T12:00:00Z") == ("2026-07-28T12:00:00Z", False)


def test_snapping_never_moves_a_close_into_the_past():
    from datetime import datetime, timedelta, timezone

    from services.accountability import _snap_to_candle_grid

    now = datetime.now(timezone.utc)
    for minutes in (1, 7, 31, 59, 61, 179):
        asked = (now + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
        effective, _ = _snap_to_candle_grid(asked)
        assert effective >= asked, f"{asked} snapped backwards to {effective}"
