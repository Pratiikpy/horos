"""The anchor payload is the only part of this system a stranger has to decode without our help.

They will have a transaction hash from the scorecard, an explorer, and the decoder published at
/verify. So the codec is tested the way they will use it: encode, then decode from a hex string with
an 0x prefix, exactly as an explorer hands it over.
"""
from __future__ import annotations

import pytest

from core.commit import (FORMAT_VERSION, MAGIC, PAYLOAD_BYTES, AnchorError, decode_payload,
                         encode_payload)

HEAD = "sha256:" + "ab" * 32


def test_payload_is_the_declared_length():
    assert len(encode_payload(HEAD, 1)) == PAYLOAD_BYTES == 46


def test_round_trip_recovers_the_head_and_the_count():
    out = decode_payload(encode_payload(HEAD, 4321))
    assert out["head"] == HEAD
    assert out["entries"] == 4321
    assert out["version"] == FORMAT_VERSION


def test_decodes_from_the_hex_string_an_explorer_shows():
    """The actual path a sceptic walks: copy the input field out of the explorer, decode it."""
    as_explorer_shows_it = "0x" + encode_payload(HEAD, 7).hex()
    assert decode_payload(as_explorer_shows_it)["head"] == HEAD
    assert decode_payload(as_explorer_shows_it[2:])["head"] == HEAD


def test_a_bare_hex_head_without_the_prefix_is_accepted():
    assert decode_payload(encode_payload("ab" * 32, 1))["head"] == HEAD


def test_the_magic_marks_it_as_ours():
    assert encode_payload(HEAD, 1).startswith(MAGIC)


def test_a_head_of_the_wrong_length_is_refused_rather_than_padded():
    """Padding would produce a valid-looking anchor pointing at a head that never existed."""
    with pytest.raises(AnchorError):
        encode_payload("sha256:" + "ab" * 16, 1)


def test_foreign_calldata_is_refused():
    with pytest.raises(AnchorError, match="magic"):
        decode_payload(b"NOTUS" + bytes(PAYLOAD_BYTES - 5))


def test_truncated_calldata_is_refused():
    with pytest.raises(AnchorError, match="46 bytes"):
        decode_payload(encode_payload(HEAD, 1)[:-3])


def test_an_unknown_format_version_is_refused_not_guessed():
    """A future format must fail loudly here rather than be misread by an old decoder."""
    raw = bytearray(encode_payload(HEAD, 1))
    raw[5] = 99
    with pytest.raises(AnchorError, match="version"):
        decode_payload(bytes(raw))


def test_entry_counts_across_the_full_uint64_range_round_trip():
    for n in (0, 1, 255, 256, 2 ** 32, 2 ** 64 - 1):
        assert decode_payload(encode_payload(HEAD, n))["entries"] == n


def test_an_out_of_range_entry_count_is_refused():
    with pytest.raises(AnchorError):
        encode_payload(HEAD, 2 ** 64)
    with pytest.raises(AnchorError):
        encode_payload(HEAD, -1)
