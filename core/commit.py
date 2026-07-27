"""Anchoring the ledger head on X Layer.

The hash chain in ``ledger.py`` makes history tamper-*evident* to anyone who already holds a copy.
It does not stop us from rebuilding the whole chain from scratch and claiming that was always it.
Anchoring closes that: periodically the current head is written into the calldata of a zero-value
transaction on X Layer, and once it is in a block nobody — us least of all — can produce a different
history that hashes to the same head.

**Why calldata and not a contract.** A contract would be one more thing a reader has to trust, one
more thing to deploy, verify and keep. Calldata on a plain transfer is permanent, costs a few
hundred gas, is visible on any explorer, and needs no ABI to read. The tradeoff is that there is no
on-chain index of anchors; the ledger holds that index, and the ledger is the thing being anchored,
so it is exactly the right place for it.

**Why the same chain the marketplace settles on.** A buyer paying in USD₮0 on X Layer already has an
X Layer client. Anchoring anywhere else would mean asking them to acquire one to check us.

The 46-byte payload is deliberately readable without tooling:

    bytes  0..5   b"HOROS"          magic
    byte   5      0x01              format version
    bytes  6..38  32-byte digest    sha256 of the ledger head entry
    bytes 38..46  uint64 big-endian number of entries the head covers

Anchoring is optional at runtime. With no funded key the ledger still chains and the scorecard still
publishes — and says, in those words, that it is not yet anchored. A missing anchor is reported as a
missing anchor, never omitted so the page reads as though everything is proven.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

MAGIC = b"HOROS"
FORMAT_VERSION = 1
PAYLOAD_BYTES = 5 + 1 + 32 + 8      # 46

# Used only when the node will not give an estimate. Measured, not derived: X Layer quoted 22,966
# for this exact payload on 2026-07-27, against 21,736 from the mainnet calldata formula. The margin
# above the measurement is for a future repricing, not for a guess.
FALLBACK_GAS = 30_000


class AnchorError(RuntimeError):
    """Anchoring failed. Always surfaced, never swallowed into a silent no-op."""


@dataclass(frozen=True)
class Anchor:
    head: str
    entries: int
    tx: str
    chain: str
    explorer_url: str
    gas_used: int | None = None
    block: int | None = None


def encode_payload(head: str, entries: int) -> bytes:
    """Pack a ledger head into the 46 bytes that go on chain."""
    digest = head.split(":", 1)[-1]          # accept "sha256:<hex>" or a bare hex digest
    raw = bytes.fromhex(digest)
    if len(raw) != 32:
        raise AnchorError(f"a ledger head must be 32 bytes, got {len(raw)}")
    if not 0 <= entries < 2 ** 64:
        raise AnchorError(f"entry count out of range: {entries}")
    return MAGIC + bytes([FORMAT_VERSION]) + raw + entries.to_bytes(8, "big")


def decode_payload(data: bytes | str) -> dict:
    """Read an anchor back out of a transaction's calldata.

    Published verbatim on the verification page. Anyone can pull the input field of one of our anchor
    transactions from any X Layer explorer and run this to recover the head we committed to.
    """
    if isinstance(data, str):
        data = bytes.fromhex(data[2:] if data.startswith("0x") else data)
    if len(data) != PAYLOAD_BYTES:
        raise AnchorError(f"expected {PAYLOAD_BYTES} bytes of calldata, got {len(data)}")
    if data[:5] != MAGIC:
        raise AnchorError("not a Horos anchor: magic bytes do not match")
    version = data[5]
    if version != FORMAT_VERSION:
        raise AnchorError(f"unknown anchor format version {version}")
    return {"version": version,
            "head": "sha256:" + data[6:38].hex(),
            "entries": int.from_bytes(data[38:46], "big")}


class Anchorer:
    """Writes ledger heads to X Layer.

    The key is its own, generated for this purpose and holding nothing but gas. It is deliberately
    *not* the wallet the marketplace settles payments into and not the wallet the other four listed
    agents use: an anchoring bug must not be able to touch money, and a routine gas top-up must not
    require unsealing a wallet that four live listings depend on.
    """

    def __init__(self, rpc_url: str, chain_id: int, private_key: str,
                 explorer_prefix: str = "https://www.oklink.com/xlayer/tx/"):
        if not private_key:
            raise AnchorError("no anchoring key is configured")
        try:
            from eth_account import Account
            from web3 import Web3
        except ImportError as e:                                    # pragma: no cover
            raise AnchorError(f"web3 is not installed: {e}") from e

        self._w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        self._account = Account.from_key(private_key)
        self.chain_id = chain_id
        self.explorer_prefix = explorer_prefix

    @property
    def address(self) -> str:
        return self._account.address

    def balance_wei(self) -> int:
        return int(self._w3.eth.get_balance(self._account.address))

    def estimate_cost_wei(self) -> int:
        """What one anchor costs at the current gas price, for the funding note on the status page.

        Asks the node rather than applying the textbook formula. X Layer is a zkEVM and charges more
        for calldata than mainnet does: 21000 + 46x16 predicts 21,736, and the node's own estimate
        for the same payload is 22,966. Publishing the smaller number would understate what the
        wallet needs, and the first thing an operator would learn about that is a failed anchor.
        """
        payload = encode_payload("sha256:" + "00" * 32, 0)
        try:
            gas = int(self._w3.eth.estimate_gas(
                {"from": self._account.address, "to": self._account.address,
                 "value": 0, "data": payload}))
        except Exception:                                            # noqa: BLE001
            gas = FALLBACK_GAS
        return int(gas * 1.2) * int(self._w3.eth.gas_price)

    def anchor(self, head: str, entries: int, wait_seconds: int = 120) -> Anchor:
        """Commit one head and wait for it to be mined.

        Waiting is not optional. An anchor that was broadcast but never included proves nothing, and
        recording it as an anchor would put a claim on the scorecard that the chain does not support.
        """
        payload = encode_payload(head, entries)
        w3 = self._w3

        base = w3.eth.get_block("latest").get("baseFeePerGas")
        nonce = w3.eth.get_transaction_count(self._account.address)
        tx: dict = {
            "chainId": self.chain_id,
            "nonce": nonce,
            "to": self._account.address,        # to self: the transfer is the carrier, not the point
            "value": 0,
            "data": payload,
        }
        if base is not None:
            # X Layer serves both type-0 and type-2; type-2 verified present in live blocks.
            tip = max(int(w3.eth.max_priority_fee), 1)
            tx["maxPriorityFeePerGas"] = tip
            tx["maxFeePerGas"] = int(base) * 2 + tip
        else:                                                        # pragma: no cover
            tx["gasPrice"] = int(w3.eth.gas_price)

        try:
            tx["gas"] = int(w3.eth.estimate_gas({**tx, "from": self._account.address}) * 1.2)
        except Exception:                                            # noqa: BLE001
            tx["gas"] = FALLBACK_GAS

        cost = tx["gas"] * tx.get("maxFeePerGas", tx.get("gasPrice", 0))
        held = self.balance_wei()
        if held < cost:
            raise AnchorError(
                f"the anchoring wallet {self.address} holds {held / 1e18:.9f} OKB but this anchor "
                f"needs up to {cost / 1e18:.9f} OKB in gas. Fund it on X Layer (chain 196) to resume "
                f"anchoring; the ledger keeps chaining in the meantime.")

        signed = self._account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        try:
            tx_hash = w3.eth.send_raw_transaction(raw)
        except Exception as e:                                       # noqa: BLE001
            raise AnchorError(f"broadcast failed: {type(e).__name__}: {e}") from e

        deadline = time.time() + wait_seconds
        receipt = None
        while time.time() < deadline:
            try:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
                break
            except Exception:                                        # noqa: BLE001
                time.sleep(2)
        if receipt is None:
            raise AnchorError(
                f"anchor {tx_hash.hex()} was broadcast but not mined within {wait_seconds}s; it is "
                f"not being recorded as an anchor until it is in a block")
        if receipt.get("status") != 1:
            raise AnchorError(f"anchor transaction {tx_hash.hex()} reverted on chain")

        h = tx_hash.hex()
        if not h.startswith("0x"):
            h = "0x" + h
        return Anchor(head=head, entries=entries, tx=h, chain=f"eip155:{self.chain_id}",
                      explorer_url=self.explorer_prefix + h,
                      gas_used=int(receipt.get("gasUsed", 0)) or None,
                      block=int(receipt.get("blockNumber", 0)) or None)

    def read_back(self, tx_hash: str) -> dict:
        """Fetch one of our anchors from the chain and decode it.

        This is the verification path a sceptic walks, so it exists in the codebase and is exercised
        by a test rather than only described in prose.
        """
        tx = self._w3.eth.get_transaction(tx_hash)
        data = tx["input"]
        if hasattr(data, "hex"):
            data = data.hex()
        return decode_payload(data)
