"""Prefixed ULIDs: sortable by creation time and readable in logs (trn_01JB..., evt_01JB...)."""

import os
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def new_id(prefix: str) -> str:
    """A new id such as 'trn_01JB7Z5K3M9Q2W4X6Y8A0C1E3G': 48 bits of time, 80 bits of randomness."""
    millis = time.time_ns() // 1_000_000
    random = int.from_bytes(os.urandom(10), "big")
    return f"{prefix}_{_encode(millis, 10)}{_encode(random, 16)}"
