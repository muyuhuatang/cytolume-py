"""Binary tile (.bin) reader/writer — byte-for-byte the Cytolume tile format.

The TypeScript reader in the viewer (src/data/bin-tile.ts) MUST match this
exactly (little-endian DataView). Quantization happens on write, dequantization
on read.

HEADER (little-endian):
    magic      u32 = ascii 'CYTL' (bytes 0x43 0x59 0x54 0x4C)
    version    u16 = 1
    dims       u8  (2 or 3)
    flags      u8  = 0
    pointCount u32
    bboxMin    float32 x dims
    bboxMax    float32 x dims
BODY (each a contiguous typed array of length pointCount, in this order):
    positions int16  x dims x pointCount  (interleaved per point)
    cell_id   uint32 x pointCount
    category  int16  x pointCount

QUANTIZATION (per dim d, with min_d/max_d == the tile's own bbox):
    q = clamp(round((v-min_d)/(max_d-min_d)*65535) - 32768, -32768, 32767)
    v = min_d + (q+32768)/65535*(max_d-min_d)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC: int = int.from_bytes(b"CYTL", "little")
VERSION: int = 1
_QSPAN: int = 65535
_QOFFSET: int = 32768

# magic u32, version u16, dims u8, flags u8, pointCount u32
_HEADER_FIXED = struct.Struct("<IHBBI")


def _quantize(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = (hi - lo).astype(np.float64)
    span_safe = np.where(span == 0.0, 1.0, span)
    norm = (values.astype(np.float64) - lo) / span_safe
    q = np.rint(norm * _QSPAN) - _QOFFSET
    return np.clip(q, -32768, 32767).astype(np.int16)


def _dequantize(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = (hi - lo).astype(np.float64)
    v = lo + (q.astype(np.float64) + _QOFFSET) / _QSPAN * span
    return v.astype(np.float32)


def write_tile(path, dims: int, positions, cell_ids, categories) -> dict:
    """Write one tile. Returns {'count', 'bboxMin', 'bboxMax'} (the tile's own bbox)."""
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims}")
    path = Path(path)
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != dims:
        raise ValueError(f"positions must be (pointCount, {dims}), got {positions.shape}")
    count = positions.shape[0]
    cell_ids = np.asarray(cell_ids, dtype=np.uint32)
    categories = np.asarray(categories, dtype=np.int16)
    if cell_ids.shape != (count,) or categories.shape != (count,):
        raise ValueError("cell_ids and categories must each be (pointCount,)")

    if count > 0:
        lo = positions.min(axis=0).astype(np.float32)
        hi = positions.max(axis=0).astype(np.float32)
    else:
        lo = np.zeros(dims, dtype=np.float32)
        hi = np.zeros(dims, dtype=np.float32)

    q = _quantize(positions, lo, hi)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_HEADER_FIXED.pack(MAGIC, VERSION, dims, 0, count))
        f.write(lo.astype("<f4").tobytes())
        f.write(hi.astype("<f4").tobytes())
        f.write(np.ascontiguousarray(q).astype("<i2").tobytes())
        f.write(np.ascontiguousarray(cell_ids).astype("<u4").tobytes())
        f.write(np.ascontiguousarray(categories).astype("<i2").tobytes())

    return {"count": count, "bboxMin": lo.tolist(), "bboxMax": hi.tolist()}


def read_tile(path) -> dict:
    """Read a tile back, dequantizing positions.

    Returns dict: dims, flags, count, bboxMin, bboxMax, positions (float32 (count, dims)),
    cell_ids (uint32 (count,)), categories (int16 (count,)).
    """
    buf = Path(path).read_bytes()

    magic, version, dims, flags, count = _HEADER_FIXED.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X}, expected 0x{MAGIC:08X}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    if dims not in (2, 3):
        raise ValueError(f"bad dims {dims}")

    off = _HEADER_FIXED.size
    lo = np.frombuffer(buf, dtype="<f4", count=dims, offset=off).astype(np.float32)
    off += dims * 4
    hi = np.frombuffer(buf, dtype="<f4", count=dims, offset=off).astype(np.float32)
    off += dims * 4
    q = np.frombuffer(buf, dtype="<i2", count=count * dims, offset=off).reshape(count, dims)
    off += count * dims * 2
    cell_ids = np.frombuffer(buf, dtype="<u4", count=count, offset=off).copy()
    off += count * 4
    categories = np.frombuffer(buf, dtype="<i2", count=count, offset=off).copy()

    return {
        "dims": dims,
        "flags": flags,
        "count": count,
        "bboxMin": lo,
        "bboxMax": hi,
        "positions": _dequantize(q, lo, hi),
        "cell_ids": cell_ids,
        "categories": categories,
    }
