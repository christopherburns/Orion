#!/usr/bin/env python3
"""Report the mean entropy of policy targets in an Orion training dataset.

Cross-entropy loss between predicted and target distributions has a lower bound
equal to the entropy of the target: H(p) = -sum(p_i * log(p_i)). The trained
policy loss can never go below this floor — most of any remaining loss is the
intrinsic noise of the MCTS visit distributions, not network underfit.

This script decompresses an Orion .bin.lz4 dataset, iterates over the stored
policy distributions, and reports the entropy distribution. Compare the mean
against your training run's final policy loss to see how much headroom remains.

Uses macOS's built-in libcompression (via ctypes) so it bit-exactly matches
the encoder Swift used. No third-party dependencies needed.

Usage:
   targetentropy.py DATAFILE
"""

import ctypes
import math
import struct
import sys
from pathlib import Path


# Orion .bin format header (six little-endian UInt32s, 24 bytes)
HEADER_FMT     = "<IIIIII"
HEADER_SIZE    = struct.calcsize(HEADER_FMT)
ORION_MAGIC    = 0x4F52494E   # "ORIN" as a UInt32
FORMAT_VERSION = 1

# Apple Compression algorithm constants (from <compression.h>)
COMPRESSION_LZ4 = 0x100


def decompressAppleLZ4 (data: bytes) -> bytes:
   """Decompress a buffer produced by Apple's compression_encode_buffer +
   COMPRESSION_LZ4 by calling the same macOS framework function in reverse."""
   try:
      lib = ctypes.cdll.LoadLibrary("libcompression.dylib")
   except OSError as e:
      sys.stderr.write(f"Failed to load libcompression.dylib: {e}\n")
      sys.stderr.write("This script requires macOS.\n")
      sys.exit(1)

   # size_t compression_decode_buffer(uint8_t *dst_buffer, size_t dst_size,
   #                                  const uint8_t *src_buffer, size_t src_size,
   #                                  void *scratch_buffer,
   #                                  compression_algorithm algorithm)
   lib.compression_decode_buffer.restype = ctypes.c_size_t
   lib.compression_decode_buffer.argtypes = [
      ctypes.c_char_p, ctypes.c_size_t,
      ctypes.c_char_p, ctypes.c_size_t,
      ctypes.c_void_p,
      ctypes.c_int,
   ]

   # Try progressively larger output buffers — the Swift side does the same
   # because compression ratios for training data are unknown ahead of time.
   capacity = max(len(data) * 20, 1 << 20)
   while capacity < (1 << 32):
      buf = ctypes.create_string_buffer(capacity)
      written = lib.compression_decode_buffer(
         buf, capacity, data, len(data), None, COMPRESSION_LZ4)
      if written > 0:
         return buf.raw[:written]
      capacity *= 2
   raise RuntimeError("compression_decode_buffer returned 0 even with 4 GiB output buffer")


def percentile (sortedValues: list[float], pct: float) -> float:
   """Linear-interpolated percentile of an already-sorted list."""
   if not sortedValues:
      return float("nan")
   k = (len(sortedValues) - 1) * (pct / 100.0)
   lo = int(math.floor(k))
   hi = int(math.ceil(k))
   if lo == hi:
      return sortedValues[lo]
   return sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * (k - lo)


def main ():
   if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
      sys.stderr.write(__doc__)
      sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 1)

   path = Path(sys.argv[1])
   if not path.exists():
      sys.stderr.write(f"File not found: {path}\n")
      sys.exit(1)

   compressed = path.read_bytes()
   raw = decompressAppleLZ4(compressed)

   magic, version, stateDim, policyDim, totalGames, exampleCount = struct.unpack_from(HEADER_FMT, raw, 0)
   if magic != ORION_MAGIC:
      sys.stderr.write(f"Bad magic: 0x{magic:08X} (expected 0x{ORION_MAGIC:08X})\n")
      sys.exit(1)
   if version != FORMAT_VERSION:
      sys.stderr.write(f"Unsupported format version: {version}\n")
      sys.exit(1)

   bytesPerExample = (stateDim + policyDim + 1) * 4
   expectedSize = HEADER_SIZE + exampleCount * bytesPerExample
   if len(raw) != expectedSize:
      sys.stderr.write(f"Size mismatch: got {len(raw):,}, expected {expectedSize:,}\n")
      sys.exit(1)

   # Each example is laid out as: state(stateDim*4) | policy(policyDim*4) | value(4)
   policyOffset = stateDim * 4
   policyFmt    = f"<{policyDim}f"
   exampleStride = bytesPerExample

   # Entropy in nats: -sum(p * ln(p)) where p > 0. Loss values are also in nats
   # because MLX logSoftmax uses natural log, so the comparison is apples-to-apples.
   ln = math.log
   entropies: list[float] = [0.0] * exampleCount
   nonZeroCounts: list[int] = [0] * exampleCount

   for i in range(exampleCount):
      off = HEADER_SIZE + i * exampleStride + policyOffset
      policy = struct.unpack_from(policyFmt, raw, off)
      h = 0.0
      nz = 0
      for p in policy:
         if p > 0.0:
            h -= p * ln(p)
            nz += 1
      entropies[i] = h
      nonZeroCounts[i] = nz

   entropies.sort()
   nonZeroCounts.sort()
   mean = sum(entropies) / exampleCount
   variance = sum((h - mean) ** 2 for h in entropies) / exampleCount
   stdev = math.sqrt(variance)
   meanNonZero = sum(nonZeroCounts) / exampleCount

   print(f"File:          {path}")
   print(f"Compressed:    {len(compressed):,} bytes")
   print(f"Decompressed:  {len(raw):,} bytes")
   print(f"Games:         {totalGames:,}")
   print(f"Examples:      {exampleCount:,}")
   print(f"State dim:     {stateDim}")
   print(f"Policy dim:    {policyDim}")
   print()
   print(f"Policy entropy (nats — same units as policy loss):")
   print(f"  Mean:    {mean:.4f}")
   print(f"  Median:  {percentile(entropies, 50):.4f}")
   print(f"  Std:     {stdev:.4f}")
   print(f"  Min:     {entropies[0]:.4f}")
   print(f"  P10:     {percentile(entropies, 10):.4f}")
   print(f"  P90:     {percentile(entropies, 90):.4f}")
   print(f"  Max:     {entropies[-1]:.4f}")
   print()
   print(f"Non-zero moves per example:")
   print(f"  Mean:    {meanNonZero:.2f}")
   print(f"  Median:  {nonZeroCounts[exampleCount // 2]}")
   print(f"  Min:     {nonZeroCounts[0]}")
   print(f"  Max:     {nonZeroCounts[-1]}")
   print()
   print(f"Reference points (uniform-distribution entropies):")
   print(f"  log({policyDim:>2}) = {math.log(policyDim):.4f}  (uniform over all canonical moves)")
   print(f"  log(20) = {math.log(20):.4f}")
   print(f"  log(10) = {math.log(10):.4f}")
   print(f"  log( 5) = {math.log(5):.4f}")
   print(f"  log( 1) =  0.0000  (one-hot — MCTS strongly prefers a single move)")


if __name__ == "__main__":
   main()
