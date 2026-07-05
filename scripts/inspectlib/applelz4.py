"""Pure-python decoder for Apple Compression-framework LZ4 output.

Apple's compression_encode_buffer(..., COMPRESSION_LZ4) emits its own frame
format: a sequence of blocks, each one of

   b"bv41" + <decodedSize:UInt32 LE> + <encodedSize:UInt32 LE> + <lz4 block payload>
   b"bv4-" + <size:UInt32 LE>        + <raw uncompressed payload>

terminated by b"bv4$". The payload of a bv41 block is a standard raw LZ4
block (token / literals / offset / match), NOT the lz4 "frame" format, so
neither gzip nor the lz4 CLI can read these files directly.

decompress(data, maxBytes) decodes block-by-block and stops early once
maxBytes of output are available — pure-python LZ4 is slow (~5 MB/s), so
early stop is what makes sampling large datasets fast.
"""

import struct


def lz4BlockDecompress (src: bytes, decodedSize: int, dst: bytearray) -> None:
   """Decode one raw LZ4 block, appending exactly decodedSize bytes to dst.

   dst carries all previously decoded output: Apple encodes in streaming mode,
   so a block's match offsets may reach back into earlier blocks' output."""
   startLen = len(dst)
   i = 0
   n = len(src)
   while i < n:
      token = src[i]; i += 1

      # Literals
      litLen = token >> 4
      if litLen == 15:
         while True:
            b = src[i]; i += 1
            litLen += b
            if b != 255: break
      dst += src[i : i + litLen]
      i += litLen

      if i >= n:
         break  # block ends after literals, no trailing match

      # Match (offset may reference bytes decoded in previous blocks)
      offset = src[i] | (src[i + 1] << 8); i += 2
      matchLen = token & 0x0F
      if matchLen == 15:
         while True:
            b = src[i]; i += 1
            matchLen += b
            if b != 255: break
      matchLen += 4
      start = len(dst) - offset
      if start < 0:
         raise ValueError(f"LZ4 match offset {offset} reaches before stream start")
      if offset >= matchLen:
         dst += dst[start : start + matchLen]  # non-overlapping fast path
      else:
         for k in range(matchLen):             # overlapping copy, byte-wise
            dst.append(dst[start + k])

   produced = len(dst) - startLen
   if produced != decodedSize:
      raise ValueError(f"LZ4 block decoded to {produced} bytes, expected {decodedSize}")


def decompress (data: bytes, maxBytes: int = None) -> bytes:
   """Decode an Apple-framed LZ4 buffer. Stops after maxBytes of output when given."""
   out = bytearray()
   pos = 0
   n = len(data)
   while pos + 4 <= n:
      magic = data[pos : pos + 4]
      if magic == b"bv4$":
         break
      elif magic == b"bv41":
         decodedSize, encodedSize = struct.unpack_from("<II", data, pos + 4)
         payload = data[pos + 12 : pos + 12 + encodedSize]
         lz4BlockDecompress(payload, decodedSize, out)
         pos += 12 + encodedSize
      elif magic == b"bv4-":
         size, = struct.unpack_from("<I", data, pos + 4)
         out += data[pos + 8 : pos + 8 + size]
         pos += 8 + size
      else:
         raise ValueError(f"Unrecognized Apple LZ4 block magic {magic!r} at offset {pos}")
      if maxBytes is not None and len(out) >= maxBytes:
         break
   return bytes(out)
