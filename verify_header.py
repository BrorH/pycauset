import pycauset as pc
import os
import struct
import time

print("Creating matrix...")
# Create a small matrix
# This might trigger a temp file creation if we force it, but save() definitely creates a file.
M = pc.zeros((10, 10), dtype="float32")
pc.save(M, "test_header.pycauset")

# Check file size
# 10x10 float32 = 100 * 4 = 400 bytes payload.
# Header = 64 bytes.
# Metadata? R1_STORAGE says metadata is in the file too?
# "Single-file container spec (header + payload offsets + metadata blocks)."
# Wait, R1_STORAGE is "completed".
# Does the current implementation ALREADY have metadata blocks?
# If so, my header addition might conflict if I didn't account for existing metadata logic?
# Let's check PersistentObject.cpp save() logic.
# But for now, let's just see what we get.

file_size = os.path.getsize("test_header.pycauset")
print(f"File size: {file_size}")

# Read header
with open("test_header.pycauset", "rb") as f:
    header = f.read(64)
    magic = header[:8]
    version = struct.unpack("<I", header[8:12])[0]
    print(f"Magic: {magic}")
    print(f"Version: {version}")

if magic != b"PYCAUSET":
    print(f"FAIL: Magic is {magic}")
    exit(1)

if version != 1:
    print(f"FAIL: Version is {version}")
    exit(1)

# Clean up
try:
    os.remove("test_header.pycauset")
except:
    pass

print("Verification successful!")
