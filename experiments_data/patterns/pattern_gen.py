import os
import random

# ---------------------------
# Configuration Variables
# ---------------------------
FILE_SIZE = 1000  # exact file size in bytes
OUTPUT_FOLDER = 'experiments_data/patterns'  # folder to place the files (default: current folder)

# File names
FILE1 = 'file1_ones.bin'
FILE2 = 'file2_pattern123.bin'
FILE3 = 'file3_growing_pattern.bin'
FILE4 = 'file4_fibonacci.bin'
FILE5 = 'file5_random.bin'

file_names = [FILE1, FILE2, FILE3, FILE4, FILE5]

# ---------------------------
# Helper: Full path for a file
# ---------------------------
def full_path(filename):
    return os.path.join(OUTPUT_FOLDER, filename)

# ---------------------------
# Cleanup: Delete existing files if present
# ---------------------------
for fname in file_names:
    path = full_path(fname)
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted existing file: {path}")

# ---------------------------
# File 1: Only contains byte 1 repeated
# ---------------------------
data1 = b'\x01' * FILE_SIZE
with open(full_path(FILE1), 'wb') as f:
    f.write(data1)
print(f"Generated {FILE1}")

# ---------------------------
# File 2: Repeating pattern of bytes 1, 2, 3
# ---------------------------
pattern = bytes([1, 2, 3])
# Repeat pattern enough times then trim to exact size
data2 = (pattern * ((FILE_SIZE // len(pattern)) + 1))[:FILE_SIZE]
with open(full_path(FILE2), 'wb') as f:
    f.write(data2)
print(f"Generated {FILE2}")

# ---------------------------
# File 3: Evolving pattern that resets after reaching 255
# Pattern groups: [0], then [0, 1], then [0, 1, 2], ... up to [0, 1, ..., 255],
# then it starts again with [0], [0,1], ...
# ---------------------------
evolving = bytearray()
group = 1
while len(evolving) < FILE_SIZE:
    # Each group is a sequence from 0 to group-1.
    # When group==256, the maximum value is 255.
    group_bytes = bytes(range(group))
    evolving.extend(group_bytes)
    group += 1
    if group > 256:
        group = 1
# Trim to the exact required size
evolving = evolving[:FILE_SIZE]
with open(full_path(FILE3), 'wb') as f:
    f.write(evolving)
print(f"Generated {FILE3}")

# ---------------------------
# File 4: Fibonacci sequence in byte form with dynamic byte representation
# Start with 1 byte per number, then increase when a number cannot be represented.
# Once the representation increases, all subsequent numbers are represented with that many bytes.
# ---------------------------
fib_bytes = bytearray()
a, b = 0, 1
representation_length = 1  # Start with 1 byte
while len(fib_bytes) < FILE_SIZE:
    # Check if current Fibonacci number 'a' fits in the current number of bytes.
    max_val = (256 ** representation_length) - 1
    if a > max_val:
        representation_length += 1
        max_val = (256 ** representation_length) - 1  # not really needed but clarifies the new range

    # Convert the number to bytes (big-endian used; no preference specified)
    encoded = a.to_bytes(representation_length, byteorder='big')
    fib_bytes.extend(encoded)

    # Generate the next Fibonacci number
    a, b = b, a + b

# Trim to exact file size
fib_bytes = fib_bytes[:FILE_SIZE]
with open(full_path(FILE4), 'wb') as f:
    f.write(fib_bytes)
print(f"Generated {FILE4}")

# ---------------------------
# File 5: Random bytes (no pattern)
# ---------------------------
random_data = os.urandom(FILE_SIZE)
with open(full_path(FILE5), 'wb') as f:
    f.write(random_data)
print(f"Generated {FILE5}")

print("All files generated successfully in folder:", OUTPUT_FOLDER)
