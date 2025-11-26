"""
COMPREHENSIVE STATE SAVE/RESTORE TEST
Run this in your Jupyter notebook or Python environment where torch is available.

Copy and paste this entire file into a notebook cell.
"""

import torch
from torchrge256 import TorchRGE256

print("=" * 70)
print("COMPREHENSIVE STATE SAVE/RESTORE TEST")
print("=" * 70)

# TEST 1: Save state BEFORE generation
print("\n[TEST 1] Save BEFORE generation, restore, generate same sequence")
print("-" * 70)
rng1 = TorchRGE256(seed=123)
state1 = rng1.get_state()
a = rng1.rand((10,))
print(f"Generated 10 numbers: {a[:3].tolist()}...")

rng2 = TorchRGE256(seed=0)  # Different seed
rng2.set_state(state1)
b = rng2.rand((10,))
print(f"After restore, generated: {b[:3].tolist()}...")

if torch.allclose(a, b):
    print("✓ PASS: Sequences match")
else:
    print(f"✗ FAIL: Sequences don't match! Max diff: {(a-b).abs().max().item()}")
    print(f"  a = {a.tolist()}")
    print(f"  b = {b.tolist()}")

# TEST 2: Save state AFTER generation (should give NEXT sequence)
print("\n[TEST 2] Save AFTER generation (for continuation)")
print("-" * 70)
rng3 = TorchRGE256(seed=456)
x1 = rng3.rand((5,))
print(f"Generated first 5: {x1.tolist()}")

state_after = rng3.get_state()
x2 = rng3.rand((5,))
print(f"Generated next 5: {x2.tolist()}")

rng4 = TorchRGE256(seed=999)
rng4.set_state(state_after)
x3 = rng4.rand((5,))
print(f"After restore, generated: {x3.tolist()}")

if torch.allclose(x2, x3):
    print("✓ PASS: Continuation works correctly")
else:
    print(f"✗ FAIL: Continuation broken! Max diff: {(x2-x3).abs().max().item()}")

# TEST 3: from_state() classmethod
print("\n[TEST 3] Using from_state() classmethod")
print("-" * 70)
rng5 = TorchRGE256(seed=789)
state5 = rng5.get_state()
y1 = rng5.rand((10,))
print(f"Generated 10 numbers: {y1[:3].tolist()}...")

rng6 = TorchRGE256.from_state(state5)
y2 = rng6.rand((10,))
print(f"From from_state(), generated: {y2[:3].tolist()}...")

if torch.allclose(y1, y2):
    print("✓ PASS: from_state() works")
else:
    print(f"✗ FAIL: from_state() broken! Max diff: {(y1-y2).abs().max().item()}")

# TEST 4: State integrity check
print("\n[TEST 4] Verifying internal state is correctly restored")
print("-" * 70)
rng_a = TorchRGE256(seed=111)
state_a = rng_a.get_state()

rng_b = TorchRGE256(seed=222)
print(f"Before restore:")
print(f"  rng_a._state: {rng_a._state}")
print(f"  rng_b._state: {rng_b._state}")

rng_b.set_state(state_a)
print(f"After restore:")
print(f"  rng_a._state: {rng_a._state}")
print(f"  rng_b._state: {rng_b._state}")

if rng_a._state == rng_b._state:
    print("✓ PASS: Internal states match exactly")
else:
    print("✗ FAIL: Internal states don't match")
    for i, (a_val, b_val) in enumerate(zip(rng_a._state, rng_b._state)):
        if a_val != b_val:
            print(f"  _state[{i}]: {a_val} != {b_val}")

# TEST 5: Generate same numbers from same state
print("\n[TEST 5] Same state -> same numbers")
print("-" * 70)
rng_x = TorchRGE256(seed=555)
rng_y = TorchRGE256(seed=555)

# Both should generate identical sequences
seq_x = rng_x.rand((20,))
seq_y = rng_y.rand((20,))

if torch.allclose(seq_x, seq_y):
    print("✓ PASS: Same seed generates identical sequences")
else:
    print("✗ FAIL: Same seed gives different sequences!")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\nIf all tests show ✓ PASS, state save/restore is working correctly.")
print("If any test shows ✗ FAIL, there's a bug that needs to be fixed.")
