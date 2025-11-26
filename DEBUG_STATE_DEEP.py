"""
DEEP DEBUGGING: State Save/Restore Issue

Run this to see EXACTLY what's happening with state save/restore.
Copy this entire script into your notebook.
"""

import torch
from torchrge256 import TorchRGE256

print("=" * 70)
print("DEEP STATE DEBUGGING")
print("=" * 70)

# Create rng1 and inspect initial state
print("\n1. Create rng1 with seed=42")
rng1 = TorchRGE256(seed=42)
print(f"   rng1._state: {rng1._state}")
print(f"   rng1._r: {rng1._r}")
print(f"   rng1._k[0:2]: {rng1._k[0:2]}")

# Save state BEFORE generation
print("\n2. Save state BEFORE generation")
state_before = rng1.get_state()
print(f"   saved state: {state_before['state']}")
print(f"   saved r: {state_before['r']}")

# Generate numbers (this will modify internal state)
print("\n3. Generate 5 numbers with rng1")
a1 = rng1.rand((5,))
print(f"   Generated: {a1.tolist()}")
print(f"   rng1._state after generation: {rng1._state}")

# Create rng2 with different seed
print("\n4. Create rng2 with seed=99")
rng2 = TorchRGE256(seed=99)
print(f"   rng2._state BEFORE restore: {rng2._state}")

# Restore state to rng2
print("\n5. Restore saved state to rng2")
rng2.set_state(state_before)
print(f"   rng2._state AFTER restore: {rng2._state}")

# Check if states match
print("\n6. Compare internal states")
print(f"   state_before['state']: {state_before['state']}")
print(f"   rng2._state:           {rng2._state}")
print(f"   States match: {state_before['state'] == rng2._state}")

# Generate with rng2
print("\n7. Generate 5 numbers with rng2 (should match a1)")
a2 = rng2.rand((5,))
print(f"   Generated: {a2.tolist()}")
print(f"   Original:  {a1.tolist()}")
print(f"   Match: {torch.allclose(a1, a2)}")

if not torch.allclose(a1, a2):
    print(f"\n   ❌ MISMATCH!")
    print(f"   Differences: {(a1 - a2).tolist()}")
    print(f"   Max diff: {(a1 - a2).abs().max().item()}")

    # More debugging
    print("\n8. Additional debugging")
    print("   Checking if _step() is the issue...")

    # Create rng3 with same seed as rng1
    rng3 = TorchRGE256(seed=42)
    print(f"   rng3._state (seed=42): {rng3._state}")
    print(f"   rng1._state originally: {state_before['state']}")
    print(f"   rng3 matches saved state: {rng3._state == state_before['state']}")

    # Generate with rng3
    a3 = rng3.rand((5,))
    print(f"   rng3 generated: {a3.tolist()}")
    print(f"   rng1 generated: {a1.tolist()}")
    print(f"   Same seed gives same output: {torch.allclose(a1, a3)}")

else:
    print("\n   ✓ SUCCESS! State save/restore works correctly")

print("\n" + "=" * 70)
