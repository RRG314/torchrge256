"""
Test suite for RGE256ctr_Torch counter-mode implementation
"""

import torch
from torchrge256 import RGE256ctr_Torch, TorchRGE256ctr


def test_basic_initialization():
    """Test basic initialization of counter-mode generator."""
    print("Testing basic initialization...")
    rng = RGE256ctr_Torch(seed=42)
    assert rng.seed == 42
    assert rng.rounds == 6
    assert rng.counter == 0
    print("✓ Basic initialization passed")


def test_reproducibility():
    """Test that same seed produces same sequence."""
    print("\nTesting reproducibility...")
    rng1 = RGE256ctr_Torch(seed=123)
    rng2 = RGE256ctr_Torch(seed=123)

    x1 = rng1.rand((10, 10))
    x2 = rng2.rand((10, 10))

    assert torch.allclose(x1, x2), "Same seed should produce same output"
    print("✓ Reproducibility test passed")


def test_different_seeds():
    """Test that different seeds produce different sequences."""
    print("\nTesting different seeds...")
    rng1 = RGE256ctr_Torch(seed=123)
    rng2 = RGE256ctr_Torch(seed=456)

    x1 = rng1.rand((10, 10))
    x2 = rng2.rand((10, 10))

    assert not torch.allclose(x1, x2), "Different seeds should produce different output"
    print("✓ Different seeds test passed")


def test_rand_range():
    """Test that rand() produces values in [0, 1)."""
    print("\nTesting rand() range...")
    rng = RGE256ctr_Torch(seed=42)
    x = rng.rand((1000,))

    assert torch.all(x >= 0.0), "All values should be >= 0"
    assert torch.all(x < 1.0), "All values should be < 1"
    print(f"  Mean: {x.mean():.4f} (expected ~0.5)")
    print(f"  Std:  {x.std():.4f}")
    print("✓ rand() range test passed")


def test_randint():
    """Test randint() generation."""
    print("\nTesting randint()...")
    rng = RGE256ctr_Torch(seed=42)
    x = rng.randint(0, 10, (100,))

    assert torch.all(x >= 0), "All values should be >= 0"
    assert torch.all(x < 10), "All values should be < 10"
    assert x.dtype == torch.int64, "Default dtype should be int64"
    print(f"  Range: [{x.min()}, {x.max()}]")
    print("✓ randint() test passed")


def test_randn():
    """Test randn() normal distribution."""
    print("\nTesting randn()...")
    rng = RGE256ctr_Torch(seed=42)
    x = rng.randn((10000,))

    mean = x.mean().item()
    std = x.std().item()

    print(f"  Mean: {mean:.4f} (expected ~0)")
    print(f"  Std:  {std:.4f} (expected ~1)")

    assert abs(mean) < 0.1, "Mean should be close to 0"
    assert abs(std - 1.0) < 0.1, "Std should be close to 1"
    print("✓ randn() test passed")


def test_state_save_restore():
    """Test state save and restore functionality."""
    print("\nTesting state save/restore...")
    rng1 = RGE256ctr_Torch(seed=42)

    # Generate some numbers
    x1 = rng1.rand((5, 5))

    # Save state
    state = rng1.get_state()

    # Generate more numbers
    x2 = rng1.rand((5, 5))

    # Restore state
    rng1.set_state(state)

    # Should generate same sequence as x2
    x3 = rng1.rand((5, 5))

    assert torch.allclose(x2, x3), "Restored state should reproduce same sequence"
    print("✓ State save/restore test passed")


def test_from_state():
    """Test creating new generator from saved state."""
    print("\nTesting from_state()...")
    rng1 = RGE256ctr_Torch(seed=42)
    x1 = rng1.rand((5, 5))

    state = rng1.get_state()

    # Create new generator from state
    rng2 = RGE256ctr_Torch.from_state(state)
    x2 = rng2.rand((5, 5))

    # Original continues
    x3 = rng1.rand((5, 5))

    assert torch.allclose(x2, x3), "New generator should continue same sequence"
    print("✓ from_state() test passed")


def test_manual_seed():
    """Test manual_seed() reseeding."""
    print("\nTesting manual_seed()...")
    rng = RGE256ctr_Torch(seed=42)
    x1 = rng.rand((10,))

    # Reseed
    rng.manual_seed(42)
    x2 = rng.rand((10,))

    assert torch.allclose(x1, x2), "manual_seed should reset to same sequence"
    print("✓ manual_seed() test passed")


def test_device_handling():
    """Test device handling (CPU only if no CUDA)."""
    print("\nTesting device handling...")
    rng = RGE256ctr_Torch(seed=42, device='cpu')
    x = rng.rand((5, 5))

    assert x.device.type == 'cpu', "Should be on CPU"

    # Test to() method
    rng.to('cpu')
    y = rng.rand((5, 5))
    assert y.device.type == 'cpu', "Should still be on CPU"

    if torch.cuda.is_available():
        print("  CUDA available, testing CUDA device...")
        rng_cuda = RGE256ctr_Torch(seed=42, device='cuda')
        z = rng_cuda.rand((5, 5))
        assert z.device.type == 'cuda', "Should be on CUDA"

        # Test cuda() method
        rng.cuda()
        w = rng.rand((5, 5))
        assert w.device.type == 'cuda', "Should be on CUDA after cuda() call"
    else:
        print("  CUDA not available, skipping CUDA tests")

    print("✓ Device handling test passed")


def test_alias():
    """Test that TorchRGE256ctr alias works."""
    print("\nTesting TorchRGE256ctr alias...")
    rng = TorchRGE256ctr(seed=42)
    x = rng.rand((5, 5))
    assert x.shape == (5, 5), "Alias should work correctly"
    print("✓ Alias test passed")


def test_counter_increment():
    """Test that counter increments correctly."""
    print("\nTesting counter increment...")
    rng = RGE256ctr_Torch(seed=42)

    initial_counter = rng.counter
    assert initial_counter == 0, "Counter should start at 0"

    # Each call to rand() will use multiple blocks
    _ = rng.rand((10,))  # Will use 2 blocks (8 + 2)

    assert rng.counter > initial_counter, "Counter should increment"
    print(f"  Counter after generating 10 values: {rng.counter}")
    print("✓ Counter increment test passed")


def test_different_shapes():
    """Test generation with different shapes."""
    print("\nTesting different shapes...")
    rng = RGE256ctr_Torch(seed=42)

    shapes = [(10,), (5, 5), (2, 3, 4), (1, 1, 1, 1)]

    for shape in shapes:
        x = rng.rand(shape)
        assert x.shape == shape, f"Shape mismatch for {shape}"

    print("✓ Different shapes test passed")


def test_repr():
    """Test string representation."""
    print("\nTesting __repr__...")
    rng = RGE256ctr_Torch(seed=42, rounds=8, domain="test")
    repr_str = repr(rng)

    assert "RGE256ctr_Torch" in repr_str
    assert "seed=42" in repr_str
    assert "rounds=8" in repr_str
    assert "domain='test'" in repr_str

    print(f"  {repr_str}")
    print("✓ __repr__ test passed")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("Running RGE256ctr_Torch Test Suite")
    print("=" * 60)

    test_functions = [
        test_basic_initialization,
        test_reproducibility,
        test_different_seeds,
        test_rand_range,
        test_randint,
        test_randn,
        test_state_save_restore,
        test_from_state,
        test_manual_seed,
        test_device_handling,
        test_alias,
        test_counter_increment,
        test_different_shapes,
        test_repr,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 All tests passed!")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
