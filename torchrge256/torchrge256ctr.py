"""
TorchRGE256ctr: Counter-Mode RGE-256 PRNG

A counter-mode implementation of RGE-256 for PyTorch that provides
deterministic random number generation with a 64-bit counter and
simpler ARX structure optimized for batch generation.

Author: Steven Reid
ORCID: 0009-0003-9132-3410
"""

import torch
import math
import hashlib
from typing import Tuple, Union

MASK32 = (1 << 32) - 1


def rotl32(x: int, r: int) -> int:
    """Rotate a 32-bit integer left by r bits."""
    r &= 31
    x &= MASK32
    return ((x << r) & MASK32) | (x >> (32 - r))


class RGE256ctr_Torch:
    """
    RGE-256 Counter-Mode — PyTorch version
    --------------------------------------
    A counter-mode variant of RGE-256 that uses a 64-bit counter
    and simplified ARX structure for efficient batch generation.

    Features:
    - Python-int ARX core for safety and portability
    - PyTorch tensor outputs (CPU or CUDA)
    - Deterministic generation based on counter values
    - Compatible with the standard RGE-256 algorithm

    Args:
        seed: Integer seed for initialization (default: 42)
        rounds: Number of mixing rounds per block (default: 6)
        domain: Domain string for seed hashing (default: "rge256ctr")
        device: PyTorch device ('cpu', 'cuda', or torch.device)

    Example:
        >>> rng = RGE256ctr_Torch(seed=123, device='cuda')
        >>> x = rng.rand((100, 100))  # Generate uniform random matrix
        >>> y = rng.randn((50, 50))   # Generate normal random matrix
    """

    K_BASE = [
        0x9E3779B9, 0x517CC1B7, 0xC2B2AE35, 0x165667B1,
        0x85EBCA77, 0x27D4EB2F, 0xDE5FB9D7, 0x94D049BB
    ]

    R = [13, 7, 11, 17, 19, 23, 29, 31]

    def __init__(
        self,
        seed: int = 42,
        rounds: int = 6,
        domain: str = "rge256ctr",
        device: Union[str, torch.device] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.seed = int(seed)
        self.rounds = int(rounds)
        self.domain = domain

        self.key = [0] * 8
        self.kmix = [0] * 8
        self.counter = 0

        self._init_from_seed(seed)

    def _init_from_seed(self, seed: int) -> None:
        """
        Initialize the generator state from a seed value.

        Uses SHA-512 to derive 8 x 32-bit key words from the seed
        and domain string. Also initializes the mixing constants.

        Args:
            seed: Integer seed value
        """
        seed_bytes = (
            self.domain.encode("utf-8") +
            int(seed).to_bytes(8, "big", signed=True)
        )
        h = hashlib.sha512(seed_bytes).digest()

        words = []
        for i in range(8):
            w = int.from_bytes(h[i*4:(i+1)*4], "little") & MASK32
            words.append(w)

        self.key = words
        self.kmix = [(self.K_BASE[i] ^ words[i]) & MASK32 for i in range(8)]
        self.counter = 0

    def _block(self) -> list:
        """
        Generate one block of 8 x 32-bit random values.

        Uses the current counter value and applies the ARX mixing
        function for the specified number of rounds.

        Returns:
            List of 8 x 32-bit unsigned integers
        """
        s = self.key.copy()

        ctr = self.counter & ((1 << 64) - 1)
        ctr_lo = ctr & MASK32
        ctr_hi = (ctr >> 32) & MASK32

        s[0] = (s[0] + ctr_lo) & MASK32
        s[1] = (s[1] + ctr_hi) & MASK32

        for _ in range(self.rounds):
            # Add-rotate-xor pairs
            for i in range(0, 8, 2):
                s[i] = (s[i] + self.kmix[i]) & MASK32
                s[i+1] ^= rotl32(s[i], self.R[i])

            # Cross-lane mixing
            s[1] ^= rotl32(s[5], 13)
            s[3] ^= rotl32(s[7], 7)
            s[5] ^= rotl32(s[1], 11)
            s[7] ^= rotl32(s[3], 17)

            # Additional diffusion
            s[0] ^= rotl32(s[4], 13)
            s[1] ^= rotl32(s[5], 19)
            s[2] ^= rotl32(s[6], 7)
            s[3] ^= rotl32(s[7], 23)

        self.counter = (self.counter + 1) & ((1 << 64) - 1)
        return s

    def random_uint32(self, n: int) -> torch.Tensor:
        """
        Generate n random 32-bit unsigned integers.

        Args:
            n: Number of random integers to generate

        Returns:
            PyTorch tensor of shape (n,) with dtype int64
        """
        out = []
        needed = int(n)

        while needed > 0:
            blk = self._block()
            take = min(8, needed)
            out.extend(blk[:take])
            needed -= take

        return torch.tensor(out, dtype=torch.int64, device=self.device)

    def rand(
        self,
        shape: Union[int, Tuple[int, ...]],
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Generate random numbers from uniform distribution [0, 1).

        Args:
            shape: Shape of output tensor (int or tuple)
            dtype: PyTorch dtype for output (default: torch.float32)

        Returns:
            PyTorch tensor with random values in [0, 1)

        Example:
            >>> rng = RGE256ctr_Torch(seed=42)
            >>> x = rng.rand((3, 3))
        """
        if isinstance(shape, int):
            shape = (shape,)
        numel = math.prod(shape)
        u = self.random_uint32(numel).to(torch.float32) / (2**32)
        return u.reshape(shape).to(dtype)

    def randint(
        self,
        low: int,
        high: int,
        shape: Union[int, Tuple[int, ...]],
        dtype: torch.dtype = torch.int64
    ) -> torch.Tensor:
        """
        Generate random integers in range [low, high).

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)
            shape: Shape of output tensor
            dtype: PyTorch dtype for output (default: torch.int64)

        Returns:
            PyTorch tensor with random integers in [low, high)

        Example:
            >>> rng = RGE256ctr_Torch(seed=42)
            >>> x = rng.randint(0, 10, (5, 5))
        """
        if isinstance(shape, int):
            shape = (shape,)
        numel = math.prod(shape)
        u = self.random_uint32(numel)
        return (low + (u % (high - low))).reshape(shape).to(dtype)

    def randn(
        self,
        shape: Union[int, Tuple[int, ...]],
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Generate random numbers from standard normal distribution.

        Uses Box-Muller transform to generate normally distributed
        random values from uniform random values.

        Args:
            shape: Shape of output tensor
            dtype: PyTorch dtype for output (default: torch.float32)

        Returns:
            PyTorch tensor with standard normal random values

        Example:
            >>> rng = RGE256ctr_Torch(seed=42)
            >>> x = rng.randn((100, 100))
        """
        if isinstance(shape, int):
            shape = (shape,)
        u1 = self.rand(shape, dtype=torch.float32)
        u2 = self.rand(shape, dtype=torch.float32)
        u1 = torch.clamp(u1, min=1e-12)
        z = torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * math.pi * u2)
        return z.to(dtype)

    def manual_seed(self, seed: int) -> 'RGE256ctr_Torch':
        """
        Reset the generator with a new seed.

        Args:
            seed: New seed value

        Returns:
            Self for method chaining
        """
        self._init_from_seed(seed)
        return self

    def to(self, device: Union[str, torch.device]) -> 'RGE256ctr_Torch':
        """
        Move the generator to a different device.

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)

        Returns:
            Self for method chaining
        """
        self.device = torch.device(device)
        return self

    def cuda(self, device: int = 0) -> 'RGE256ctr_Torch':
        """
        Move the generator to CUDA device.

        Args:
            device: CUDA device index (default: 0)

        Returns:
            Self for method chaining
        """
        return self.to(f"cuda:{device}")

    def cpu(self) -> 'RGE256ctr_Torch':
        """
        Move the generator to CPU.

        Returns:
            Self for method chaining
        """
        return self.to("cpu")

    def get_state(self) -> dict:
        """
        Get the current state of the generator.

        Returns:
            Dictionary containing all state information
        """
        return {
            'seed': self.seed,
            'rounds': self.rounds,
            'domain': self.domain,
            'key': self.key.copy(),
            'kmix': self.kmix.copy(),
            'counter': self.counter,
            'device': str(self.device)
        }

    def set_state(self, state: dict) -> 'RGE256ctr_Torch':
        """
        Restore the generator state.

        Args:
            state: State dictionary from get_state()

        Returns:
            Self for method chaining
        """
        self.seed = state['seed']
        self.rounds = state['rounds']
        self.domain = state['domain']
        self.key = state['key'].copy()
        self.kmix = state['kmix'].copy()
        self.counter = state['counter']
        self.device = torch.device(state.get('device', 'cpu'))
        return self

    @classmethod
    def from_state(cls, state: dict) -> 'RGE256ctr_Torch':
        """
        Create a new generator from a saved state.

        Args:
            state: State dictionary from get_state()

        Returns:
            New RGE256ctr_Torch instance with restored state
        """
        instance = cls(seed=0)
        instance.set_state(state)
        return instance

    def __repr__(self) -> str:
        return (f"RGE256ctr_Torch(seed={self.seed}, rounds={self.rounds}, "
                f"domain='{self.domain}', device='{self.device}')")


# Alias for consistency with main module naming
TorchRGE256ctr = RGE256ctr_Torch
