"""
TorchRGE256: PyTorch-Native RGE-256 PRNG
"""

import math
import hashlib
import torch
import torch.nn as nn
from typing import Tuple, Union, List


class TorchRGE256(nn.Module):
    """
    PyTorch-native RGE-256 PRNG.
    
    Works on CPU or CUDA. Full state checkpointing for reproducible training.
    """
    
    DEFAULT_ZETAS = (1.585, 1.926, 1.262)
    K_BASE = [
        0x9E3779B9, 0x517CC1B7, 0xC2B2AE35, 0x165667B1,
        0x85EBCA77, 0x27D4EB2F, 0xDE5FB9D7, 0x94D049BB
    ]
    
    def __init__(
        self,
        seed: int = 42,
        rounds: int = 3,
        zetas: Tuple[float, float, float] = None,
        domain: str = "torch-rge256",
        device: Union[str, torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            device = "cpu"
        self._device = torch.device(device)
        
        self._seed = seed
        self._rounds = max(1, min(10, rounds))
        self._zetas = zetas if zetas else self.DEFAULT_ZETAS
        self._domain = domain
        
        self._state: List[int] = [0] * 8
        self._r: List[int] = [0] * 8
        self._k: List[int] = [0] * 8
        
        self._init_from_seed(seed)
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def _init_from_seed(self, seed: int) -> None:
        seed_bytes = self._domain.encode('utf-8')
        seed_bytes += abs(seed).to_bytes(8, 'big')
        h = hashlib.sha512(seed_bytes).digest()
        
        words = []
        for i in range(16):
            word = int.from_bytes(h[i*4:(i+1)*4], 'little')
            words.append(word)
        
        self._state = words[:8]
        
        tri, meng, tet = self._zetas
        base = [
            tri, meng, tet, (tri + meng + tet) / 3.0,
            tri + 0.5, meng + 0.75, tet + 0.25,
            (tri * 1.25 + tet * 0.75) / 2
        ]
        self._r = [max(1, int(abs(z * 977 + (i * 7 + 13)) % 31) or 1) 
                   for i, z in enumerate(base)]
        
        self._k = [(self.K_BASE[i] ^ words[8 + i]) & 0xFFFFFFFF 
                   for i in range(8)]
        
        for _ in range(10):
            self._step()
    
    @staticmethod
    def _rotl32(x: int, r: int) -> int:
        x = x & 0xFFFFFFFF
        r = r % 32
        return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF
    
    def _step(self) -> None:
        s = self._state
        r = self._r
        k = self._k
        rotl = self._rotl32
        
        s[0] = (s[0] + s[1] + k[0]) & 0xFFFFFFFF
        s[1] = rotl(s[1] ^ s[0], r[0])
        s[2] = (s[2] + s[3] + k[1]) & 0xFFFFFFFF
        s[3] = rotl(s[3] ^ s[2], r[1])
        
        tA0, tA1 = s[0] ^ s[2], s[1] ^ s[3]
        s[0] = (s[0] + rotl(tA0, r[2]) + k[2]) & 0xFFFFFFFF
        s[2] = (s[2] + rotl(tA1, r[3]) + k[3]) & 0xFFFFFFFF
        s[1] = s[1] ^ rotl(s[0], (r[0] + r[2]) % 31 or 1)
        s[3] = s[3] ^ rotl(s[2], (r[1] + r[3]) % 31 or 1)
        
        s[4] = (s[4] + s[5] + k[4]) & 0xFFFFFFFF
        s[5] = rotl(s[5] ^ s[4], r[4])
        s[6] = (s[6] + s[7] + k[5]) & 0xFFFFFFFF
        s[7] = rotl(s[7] ^ s[6], r[5])
        
        tB0, tB1 = s[4] ^ s[6], s[5] ^ s[7]
        s[4] = (s[4] + rotl(tB0, r[6]) + k[6]) & 0xFFFFFFFF
        s[6] = (s[6] + rotl(tB1, r[7]) + k[7]) & 0xFFFFFFFF
        s[5] = s[5] ^ rotl(s[4], (r[4] + r[6]) % 31 or 1)
        s[7] = s[7] ^ rotl(s[6], (r[5] + r[7]) % 31 or 1)
        
        s[1] = s[1] ^ rotl(s[5], 13)
        s[3] = s[3] ^ rotl(s[7], 7)
        s[5] = s[5] ^ rotl(s[1], 11)
        s[7] = s[7] ^ rotl(s[3], 17)
        
        m0, m1 = s[0] ^ s[4], s[1] ^ s[5]
        m2, m3 = s[2] ^ s[6], s[3] ^ s[7]
        
        s[0] = (s[0] + rotl(m1, 3)) & 0xFFFFFFFF
        s[1] = (s[1] + rotl(m2, 5)) & 0xFFFFFFFF
        s[2] = (s[2] + rotl(m3, 7)) & 0xFFFFFFFF
        s[3] = (s[3] + rotl(m0, 11)) & 0xFFFFFFFF
        s[4] = s[4] ^ rotl(s[0], 19)
        s[5] = s[5] ^ rotl(s[1], 23)
        s[6] = s[6] ^ rotl(s[2], 29)
        s[7] = s[7] ^ rotl(s[3], 31)
    
    def _next32(self) -> int:
        for _ in range(self._rounds):
            self._step()
        return (self._state[0] ^ self._rotl32(self._state[4], 13)) & 0xFFFFFFFF
    
    def _resolve_device(self, device) -> torch.device:
        if device is None:
            return self._device
        return torch.device(device)
    
    @staticmethod
    def _numel(shape) -> int:
        n = 1
        for s in shape:
            n *= s
        return n
    
    def rand(self, shape, *, dtype=torch.float32, device=None, requires_grad=False) -> torch.Tensor:
        device = self._resolve_device(device)
        numel = self._numel(shape)
        data = [self._next32() / 0x100000000 for _ in range(numel)]
        out = torch.tensor(data, dtype=dtype, device=device).reshape(shape)
        if requires_grad:
            out.requires_grad_(True)
        return out
    
    def randn(self, shape, *, dtype=torch.float32, device=None, requires_grad=False) -> torch.Tensor:
        device = self._resolve_device(device)
        numel = self._numel(shape)
        n_pairs = (numel + 1) // 2
        samples = []
        for _ in range(n_pairs):
            u1 = max(self._next32() / 0x100000000, 1e-10)
            u2 = self._next32() / 0x100000000
            mag = math.sqrt(-2.0 * math.log(u1))
            samples.append(mag * math.cos(2.0 * math.pi * u2))
            samples.append(mag * math.sin(2.0 * math.pi * u2))
        out = torch.tensor(samples[:numel], dtype=dtype, device=device).reshape(shape)
        if requires_grad:
            out.requires_grad_(True)
        return out
    
    def randint(self, low: int, high: int, shape, *, dtype=torch.int64, device=None) -> torch.Tensor:
        device = self._resolve_device(device)
        numel = self._numel(shape)
        range_size = high - low
        data = [low + (self._next32() % range_size) for _ in range(numel)]
        return torch.tensor(data, dtype=dtype, device=device).reshape(shape)
    
    def rand_like(self, tensor: torch.Tensor, *, dtype=None, device=None) -> torch.Tensor:
        dtype = dtype if dtype is not None else tensor.dtype
        device = device if device is not None else tensor.device
        return self.rand(tensor.shape, dtype=dtype, device=device)
    
    def randn_like(self, tensor: torch.Tensor, *, dtype=None, device=None) -> torch.Tensor:
        dtype = dtype if dtype is not None else tensor.dtype
        device = device if device is not None else tensor.device
        return self.randn(tensor.shape, dtype=dtype, device=device)
    
    def bernoulli(self, p: float, shape, *, device=None) -> torch.Tensor:
        u = self.rand(shape, dtype=torch.float32, device=device)
        return (u < p).float()
    
    def dropout_mask(self, shape, p: float = 0.5, *, device=None) -> torch.Tensor:
        device = self._resolve_device(device)
        if p <= 0.0:
            return torch.ones(shape, dtype=torch.float32, device=device)
        if p >= 1.0:
            return torch.zeros(shape, dtype=torch.float32, device=device)
        u = self.rand(shape, dtype=torch.float32, device=device)
        scale = 1.0 / (1.0 - p)
        return torch.where(u >= p, scale, 0.0)
    
    def uniform(self, low: float, high: float, shape, *, dtype=torch.float32, device=None) -> torch.Tensor:
        u = self.rand(shape, dtype=dtype, device=device)
        return low + (high - low) * u
    
    def exponential(self, rate: float, shape, *, dtype=torch.float32, device=None) -> torch.Tensor:
        u = self.rand(shape, dtype=torch.float32, device=device)
        u = torch.clamp(u, min=1e-10)
        return (-torch.log(u) / rate).to(dtype)
    
    def normal(self, mean: float, std: float, shape, *, dtype=torch.float32, device=None) -> torch.Tensor:
        z = self.randn(shape, dtype=dtype, device=device)
        return mean + std * z
    
    def shuffle(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        n = tensor.shape[dim]
        indices = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self._next32() % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        idx = torch.tensor(indices, dtype=torch.long, device=tensor.device)
        return torch.index_select(tensor, dim, idx)
    
    def choice(self, tensor: torch.Tensor, n: int, replace: bool = True, dim: int = 0) -> torch.Tensor:
        size = tensor.shape[dim]
        if replace:
            indices = [self._next32() % size for _ in range(n)]
        else:
            if n > size:
                raise ValueError(f"Cannot sample {n} from {size} without replacement")
            pool = list(range(size))
            indices = []
            for _ in range(n):
                i = self._next32() % len(pool)
                indices.append(pool.pop(i))
        idx = torch.tensor(indices, dtype=torch.long, device=tensor.device)
        return torch.index_select(tensor, dim, idx)
    
    def permutation(self, n: int, *, device=None) -> torch.Tensor:
        device = self._resolve_device(device)
        return self.shuffle(torch.arange(n, device=device))
    
    def fork(self, name: str) -> 'TorchRGE256':
        return TorchRGE256(
            seed=self._next32(),
            rounds=self._rounds,
            zetas=self._zetas,
            domain=f"{self._domain}/{name}",
            device=self._device
        )
    
    def manual_seed(self, seed: int) -> 'TorchRGE256':
        self._seed = seed
        self._init_from_seed(seed)
        return self
    
    def to(self, device: Union[str, torch.device]) -> 'TorchRGE256':
        self._device = torch.device(device)
        return self
    
    def cuda(self, device: int = 0) -> 'TorchRGE256':
        return self.to(f"cuda:{device}")
    
    def cpu(self) -> 'TorchRGE256':
        return self.to("cpu")
    
    def state_dict(self) -> dict:
        return {
            'state': list(self._state),
            'r': list(self._r),
            'k': list(self._k),
            'rounds': self._rounds,
            'zetas': self._zetas,
            'domain': self._domain,
            'device': str(self._device),
            'seed': self._seed,
        }
    
    def load_state_dict(self, state_dict: dict) -> 'TorchRGE256':
        self._state = list(state_dict['state'])
        self._r = list(state_dict['r'])
        self._k = list(state_dict['k'])
        self._rounds = state_dict['rounds']
        self._zetas = tuple(state_dict['zetas'])
        self._domain = state_dict['domain']
        self._device = torch.device(state_dict.get('device', 'cpu'))
        self._seed = state_dict['seed']
        return self
    
    def get_state(self) -> dict:
        return self.state_dict()
    
    def set_state(self, state: dict) -> 'TorchRGE256':
        return self.load_state_dict(state)

    @classmethod
    def from_state(cls, state: dict) -> 'TorchRGE256':
        """
        Create a new TorchRGE256 instance from a saved state dictionary.

        Args:
            state: State dictionary from get_state() or state_dict()

        Returns:
            New TorchRGE256 instance with restored state
        """
        # Create instance with dummy seed (will be overwritten)
        instance = cls(seed=0)
        instance.load_state_dict(state)
        return instance

    def __repr__(self) -> str:
        return f"TorchRGE256(seed={self._seed}, rounds={self._rounds}, domain='{self._domain}', device='{self._device}')"
    
    def __call__(self, shape, **kwargs) -> torch.Tensor:
        return self.rand(shape, **kwargs)
