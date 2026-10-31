from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 42
    dataset_repo: str = "takala/financial_phrasebank"
    revision: str = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
    archive_sha256: str = "0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6"
    fractions: tuple = (0.7, 0.15, 0.15)
    near_duplicate_threshold: float = 0.9
    output_dir: str = "artifacts/phase1"
    cache_dir: str = ".cache/huggingface"

    def __post_init__(self):
        if len(self.fractions) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.fractions) or not math.isclose(sum(self.fractions), 1):
            raise ValueError("fractions must be three positive values summing to one")
        if not 0 < self.near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        if len(self.revision) != 40 or any(c not in '0123456789abcdef' for c in self.revision):
            raise ValueError("revision must be an immutable commit SHA")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)
