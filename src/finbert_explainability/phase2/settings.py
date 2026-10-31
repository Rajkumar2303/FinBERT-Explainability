from dataclasses import asdict, dataclass, field
import itertools
import math

from .artifacts import read_json


@dataclass(frozen=True)
class Settings:
    phase1_dir: str = "artifacts/phase1"
    cache_dir: str = ".cache/huggingface"
    seeds: list = field(default_factory=lambda: [42, 123, 2026])
    search_seed: int = 42
    max_length: int = 64
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    learning_rates: list = field(default_factory=lambda: [2e-5, 3e-5])
    epochs: list = field(default_factory=lambda: [3, 4])
    weight_decay: float = 0.01
    baseline_c: list = field(default_factory=lambda: [0.1, 1, 10])
    baseline_ngram_ranges: list = field(default_factory=lambda: [[1, 1], [1, 2]])
    baseline_max_features: int = 20000
    baseline_max_iter: int = 2000
    models: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("Exactly three distinct training seeds are required")
        if any(not isinstance(s, int) or not 0 <= s < 2**32 for s in [*self.seeds, self.search_seed]):
            raise ValueError("Seeds must be uint32")
        if self.max_length != 64 or self.per_device_batch_size != 8 or self.gradient_accumulation_steps != 2:
            raise ValueError("Full experiments require length 64, device batch 8 and accumulation 2")
        if not self.fp16:
            raise ValueError("Full CUDA experiments require FP16")
        for values in (self.learning_rates, self.epochs, self.baseline_c):
            if not values or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Search values must be positive and finite")
        if len(self.learning_rates) * len(self.epochs) > 6:
            raise ValueError("Fine-tuning search is limited to six candidates")
        if any(e != int(e) for e in self.epochs):
            raise ValueError("Epochs must be integers")
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be nonnegative and finite")
        if self.baseline_max_features < 1 or self.baseline_max_iter < 1:
            raise ValueError("Baseline feature and iteration limits must be positive")
        if not self.baseline_ngram_ranges or any(len(n) != 2 or not 1 <= n[0] <= n[1] for n in self.baseline_ngram_ranges):
            raise ValueError("Invalid n-gram ranges")
        for model in self.models.values():
            revision = model["revision"]
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("Model revisions must be immutable commit hashes")

    @classmethod
    def load(cls, path):
        return cls(**read_json(path))

    def to_dict(self):
        return asdict(self)

    def candidates(self):
        return [dict(learning_rate=lr, epochs=e) for lr, e in itertools.product(self.learning_rates, self.epochs)]
