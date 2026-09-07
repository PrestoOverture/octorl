"""Eighteen prespecified count/hint/span buckets; no monotonicity claim."""
from dataclasses import dataclass
import random
from .core import BUG_TYPES, SOURCES, Difficulty

@dataclass(frozen=True)
class Bucket:
    count: int
    hint: str
    span: str
    @property
    def id(self) -> str:
        return f'c{self.count}-{self.hint}-{self.span}'
    def sample(self, seed: int) -> Difficulty:
        if type(seed) is not int:
            raise TypeError('Integer seed required')
        rng = random.Random(seed)
        return Difficulty(rng.choice(SOURCES), rng.choice(BUG_TYPES), self.count, self.hint, self.span)

BUCKETS = tuple(Bucket(count, hint, span) for count, span in (
    (1, 'single-function'), (2, 'single-function'), (3, 'single-function'),
    (2, 'single-file'), (3, 'single-file'), (2, 'cross-file')) for hint in ('L0','L1','L2'))
