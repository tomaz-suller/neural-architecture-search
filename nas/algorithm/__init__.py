from dataclasses import dataclass, field
from typing import Callable, Any

import numpy.random


@dataclass(kw_only=True)
class TrialOptimiser:
    maximise: bool
    rng: numpy.random.Generator
    evaluator: Callable[..., float]
    candidate_generator: Callable | None
    current: Any | None = None
    candidate: Any | None = field(init=False)

    def _generate_candidate(self):
        if self.candidate_generator is None:
            raise NotImplementedError()
        return self.candidate_generator(self)

    def _accept_transition(self) -> bool:
        return True

    def _update_state(self):
        return

    def step(self):
        self.candidate = self._generate_candidate()
        self._update_state()
        if self._accept_transition():
            self.current = self.candidate
