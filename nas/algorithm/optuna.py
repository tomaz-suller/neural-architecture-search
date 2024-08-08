from dataclasses import dataclass, field

import optuna

from . import TrialOptimiser


@dataclass(kw_only=True)
class OptunaOptimiser(TrialOptimiser):
    sampler: optuna.samplers.BaseSampler
    trial: optuna.Trial = field(init=False)
    study: optuna.Study = field(init=False)

    def __post_init__(self):
        direction = (
            optuna.study.StudyDirection.MAXIMIZE
            if self.maximise
            else optuna.study.StudyDirection.MINIMIZE
        )
        self.study = optuna.create_study(sampler=self.sampler, direction=direction)

    def _update_state(self):
        self.study.tell(self.trial, self.evaluator(self.candidate))
