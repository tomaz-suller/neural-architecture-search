from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np
from omegaconf import DictConfig


@dataclass
class CoolingSchedule:
    initial: float
    length: int | None = None
    min_: float | None = None
    current: float = field(init=False)

    def __post_init__(self):
        self.current = self.initial

    @staticmethod
    def from_config(cfg: DictConfig) -> tuple["CoolingSchedule", Iterator[float]]:
        schedule = CoolingSchedule(
            cfg.initial,
            cfg.get("length", None),
            cfg.get("min", None),
        )
        if cfg.type == "linear":
            generator = schedule.linear()
        elif cfg.type == "exponential":
            generator = schedule.exponential(cfg.decay_rate)
        return schedule, generator

    def _exponential_update(self, decay_rate: float, **_):
        self.current *= decay_rate

    def _linear_update(self, iteration: int, **_):
        if self.length is None:
            raise ValueError("Linear update requires specifying length")
        self.current = 1 - iteration / self.length

    def exponential(self, decay_rate: float):
        return self._yield_parameter(
            self._exponential_update,
            decay_rate=decay_rate,
        )

    def linear(self):
        return self._yield_parameter(
            self._linear_update,
        )

    def _yield_parameter(self, parameter_update, **update_params):
        if self.length is None and self.min_ is None:
            raise ValueError("One of 'length' or 'min' must be specified")
        i = 0
        while (self.length is None) or (i < self.length):
            # self.current
            yield self.current
            update_params["iteration"] = i + 1
            parameter_update(**update_params)
            i += 1
            if (self.min_ is not None) and (self.current < self.min_):
                break


def accept_transition(
    new_value: float,
    old_value: float,
    control_parameter: float,
    rng: np.random.Generator,
    direction: Literal["max", "min"] = "max",
) -> bool:
    difference = new_value - old_value
    difference *= -1 if direction == "max" else 1
    if difference < 0:
        return True
    if control_parameter == 0:
        return False
    acceptance_probability = np.exp(-difference / control_parameter)
    return acceptance_probability > rng.random()
