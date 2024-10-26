import subprocess
from multiprocessing import Pool
from typing import Any, Iterable

EXPERIMENT_TEMPLATE = "python nas/experiment/nats_simulated_annealing.py --multirun {}"
# EXPERIMENT_TEMPLATE = "echo '{}'"

SIMPLE_OVERRIDES: dict[str, Any] = {
    "experiment_name": "sa_cifar10_zero_temperature_all",
    "seed": "'range(10)'",
    "optimiser.cooling_schedule.initial": 0,
    "benchmark.dataset": "CIFAR10",
    "benchmark.initial.manual": True,
}

SWEEP_OVERRIDES: list[tuple[str, Iterable]] = [
    ("benchmark.initial.0", range(5)),
    ("benchmark.initial.1", range(5)),
    ("benchmark.initial.2", range(5)),
    ("benchmark.initial.3", range(5)),
    ("benchmark.initial.4", range(5)),
    ("benchmark.initial.5", range(5)),
]


def combine(configs: list[tuple[str, Iterable]]) -> list[dict[str, Any]]:
    config, options = configs.pop()
    if not configs:
        return [{config: option} for option in options]
    child_combinations = combine(configs)
    return [
        {config: option} | combination
        for option in options
        for combination in child_combinations
    ]


combinations = combine(SWEEP_OVERRIDES)
combinations = [combination | SIMPLE_OVERRIDES for combination in combinations]


def execute_experiment(overrides: dict[str, Any]) -> None:
    override_strings = (f"{key}={value}" for key, value in overrides.items())
    subprocess.call(EXPERIMENT_TEMPLATE.format(" ".join(override_strings)), shell=True)  # noqa: S602


with Pool() as pool:
    pool.map(execute_experiment, combinations)
