from dataclasses import dataclass
from enum import Enum, auto


class Result: ...


class Benchmark:
    def query(self, *args, **kwargs) -> Result: ...


@dataclass
class Metrics:
    loss: float | None = None
    accuracy: float | None = None
    time_per_epoch: float | None = None
    time: float | None = None


class SearchSpace(Enum):
    NATS_BENCH_TOPOLOGY = auto()
    NAS_BENCH_301 = auto()

    @staticmethod
    def benchmark_from_name(
        name: str, *_, **benchmark_kwargs
    ) -> tuple["SearchSpace", Benchmark]:
        try:
            search_space = SearchSpace[name]
        except KeyError:
            raise KeyError(f"Unsupported benchmark '{name}'")
        benchmark = None
        if search_space == SearchSpace.NATS_BENCH_TOPOLOGY:
            from .nats_bench import NatsBenchTopology

            benchmark = NatsBenchTopology(**benchmark_kwargs)
        elif search_space == SearchSpace.NAS_BENCH_301:
            from .naslib import NasBench301

            benchmark = NasBench301(**benchmark_kwargs)
        return search_space, benchmark
