from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import nats_bench
import nats_bench.api_topology
import nats_bench.api_utils

from . import Benchmark, Metrics, Result


class Set(str, Enum):
    TRAIN = "train"
    VAL = "valid"
    TEST = "test"
    VAL_TEST = "valtest"


class Dataset(str, Enum):
    CIFAR10 = "cifar10-valid"
    CIFAR10_VAL = "cifar10"
    CIFAR100 = "cifar100"
    IMAGENET = "ImageNet16-120"


class Operation(int, Enum):
    none = 0
    skip_connect = 1
    nor_conv_1x1 = 2
    nor_conv_3x3 = 3
    avg_pool_3x3 = 4

    def __str__(self) -> str:
        return self.name


@dataclass
class CellTopology:
    edge_0_to_1: Operation
    edge_0_to_2: Operation
    edge_1_to_2: Operation
    edge_0_to_3: Operation
    edge_1_to_3: Operation
    edge_2_to_3: Operation

    def __str__(self) -> str:
        return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*self)

    def __iter__(self):
        yield from (
            self.edge_0_to_1,
            self.edge_0_to_2,
            self.edge_1_to_2,
            self.edge_0_to_3,
            self.edge_1_to_3,
            self.edge_2_to_3,
        )

    def __hash__(self) -> int:
        return hash((operation for operation in self))

    @staticmethod
    def number_operations() -> int:
        return 6


@dataclass
class ArchitectureResult(Result):
    index: int
    train: Metrics
    val: Metrics | None
    test: Metrics
    flops: float
    size_parameters: float
    latency: float
    architecture: CellTopology = field(repr=False)


class NatsBenchTopology(Benchmark):
    _api: nats_bench.api_topology.NATStopology
    _dataset: Dataset

    def __init__(
        self,
        path: Path | None,
        dataset: str | Dataset,
        eager: bool = False,
        verbose: bool = False,
        **_,
    ):
        self._api = nats_bench.create(str(path), "topology", not eager, verbose)
        self._dataset = dataset if isinstance(dataset, Dataset) else Dataset[dataset]

    def query(
        self,
        topology: CellTopology,
        epoch: int | None = None,
        max_epochs: Literal["01", "12", "90", "200"] = "12",
    ) -> ArchitectureResult:
        if epoch is not None and int(max_epochs) < epoch:
            raise ValueError(
                f"Cannot query epoch '{epoch}': trial ends in epoch '{max_epochs}'"
            )

        index = self._api.query_index_by_arch(str(topology))
        info = self._api.get_more_info(
            index,
            self._dataset,
            epoch,
            max_epochs,
            is_random=False,
        )
        api_results: nats_bench.api_utils.ArchResults = self._api.query_by_index(
            index, hp=max_epochs
        )
        computational_cost = api_results.get_compute_costs(self._dataset)

        return ArchitectureResult(
            architecture=topology,
            index=index,
            train=self._info_to_metrics(info, Set.TRAIN),
            val=self._info_to_metrics(info, Set.VAL),
            test=self._info_to_metrics(info, Set.TEST),
            flops=computational_cost["flops"],
            size_parameters=computational_cost["params"],
            latency=computational_cost["latency"],
        )

    @staticmethod
    def _info_to_metrics(info: dict[str, float], set_: Set) -> Metrics:
        set_name = set_.value
        return Metrics(
            loss=info.get(f"{set_name}-loss", None),
            accuracy=info.get(f"{set_name}-accuracy", None),
            time_per_epoch=info.get(f"{set_name}-per-time", None),
            time=info.get(f"{set_name}-all-time", None),
        )
