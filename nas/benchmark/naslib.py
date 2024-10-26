import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from naslib.search_spaces import NasBench301SearchSpace
from naslib.search_spaces.core import Metric
import nasbench301

from . import Benchmark, Metrics, Result


class PerformanceModel(str, Enum):
    GIN = "gnn_gin"
    XGBOOST = "xgb"


class TimeModel(str, Enum):
    LIGHTGBM = "lgb_runtime"


# TODO: Store fields instead of directly using a dict
# will have to implement __getitem__ to abide by
# expected interface from naslib
class Api: ...


@dataclass
class NasBench301Result(Result):
    val: Metrics
    architecture: tuple = field(repr=False)


class NasBench301(Benchmark):
    def __init__(
        self,
        path: str | Path,
        model_dirname: str,
        data_filename: str,
        performance_model: str | PerformanceModel,
        time_model: str | TimeModel,
        version: str = "1.0",
        **_,
    ):
        self.path = Path(path)
        self.model_dir = self.path / f"{model_dirname}_{version}"
        self.data_path = self.path / data_filename
        self._api = self.from_path(
            self.model_dir,
            self.data_path,
            f"{performance_model}_v{version}",
            f"{time_model}_v{version}",
        )

    @staticmethod
    def from_path(
        model_dir: Path, data_path: Path, performance_model: str, time_model: str
    ) -> dict[str]:
        performance_ensemble = nasbench301.load_ensemble(model_dir / performance_model)
        time_ensenble = nasbench301.load_ensemble(model_dir / time_model)
        with data_path.open("rb") as f:
            data: dict[tuple, Any] = pickle.load(f)
        compact_representations = list(data.keys())
        models = [performance_ensemble, time_ensenble]

        return {
            "nb301_data": data,
            "nb301_arches": compact_representations,
            "nb301_model": models,
        }

    def query(self, architecture: NasBench301SearchSpace) -> NasBench301Result:
        train_time: float = architecture.query(Metric.TRAIN_TIME, dataset_api=self._api)
        val_accuracy: float = architecture.query(
            Metric.VAL_ACCURACY, dataset_api=self._api
        )
        return NasBench301Result(
            architecture=architecture.get_compact(),
            val=Metrics(accuracy=val_accuracy, time=train_time),
        )
