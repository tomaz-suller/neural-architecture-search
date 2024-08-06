from dataclasses import asdict
from functools import cache
import logging
import pickle
from pathlib import Path

import hydra
from loguru import logger
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf

from nas import _REPO_ROOT
from nas.benchmark.nats_bench import CellTopology, Operation


# Add logging handler to loguru for Hydra compatibility
# https://github.com/facebookresearch/hydra/issues/2735#issuecomment-1821529977
class PropagateHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


# TODO Figure out how to remove hydra logs from stdout rather than loguru's
logger.remove()  # Prevent duplicate logs in stdout
logger.add(PropagateHandler(), format="{message}")


def flatten(container: dict) -> dict:
    flat_container = {}
    for key, value in container.items():
        if not isinstance(value, (dict, list)):
            flat_container[key] = value
            continue
        if isinstance(value, dict):
            replacement_dict = flatten(value)
        elif isinstance(value, list):
            replacement_dict = {i: value_i for i, value_i in enumerate(value)}
        for nested_key, nested_value in replacement_dict.items():
            flat_container[f"{key}__{nested_key}"] = nested_value
    return flat_container


@hydra.main(
    version_base=None,
    config_path=str(_REPO_ROOT / "config"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    from nas.benchmark.nats_bench import NatsBenchTopology, Dataset
    from nas.algorithm import TrialOptimiser
    from nas.algorithm.simulated_annealing import SimulatedAnnealing, CoolingSchedule

    def generate_neighbour(optimiser: TrialOptimiser) -> CellTopology:
        topology: CellTopology = optimiser.current
        topology_operations = list(topology)
        random_edge = optimiser.rng.integers(0, len(topology_operations))
        topology_operations[random_edge] = Operation(optimiser.rng.choice(Operation))
        return CellTopology(*topology_operations)

    @cache
    def nats_evaluator(topology: CellTopology) -> float:
        results = nats_bench.query(topology)
        return results.val.accuracy

    cfg.results_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger.debug(
        "Executing experiment with the following config:\n{}", OmegaConf.to_yaml(cfg)
    )

    nats_bench = NatsBenchTopology(
        _REPO_ROOT / cfg.benchmark.path,
        Dataset[cfg.benchmark.dataset],
    )
    logger.success("Loaded NATS-Bench Topology")

    rng = np.random.default_rng(cfg.seed)

    # Start from a random cell topology
    current_topology = CellTopology(
        *(Operation(i) for i in rng.choice(Operation, cfg.benchmark.edges_per_cell))
    )
    logger.info("Initial topology is '{}'", current_topology)

    _, parameter_generator = CoolingSchedule.from_config(cfg.optimiser.cooling_schedule)
    optimiser: TrialOptimiser = SimulatedAnnealing(
        maximise=True,  # optimising accuracy
        rng=rng,
        evaluator=nats_evaluator,
        current=current_topology,
        candidate_generator=generate_neighbour,
        cooling_schedule=parameter_generator,
    )

    logger.info("Starting optimisation run")
    optimisation_metrics: list[dict[str, float]] = []
    for i in range(cfg.optimiser.number_iterations):
        optimiser.step()

        logger.debug("Logging optimisation metrics")
        control_parameter = optimiser._control_parameter
        do_transition = optimiser._accept_transition()
        current_results = nats_bench.query(optimiser.current)
        optimisation_metrics.append(
            {
                "control_parameter": control_parameter,
                "transition": float(do_transition),
                **asdict(current_results.val),
            }
        )

        logger.debug("Candidate topology is '{}'", optimiser.candidate)
        logger.debug(
            "Moving to candidate topology"
            if do_transition
            else "Staying in the same topology"
        )

        logger.info("Iteration {}", i + 1)
        logger.debug("    Control parameter     {}", control_parameter)
        logger.debug("    Validation accuracy   {}", current_results.val.accuracy)

    logger.success("Optimisation run concluded")
    logger.info("Optimisation result")
    logger.info("    Topology               '{}'", current_topology)
    logger.info("    Validation accuracy    {}", current_results.val.accuracy)

    logger.info("Logging experiment parameters on MLFlow")
    mlflow.set_tracking_uri(_REPO_ROOT / cfg.results_base_dir / "mlruns")
    experiment = mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(flatten(OmegaConf.to_object(cfg)))

        for i, step_metrics in enumerate(optimisation_metrics):
            mlflow.log_metrics(step_metrics, i)
        # Save final optimisation results
        mlflow.log_metrics(
            {
                f"final_val_{key}": value
                for key, value in asdict(current_results.val).items()
            }
        )
        mlflow.log_metrics(
            {
                f"final_test_{key}": value
                for key, value in asdict(current_results.test).items()
            }
        )

        final_result_path = Path(cfg.results_dir) / "result.pkl"
        with final_result_path.open("wb") as f:
            pickle.dump(current_results, f)
        mlflow.log_artifact(str(final_result_path))

    logger.success("Logged parameters to experiment ID {}", experiment.experiment_id)


if __name__ == "__main__":
    main()
