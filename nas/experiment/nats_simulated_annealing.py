from dataclasses import asdict
import gc
import logging
import pickle
import random
from pathlib import Path

import hydra
from loguru import logger
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf

from nas import _REPO_ROOT


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
    from naslib.search_spaces import NasBench301SearchSpace

    from nas.benchmark.nats_bench import CellTopology, Operation
    from nas.benchmark import SearchSpace
    from nas.algorithm import TrialOptimiser
    from nas.algorithm.simulated_annealing import SimulatedAnnealing, CoolingSchedule

    def nats_neighbour_generator(optimiser: TrialOptimiser) -> CellTopology:
        # TODO: Force new architecture to be different
        topology: CellTopology = optimiser.current
        topology_operations = list(topology)
        random_edge = optimiser.rng.integers(0, len(topology_operations))
        topology_operations[random_edge] = Operation(optimiser.rng.choice(Operation))
        return CellTopology(*topology_operations)

    def naslib_neighbour_generator(optimiser: TrialOptimiser) -> CellTopology:
        architecture: NasBench301SearchSpace = optimiser.current
        return architecture.get_nbhd()[0].arch

    def evaluator(architecture) -> float:
        results = benchmark.query(architecture)
        return results.val.accuracy

    cfg.results_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger.debug(
        "Executing experiment with the following config:\n{}", OmegaConf.to_yaml(cfg)
    )

    # TODO Review benchmark configuration names
    search_space, benchmark = SearchSpace.benchmark_from_name(**cfg.benchmark)
    logger.success(f"Loaded {cfg.benchmark.name}")

    # Set random seeds
    rng = np.random.default_rng(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002
    random.seed(cfg.seed)

    # Start from a random cell topology
    if search_space == SearchSpace.NATS_BENCH_TOPOLOGY:
        current_architecture = CellTopology(
            *(
                Operation(i)
                for i in rng.choice(Operation, CellTopology.number_operations())
            )
        )
        neighbour_generator = nats_neighbour_generator
    elif search_space == SearchSpace.NAS_BENCH_301:
        current_architecture = NasBench301SearchSpace()
        current_architecture.sample_random_architecture()
        neighbour_generator = naslib_neighbour_generator

    logger.info("Initial architecture is '{}'", current_architecture)

    _, parameter_generator = CoolingSchedule.from_config(cfg.optimiser.cooling_schedule)
    optimiser: TrialOptimiser = SimulatedAnnealing(
        maximise=True,  # optimising accuracy
        rng=rng,
        evaluator=evaluator,
        current=current_architecture,
        candidate_generator=neighbour_generator,
        cooling_schedule=parameter_generator,
    )

    logger.info("Starting optimisation run")
    optimisation_metrics: list[dict[str, float]] = []
    for i in range(cfg.optimiser.number_iterations):
        logger.debug("Current   topology is '{}'", optimiser.current)

        optimiser.step()

        logger.debug("Candidate topology is '{}'", optimiser.candidate)

        logger.debug("Logging optimisation metrics")
        control_parameter = optimiser._control_parameter
        do_transition = optimiser.current is optimiser.candidate
        current_results = benchmark.query(optimiser.current)
        current_results_dict: dict[str, float] = {
            key: value
            for key, value in asdict(current_results.val).items()
            if value is not None
        }
        current_results_dict["control_parameter"] = control_parameter
        current_results_dict["transition"] = float(do_transition)
        optimisation_metrics.append(current_results_dict)

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
    logger.info("    Topology               '{}'", current_architecture)
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
                if value is not None
            }
        )
        # mlflow.log_metrics(
        #     {
        #         f"final_test_{key}": value
        #         for key, value in asdict(current_results.test).items()
        #     }
        # )

        final_result_path = Path(cfg.results_dir) / "result.pkl"
        with final_result_path.open("wb") as f:
            pickle.dump(current_results, f)
        mlflow.log_artifact(str(final_result_path))

    logger.success("Logged parameters to experiment ID {}", experiment.experiment_id)

    del optimiser
    del optimisation_metrics
    gc.collect()


if __name__ == "__main__":
    main()
