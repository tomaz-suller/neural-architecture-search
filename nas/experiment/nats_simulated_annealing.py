import logging

import hydra
from loguru import logger
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


def generate_neighbour(
    topology: CellTopology, rng: np.random.Generator
) -> CellTopology:
    topology_operations = list(topology)
    random_edge = rng.integers(0, len(topology_operations))
    topology_operations[random_edge] = Operation(rng.choice(Operation))
    return CellTopology(*topology_operations)


@hydra.main(
    version_base=None,
    config_path=str(_REPO_ROOT / "config"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    from nas.benchmark.nats_bench import NatsBenchTopology, Dataset
    from nas.algorithm.simulated_annealing import CoolingSchedule, accept_transition

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
    current_results = nats_bench.query(current_topology)

    _, parameter_generator = CoolingSchedule.from_config(cfg.optimiser.cooling_schedule)

    logger.info("Starting optimisation run")
    for i, control_parameter in enumerate(parameter_generator):
        neighbour = generate_neighbour(current_topology, rng)
        logger.debug("Candidate topology is '{}'", neighbour)
        neighbour_results = nats_bench.query(neighbour)
        do_transition = accept_transition(
            neighbour_results.val.loss, current_results.val.loss, control_parameter, rng
        )
        if do_transition:
            logger.debug("Moving to candidate topology")
            current_topology = neighbour
            current_results = neighbour_results
        else:
            logger.debug("Staying in the same topology")

        logger.info("Iteration {}", i + 1)
        logger.info("    Control parameter    {}", control_parameter)
        logger.info("    Validation loss      {}", current_results.val.loss)

    logger.success("Optimisation run concluded")
    logger.info("Optimisation result")
    logger.info("    Topology           '{}'", current_topology)
    logger.info("    Validation loss    {}", current_results.val.loss)


if __name__ == "__main__":
    main()
