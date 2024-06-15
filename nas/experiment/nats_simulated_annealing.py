import hydra
import numpy as np
from omegaconf import DictConfig

from nas import _REPO_ROOT
from nas.benchmark.nats_bench import CellTopology, Operation


def generate_neighbour(
    topology: CellTopology, rng: np.random.Generator
) -> CellTopology:
    topology_operations = list(topology)
    random_edge = rng.integers(0, len(topology_operations))
    topology_operations[random_edge] = Operation(rng.choice(Operation))
    return CellTopology(*topology_operations)


@hydra.main(
    version_base=None,
    config_path=str(_REPO_ROOT / "conf"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    from nas.benchmark.nats_bench import NatsBenchTopology, Dataset
    from nas.algorithm.simulated_annealing import CoolingSchedule, accept_transition

    nats_bench = NatsBenchTopology(
        _REPO_ROOT / cfg.benchmark.path,
        Dataset[cfg.benchmark.dataset],
    )
    rng = np.random.default_rng(cfg.seed)

    # Start from a random cell topology
    current_topology = CellTopology(
        *(Operation(i) for i in rng.choice(Operation, cfg.benchmark.edges_per_cell))
    )
    current_results = nats_bench.query(current_topology)

    cooling_schedule = CoolingSchedule(
        cfg.optimiser.cooling_schedule.initial,
        cfg.optimiser.number_iterations,
    )
    parameter_generator = cooling_schedule.linear()

    for i, control_parameter in enumerate(parameter_generator):
        neighbour = generate_neighbour(current_topology, rng)
        neighbour_results = nats_bench.query(neighbour)
        do_transition = accept_transition(
            neighbour_results.val.loss, current_results.val.loss, control_parameter, rng
        )
        if do_transition:
            current_topology = neighbour
            current_results = neighbour_results
        print(f"Iteration {i+1}")
        print(f"\tControl parameter {control_parameter}")
        print(f"\tValidation loss   {current_results.val.loss}")


if __name__ == "__main__":
    main()
