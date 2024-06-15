from nas.benchmark.nats_bench import CellTopology, Operation

import numpy as np


def generate_neighbour(
    topology: CellTopology, rng: np.random.Generator
) -> CellTopology:
    topology_operations = list(topology)
    random_edge = rng.integers(0, len(topology_operations))
    topology_operations[random_edge] = Operation(rng.choice(Operation))
    return CellTopology(*topology_operations)


if __name__ == "__main__":
    from nas import _REPO_ROOT
    from nas.benchmark.nats_bench import NatsBenchTopology, Benchmark
    from nas.algorithm.simulated_annealing import CoolingSchedule, accept_transition

    # TODO Move parameters to configuration file
    ITERATIONS = 100
    SEED = 42
    NUMBER_EDGES = 6

    # TODO Write utils with paths
    NATS_PATH = _REPO_ROOT / "models" / "NATS-tss-v1_0-3ffb9-simple"

    nats_bench = NatsBenchTopology(NATS_PATH, Benchmark.CIFAR10)
    rng = np.random.default_rng(SEED)

    # Start from a random cell topology
    current_topology = CellTopology(
        *(Operation(i) for i in rng.choice(Operation, NUMBER_EDGES))
    )
    current_results = nats_bench.query(current_topology)

    cooling_schedule = CoolingSchedule(1, ITERATIONS)
    for i, control_parameter in enumerate(cooling_schedule.exponential(0.99)):
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
