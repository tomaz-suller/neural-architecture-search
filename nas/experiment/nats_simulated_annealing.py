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
    from pathlib import Path

    from nas.benchmark.nats_bench import NatsBenchTopology, Benchmark
    from nas.algorithm.simulated_annealing import CoolingSchedule, accept_transition

    # TODO Move parameters to configuration file
    ITERATIONS = 100
    SEED = 42
    NUMBER_EDGES = 6

    # TODO Write utils with paths
    REPO_ROOT = Path(__file__).parent.parent.parent
    NATS_PATH = REPO_ROOT / "models" / "NATS-tss-v1_0-3ffb9-simple"

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

# def search(dataset, save_path):
#     ss = NASBench201(dataset, save_path)

#     arch_i = ss.sample(n_samples=1)[0]
#     vector_i = ss.encode(arch_i)
#     top1_i = ss.get_info_from_arch(arch_i)["val-acc"]
#     iterations = 100
#     for i in range(iterations):
#         T = temperature(i, iterations)
#         print("Temperature:", T)
#         vector_j = random_action(vector_i)
#         arch_j = ss.decode(vector_j)
#         top1_j = ss.get_info_from_arch(arch_j)["val-acc"]
#         print("Top1_i:", top1_i)
#         print("Top1_j:", top1_j)
#         if P(top1_i - top1_j, T) > random.random():
#             arch_i = arch_j
#             top1_i = top1_j
#         print(f"Iteration {i+1}/{iterations} - Top1: {top1_i} - Temperature: {T}")
