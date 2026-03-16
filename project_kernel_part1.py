from __future__ import annotations

import numpy as np

from project_kernel_common import (
    BASE_AGENTS,
    BASE_M,
    BASE_N,
    FIT_GRID_SIZE,
    PART1_ITERS,
    PART1_SCALING_ITERS,
    RESULTS_DIR,
    SEED,
    ConsensusProblem,
    RunResult,
    append_run_history,
    average_optimality_gap,
    build_consensus_problem,
    build_edges,
    consensus_error,
    directed_ring_weights,
    edge_map,
    finalize_run,
    global_objective,
    incidence_matrix,
    initialize_run_history,
    lazy_metropolis_weights,
    local_gradients,
    local_smoothness,
    plot_gap_curves,
    predict_consensus,
    save_figure,
    single_local_gradient,
)


def run_dgd(
    problem: ConsensusProblem,
    weights: np.ndarray,
    step_size: float,
    num_iters: int,
) -> RunResult:
    alphas = np.zeros((problem.num_agents, problem.dim), dtype=float)
    history = initialize_run_history(problem, alphas)
    for _ in range(num_iters):
        alphas = weights @ alphas - step_size * local_gradients(problem, alphas)
        append_run_history(history, problem, alphas)
    return finalize_run(problem, alphas, history)


def run_gradient_tracking(
    problem: ConsensusProblem,
    weights: np.ndarray,
    step_size: float,
    num_iters: int,
) -> RunResult:
    alphas = np.zeros((problem.num_agents, problem.dim), dtype=float)
    trackers = local_gradients(problem, alphas)
    history = initialize_run_history(problem, alphas)

    for _ in range(num_iters):
        previous_gradients = local_gradients(problem, alphas)
        alphas = weights @ alphas - step_size * trackers
        trackers = weights @ trackers + local_gradients(problem, alphas) - previous_gradients
        append_run_history(history, problem, alphas)

    return finalize_run(problem, alphas, history)


def compute_dual_step(problem: ConsensusProblem, incidence: np.ndarray) -> float:
    size = problem.num_agents * problem.dim
    inverse = np.zeros((size, size), dtype=float)
    for agent_id, block in enumerate(problem.local_inverses):
        start = agent_id * problem.dim
        inverse[start : start + problem.dim, start : start + problem.dim] = block

    lifted_incidence = np.kron(incidence, np.eye(problem.dim))
    smoothness = np.linalg.eigvalsh(lifted_incidence @ inverse @ lifted_incidence.T).max()
    return 0.9 / float(smoothness)


def run_dual_decomposition(
    problem: ConsensusProblem,
    incidence: np.ndarray,
    step_size: float,
    num_iters: int,
) -> RunResult:
    multipliers = np.zeros((incidence.shape[0], problem.dim), dtype=float)
    coefficients = incidence.T @ multipliers
    alphas = np.vstack(
        [
            problem.local_inverses[agent_id]
            @ (problem.local_linear_terms[agent_id] - coefficients[agent_id])
            for agent_id in range(problem.num_agents)
        ]
    )
    history = initialize_run_history(problem, alphas)

    for _ in range(num_iters):
        multipliers += step_size * (incidence @ alphas)
        coefficients = incidence.T @ multipliers
        alphas = np.vstack(
            [
                problem.local_inverses[agent_id]
                @ (problem.local_linear_terms[agent_id] - coefficients[agent_id])
                for agent_id in range(problem.num_agents)
            ]
        )
        append_run_history(history, problem, alphas)

    return finalize_run(problem, alphas, history)


def run_admm(
    problem: ConsensusProblem,
    edges: list[tuple[int, int]],
    rho: float,
    num_iters: int,
) -> RunResult:
    mapping = edge_map(edges, problem.num_agents)
    identity = np.eye(problem.dim)
    alphas = np.zeros((problem.num_agents, problem.dim), dtype=float)
    edge_variables = np.zeros((len(edges), problem.dim), dtype=float)
    scaled_duals = np.zeros((len(edges), 2, problem.dim), dtype=float)
    history = initialize_run_history(problem, alphas)

    for _ in range(num_iters):
        updated_alphas = np.zeros_like(alphas)
        for agent_id in range(problem.num_agents):
            system = problem.local_hessians[agent_id] + rho * len(mapping[agent_id]) * identity
            rhs = problem.local_linear_terms[agent_id].copy()
            for edge_id, slot in mapping[agent_id]:
                rhs += rho * (edge_variables[edge_id] - scaled_duals[edge_id, slot])
            updated_alphas[agent_id] = np.linalg.solve(system, rhs)

        updated_edge_variables = np.zeros_like(edge_variables)
        for edge_id, (left, right) in enumerate(edges):
            updated_edge_variables[edge_id] = 0.5 * (
                updated_alphas[left]
                + updated_alphas[right]
                + scaled_duals[edge_id, 0]
                + scaled_duals[edge_id, 1]
            )

        for edge_id, (left, right) in enumerate(edges):
            scaled_duals[edge_id, 0] += updated_alphas[left] - updated_edge_variables[edge_id]
            scaled_duals[edge_id, 1] += updated_alphas[right] - updated_edge_variables[edge_id]

        alphas = updated_alphas
        edge_variables = updated_edge_variables
        append_run_history(history, problem, alphas)

    return finalize_run(problem, alphas, history)


def run_packet_loss_dgd(
    problem: ConsensusProblem,
    base_weights: np.ndarray,
    step_size: float,
    num_iters: int,
    drop_probability: float,
    seed: int,
) -> RunResult:
    rng = np.random.default_rng(seed)
    alphas = np.zeros((problem.num_agents, problem.dim), dtype=float)
    history = initialize_run_history(problem, alphas)
    support = base_weights > 0
    off_diagonal = ~np.eye(problem.num_agents, dtype=bool)

    for _ in range(num_iters):
        weights = base_weights.copy()
        losses = (rng.random(weights.shape) < drop_probability) & support & off_diagonal
        weights[losses] = 0.0
        np.fill_diagonal(weights, 0.0)
        np.fill_diagonal(weights, 1.0 - weights.sum(axis=1))
        alphas = weights @ alphas - step_size * local_gradients(problem, alphas)
        append_run_history(history, problem, alphas)

    return finalize_run(problem, alphas, history)


def run_async_dgd(
    problem: ConsensusProblem,
    weights: np.ndarray,
    step_size: float,
    num_iters: int,
    seed: int,
    max_delay: int = 4,
) -> RunResult:
    rng = np.random.default_rng(seed)
    alphas = np.zeros((problem.num_agents, problem.dim), dtype=float)
    history = initialize_run_history(problem, alphas)
    stale_buffer = [alphas.copy()]

    for _ in range(num_iters):
        agent_id = int(rng.integers(problem.num_agents))
        stale_alphas = stale_buffer[int(rng.integers(len(stale_buffer)))]
        updated = alphas.copy()
        gradient = single_local_gradient(problem, agent_id, alphas[agent_id])
        updated[agent_id] = weights[agent_id] @ stale_alphas - step_size * gradient
        alphas = updated
        stale_buffer.append(alphas.copy())
        if len(stale_buffer) > max_delay:
            stale_buffer.pop(0)
        append_run_history(history, problem, alphas)

    return finalize_run(problem, alphas, history)


def plot_topologies(
    dgd_results: dict[str, RunResult],
    gt_results: dict[str, RunResult],
    filename,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    iterations = np.arange(1, len(next(iter(dgd_results.values())).gaps) + 1)

    for graph_name, result in dgd_results.items():
        axes[0].loglog(iterations, np.maximum(result.gaps, 1e-14), label=graph_name.replace("_", " "))
    axes[0].set_title("DGD")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Average optimality gap")
    axes[0].legend()

    for graph_name, result in gt_results.items():
        axes[1].loglog(iterations, np.maximum(result.gaps, 1e-14), label=graph_name.replace("_", " "))
    axes[1].set_title("Gradient tracking")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()

    save_figure(fig, filename)


def plot_part1_data(problem: ConsensusProblem, filename) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = ["#2563eb", "#dc2626", "#059669", "#ea580c", "#7c3aed"]
    for color, indices in zip(colors, problem.agent_indices):
        ax.scatter(problem.x[indices], problem.y[indices], s=18, alpha=0.55, color=color)
    ax.scatter(
        problem.x_centers,
        problem.y[problem.center_indices],
        marker="x",
        s=60,
        color="#111827",
        label="Nyström centers",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Part I data split across agents")
    ax.legend()
    save_figure(fig, filename)


def plot_part1_fits(problem: ConsensusProblem, curves: list[tuple[np.ndarray, str]], filename) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), width_ratios=[1.7, 1.0])
    ax_fit, ax_delta = axes
    x_grid = np.linspace(-1.0, 1.0, FIT_GRID_SIZE)

    ax_fit.scatter(problem.x, problem.y, s=18, alpha=0.3, color="#4b5563", label="data")
    ax_fit.scatter(
        problem.x_centers,
        problem.y[problem.center_indices],
        s=45,
        marker="x",
        color="#111827",
        label="Nyström centers",
    )

    styles = [
        ("#111827", "-", 2.8),
        ("#2563eb", "--", 2.2),
        ("#dc2626", "-.", 2.2),
        ("#059669", ":", 2.4),
        ("#7c3aed", (0, (5, 1, 1, 1)), 2.2),
    ]
    predictions = {}

    for (color, linestyle, linewidth), (alpha, label) in zip(styles, curves):
        values = predict_consensus(problem, alpha, x_grid)
        predictions[label] = values
        ax_fit.plot(x_grid, values, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

    reference = predictions[curves[0][1]]
    for (color, linestyle, linewidth), (_, label) in zip(styles[1:], curves[1:]):
        ax_delta.plot(
            x_grid,
            predictions[label] - reference,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    ax_fit.set_xlabel(r"$x$")
    ax_fit.set_ylabel(r"$y$")
    ax_fit.set_title("Recovered functions")
    ax_fit.legend(loc="best")

    ax_delta.axhline(0.0, color="#111827", linewidth=1.2, alpha=0.7)
    ax_delta.set_xlabel(r"$x$")
    ax_delta.set_ylabel("Difference to centralized")
    ax_delta.set_title("Deviation from centralized fit")
    ax_delta.legend(loc="best")
    save_figure(fig, filename)


def plot_stress_tests(
    baseline: RunResult,
    directed: RunResult,
    packet_loss: RunResult,
    asynchronous: RunResult,
    filename,
) -> None:
    plot_gap_curves(
        [
            (baseline.gaps, "DGD", "-"),
            (directed.gaps, "Directed graph", "--"),
            (packet_loss.gaps, "Packet losses", "-."),
            (asynchronous.gaps, "Asynchronous updates", ":"),
        ],
        filename,
        ylabel="Average optimality gap",
        title="Communication issues can slow or break convergence",
    )


def plot_scaling(n_values: list[int], gap_results: dict[str, list[float]], filename) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for label, values in gap_results.items():
        ax.loglog(n_values, np.maximum(values, 1e-14), marker="o", label=label)
    ax.set_xlabel("n")
    ax.set_ylabel("Final average optimality gap")
    ax.set_title("Dependence on the dataset size")
    ax.legend()
    save_figure(fig, filename)


def generate_part1_results() -> dict[str, object]:
    problem = build_consensus_problem(BASE_N, BASE_M)
    primal_step = 0.8 / local_smoothness(problem)
    edges_complete = build_edges("complete", problem.num_agents)
    weights_complete = lazy_metropolis_weights(edges_complete, problem.num_agents)
    incidence_complete = incidence_matrix(edges_complete, problem.num_agents)
    dual_step = compute_dual_step(problem, incidence_complete)
    rho = 1.0

    dgd_complete = run_dgd(problem, weights_complete, primal_step, PART1_ITERS)
    gt_complete = run_gradient_tracking(problem, weights_complete, primal_step, PART1_ITERS)
    dual_complete = run_dual_decomposition(problem, incidence_complete, dual_step, PART1_ITERS)
    admm_complete = run_admm(problem, edges_complete, rho, PART1_ITERS)

    topology_results_dgd = {}
    topology_results_gt = {}
    for graph_name in ("line", "small_world", "complete"):
        edges = build_edges(graph_name, problem.num_agents)
        weights = lazy_metropolis_weights(edges, problem.num_agents)
        topology_results_dgd[graph_name] = run_dgd(problem, weights, primal_step, PART1_ITERS)
        topology_results_gt[graph_name] = run_gradient_tracking(problem, weights, primal_step, PART1_ITERS)

    directed = run_dgd(problem, directed_ring_weights(problem.num_agents), primal_step, PART1_ITERS)
    packet_loss = run_packet_loss_dgd(problem, weights_complete, primal_step, PART1_ITERS, 0.45, SEED)
    asynchronous = run_async_dgd(problem, weights_complete, primal_step, PART1_ITERS, SEED)

    scaling_n = [100, 400, 1600, 6400]
    scaling_gaps = {"DGD": [], "Gradient tracking": []}
    for n_value in scaling_n:
        scaling_problem = build_consensus_problem(
            n=n_value,
            m=int(np.ceil(np.sqrt(n_value))),
            num_agents=BASE_AGENTS,
            seed=SEED,
        )
        scaling_step = 0.8 / local_smoothness(scaling_problem)
        scaling_weights = lazy_metropolis_weights(
            build_edges("complete", scaling_problem.num_agents),
            scaling_problem.num_agents,
        )
        scaling_gaps["DGD"].append(run_dgd(scaling_problem, scaling_weights, scaling_step, PART1_SCALING_ITERS).gaps[-1])
        scaling_gaps["Gradient tracking"].append(
            run_gradient_tracking(scaling_problem, scaling_weights, scaling_step, PART1_SCALING_ITERS).gaps[-1]
        )

    plot_part1_data(problem, RESULTS_DIR / "part1_data_overview.pdf")
    plot_gap_curves(
        [(dgd_complete.gaps, "DGD", "-"), (gt_complete.gaps, "Gradient tracking", "--")],
        RESULTS_DIR / "dgd_vs_gradient_tracking.pdf",
        ylabel="Average optimality gap",
        title="Part I: DGD versus Gradient Tracking",
    )
    plot_gap_curves(
        [(gt_complete.gaps, "Gradient tracking", "-"), (dual_complete.gaps, "Dual decomposition", "--")],
        RESULTS_DIR / "gradient_tracking_vs_dual_decomposition.pdf",
        ylabel="Average optimality gap",
        title="Part I: Gradient Tracking versus Dual Decomposition",
    )
    plot_gap_curves(
        [
            (dgd_complete.gaps, "DGD", "-"),
            (gt_complete.gaps, "Gradient tracking", "--"),
            (dual_complete.gaps, "Dual decomposition", "-."),
            (admm_complete.gaps, "ADMM (rho=1)", ":"),
        ],
        RESULTS_DIR / "part1_all_algorithms.pdf",
        ylabel="Average optimality gap",
        title="Part I: all required algorithms",
    )
    plot_gap_curves(
        [(dual_complete.gaps, "Dual decomposition", "--"), (admm_complete.gaps, "ADMM (rho=1)", "-")],
        RESULTS_DIR / "admm_vs_dual_decomposition.pdf",
        ylabel="Average optimality gap",
        title="Part I: ADMM versus Dual Decomposition",
    )
    plot_topologies(topology_results_dgd, topology_results_gt, RESULTS_DIR / "graph_topology_comparison.pdf")
    plot_part1_fits(
        problem,
        [
            (problem.alpha_star, "Centralized optimum"),
            (dgd_complete.mean_alpha, "DGD"),
            (gt_complete.mean_alpha, "Gradient tracking"),
            (dual_complete.mean_alpha, "Dual decomposition"),
            (admm_complete.mean_alpha, "ADMM"),
        ],
        RESULTS_DIR / "fitted_functions.pdf",
    )
    plot_stress_tests(dgd_complete, directed, packet_loss, asynchronous, RESULTS_DIR / "part1_stress_tests.pdf")
    plot_scaling(scaling_n, scaling_gaps, RESULTS_DIR / "part1_scaling.pdf")

    summary_lines = [
        "[Part I]",
        f"primal_step = {primal_step:.8f}",
        f"dual_step = {dual_step:.8f}",
        f"admm_rho = {rho:.2f}",
        f"DGD final gap = {dgd_complete.gaps[-1]:.6e}",
        f"GT final gap = {gt_complete.gaps[-1]:.6e}",
        f"Dual decomposition final gap = {dual_complete.gaps[-1]:.6e}",
        f"ADMM final gap = {admm_complete.gaps[-1]:.6e}",
        f"Directed DGD final gap = {directed.gaps[-1]:.6e}",
        f"Packet-loss DGD final gap = {packet_loss.gaps[-1]:.6e}",
        f"Asynchronous DGD final gap = {asynchronous.gaps[-1]:.6e}",
        "Scaling final gaps:",
    ]
    summary_lines.extend(
        [
            f"n = {n_value}: DGD = {dgd_gap:.6e}, GT = {gt_gap:.6e}"
            for n_value, dgd_gap, gt_gap in zip(
                scaling_n, scaling_gaps["DGD"], scaling_gaps["Gradient tracking"]
            )
        ]
    )

    return {
        "problem": problem,
        "summary_lines": summary_lines,
        "metrics": {
            "dgd": dgd_complete,
            "gt": gt_complete,
            "dual": dual_complete,
            "admm": admm_complete,
        },
    }
