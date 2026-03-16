from __future__ import annotations

import numpy as np

from project_kernel_common import (
    BASE_M,
    PART2_ROUNDS,
    RESULTS_DIR,
    SEED,
    FedAvgResult,
    FederatedProblem,
    build_federated_problem,
    federated_objective,
    predict_federated,
    save_figure,
)


def federated_gradient(
    problem: FederatedProblem,
    client_id: int,
    alpha: np.ndarray,
    batch_indices: np.ndarray,
) -> np.ndarray:
    features = problem.client_features[client_id][batch_indices]
    targets = problem.client_targets[client_id][batch_indices]
    return (
        (0.5**2) * (problem.kernel_mm @ alpha)
        + 1.0 * alpha
        + (features.T @ (features @ alpha - targets)) / batch_indices.size
    )


def run_fedavg(
    problem: FederatedProblem,
    batch_size: int,
    active_clients: int,
    local_epochs: int,
    num_rounds: int,
    learning_rate_schedule,
    seed: int,
) -> FedAvgResult:
    rng = np.random.default_rng(seed)
    server_alpha = np.zeros(problem.dim, dtype=float)
    optimum = federated_objective(problem, problem.alpha_star)
    errors = [federated_objective(problem, server_alpha) - optimum]
    communications = [0]

    for round_id in range(1, num_rounds + 1):
        selected = rng.choice(problem.num_clients, size=active_clients, replace=False)
        local_models = []
        local_weights = []
        for client_id in selected:
            alpha_local = server_alpha.copy()
            client_size = int(problem.client_sizes[client_id])
            full_indices = np.arange(client_size)
            for local_epoch in range(local_epochs):
                if batch_size >= client_size:
                    batch_indices = full_indices
                else:
                    batch_indices = np.sort(rng.choice(client_size, size=batch_size, replace=False))
                learning_rate = learning_rate_schedule(round_id - 1, local_epoch)
                gradient = federated_gradient(problem, client_id, alpha_local, batch_indices)
                alpha_local = alpha_local - learning_rate * gradient
            local_models.append(alpha_local)
            local_weights.append(problem.client_sizes[client_id])

        server_alpha = np.average(np.vstack(local_models), axis=0, weights=local_weights)
        errors.append(federated_objective(problem, server_alpha) - optimum)
        communications.append(round_id * active_clients)

    return FedAvgResult(
        final_alpha=server_alpha,
        objective_errors=np.asarray(errors),
        communications=np.asarray(communications),
    )


def plot_fedavg_family(results: dict[int, FedAvgResult], filename, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    iterations = np.arange(1, len(next(iter(results.values())).objective_errors) + 1)
    for epochs, result in results.items():
        ax.loglog(iterations, np.maximum(result.objective_errors, 1e-14), label=f"E = {epochs}")
    ax.set_xlabel("Communication rounds T")
    ax.set_ylabel(r"Objective error $F(\alpha)-F(\alpha^\star)$")
    ax.set_title(title)
    ax.legend()
    save_figure(fig, filename)


def plot_fedavg_communication(curves: list[tuple[FedAvgResult, str, str]], filename) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for result, label, style in curves:
        ax.loglog(
            np.maximum(result.communications, 1),
            np.maximum(result.objective_errors, 1e-14),
            style,
            label=label,
        )
    ax.set_xlabel("Communication sent C·T")
    ax.set_ylabel(r"Objective error $F(\alpha)-F(\alpha^\star)$")
    ax.set_title("FedAvg under a communication budget")
    ax.legend()
    save_figure(fig, filename)


def plot_federated_fit(problem: FederatedProblem, alpha: np.ndarray, filename) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x_grid = np.linspace(-1.0, 1.0, 250)
    ax.scatter(problem.all_x, problem.all_y, s=18, alpha=0.35, color="#4b5563", label="data")
    ax.scatter(
        problem.x_centers,
        predict_federated(problem, problem.alpha_star, problem.x_centers),
        s=40,
        marker="x",
        color="#111827",
        label="kernel centers",
    )
    ax.plot(x_grid, predict_federated(problem, problem.alpha_star, x_grid), color="#111827", label="centralized optimum")
    ax.plot(x_grid, predict_federated(problem, alpha, x_grid), color="#2563eb", label="FedAvg final model")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Part II fitted function")
    ax.legend()
    save_figure(fig, filename)


def generate_part2_results() -> dict[str, object]:
    problem = build_federated_problem(BASE_M)
    constant_lr = lambda round_idx, local_epoch: 0.002
    diminishing_lr = lambda round_idx, local_epoch: 0.002 / (1.0 + 0.01 * round_idx)
    epochs_list = [1, 5, 50]

    fullbatch_constant = {
        epochs: run_fedavg(problem, 20, 5, epochs, PART2_ROUNDS, constant_lr, SEED + epochs)
        for epochs in epochs_list
    }
    fullbatch_diminishing = {
        epochs: run_fedavg(problem, 20, 5, epochs, PART2_ROUNDS, diminishing_lr, SEED + 10 + epochs)
        for epochs in epochs_list
    }
    minibatch_constant = {
        epochs: run_fedavg(problem, 15, 5, epochs, PART2_ROUNDS, constant_lr, SEED + 20 + epochs)
        for epochs in epochs_list
    }
    partial_constant = {
        epochs: run_fedavg(problem, 15, 3, epochs, PART2_ROUNDS, constant_lr, SEED + 30 + epochs)
        for epochs in epochs_list
    }

    plot_fedavg_family(fullbatch_constant, RESULTS_DIR / "part2_fullbatch_constant.pdf", "Part II: FedAvg with B=20, C=5, constant learning rate")
    plot_fedavg_family(fullbatch_diminishing, RESULTS_DIR / "part2_fullbatch_diminishing.pdf", "Part II: FedAvg with B=20, C=5, diminishing learning rate")
    plot_fedavg_family(minibatch_constant, RESULTS_DIR / "part2_minibatch_constant.pdf", "Part II: FedAvg with B=15, C=5, constant learning rate")
    plot_fedavg_family(partial_constant, RESULTS_DIR / "part2_partial_participation.pdf", "Part II: FedAvg with B=15, C=3, constant learning rate")
    plot_fedavg_communication(
        [
            (fullbatch_constant[1], "FedAvg, E=1, C=5, B=20", "--"),
            (partial_constant[50], "FedAvg, E=50, C=3, B=15", "-"),
        ],
        RESULTS_DIR / "part2_communication_budget.pdf",
    )
    plot_federated_fit(problem, partial_constant[50].final_alpha, RESULTS_DIR / "part2_fitted_functions.pdf")

    summary_lines = [
        "",
        "[Part II]",
        "Constant learning rate 0.002, full batch B=20, full participation C=5:",
    ]
    summary_lines.extend(
        [f"E = {epochs}: final objective error = {fullbatch_constant[epochs].objective_errors[-1]:.6e}" for epochs in epochs_list]
    )
    summary_lines.append("Diminishing learning rate 0.002/(1+0.01 k), B=20, C=5:")
    summary_lines.extend(
        [f"E = {epochs}: final objective error = {fullbatch_diminishing[epochs].objective_errors[-1]:.6e}" for epochs in epochs_list]
    )
    summary_lines.append("Constant learning rate 0.002, mini-batch B=15, C=5:")
    summary_lines.extend(
        [f"E = {epochs}: final objective error = {minibatch_constant[epochs].objective_errors[-1]:.6e}" for epochs in epochs_list]
    )
    summary_lines.append("Constant learning rate 0.002, mini-batch B=15, partial participation C=3:")
    summary_lines.extend(
        [f"E = {epochs}: final objective error = {partial_constant[epochs].objective_errors[-1]:.6e}" for epochs in epochs_list]
    )

    return {
        "problem": problem,
        "summary_lines": summary_lines,
        "metrics": {
            "fullbatch_constant": fullbatch_constant,
            "fullbatch_diminishing": fullbatch_diminishing,
            "minibatch_constant": minibatch_constant,
            "partial_constant": partial_constant,
        },
    }
