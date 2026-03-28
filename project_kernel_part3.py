from __future__ import annotations

import numpy as np

from project_kernel_common import (
    PART3_ITERS,
    RESULTS_DIR,
    SEED,
    ConsensusProblem,
    RunResult,
    append_run_history,
    build_edges,
    finalize_run,
    initialize_run_history,
    lazy_metropolis_weights,
    local_gradients,
    plot_gap_curves,
)


def laplace_noise(shape: tuple[int, ...], variance: float, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(max(variance, 1e-15) / 2.0)
    return rng.laplace(loc=0.0, scale=scale, size=shape)


def dp_schedule(iteration: int, epsilon: float | None) -> tuple[float, float, float | None]:
    alpha_k = 0.002 / (1.0 + 0.001 * iteration)
    gamma_k = 1.0 / (1.0 + 0.001 * iteration) ** 0.9
    if epsilon is None:
        return alpha_k, gamma_k, None
    variance = (0.01 / epsilon) / (1.0 + 0.001 * iteration) ** 0.1
    return alpha_k, gamma_k, variance


def run_dgd_dp(
    problem: ConsensusProblem,
    weights: np.ndarray,
    epsilon: float | None,
    num_iters: int,
    seed: int,
) -> RunResult:
    alphas = np.zeros((problem.num_agents, problem.dim), dtype=float)
    history = initialize_run_history(problem, alphas)
    rng = np.random.default_rng(seed)

    for iteration in range(num_iters):
        alpha_k, gamma_k, variance = dp_schedule(iteration, epsilon)
        if variance is None:
            noisy_state = alphas
        else:
            noisy_state = alphas + laplace_noise(alphas.shape, variance, rng)

        alphas = (
            (1.0 - gamma_k) * alphas
            + gamma_k * (weights @ noisy_state)
            - gamma_k * alpha_k * local_gradients(problem, alphas)
        )
        append_run_history(history, problem, alphas)

    return finalize_run(problem, alphas, history)


def generate_part3_results(problem: ConsensusProblem) -> dict[str, object]:
    weights = lazy_metropolis_weights(build_edges("complete", problem.num_agents), problem.num_agents)
    baseline = run_dgd_dp(problem, weights, epsilon=None, num_iters=PART3_ITERS, seed=SEED)
    dp_results = {
        epsilon: run_dgd_dp(problem, weights, epsilon, PART3_ITERS, SEED + int(10 * epsilon))
        for epsilon in (0.1, 1.0, 10.0)
    }

    plot_gap_curves(
        [
            (dp_results[0.1].gaps, r"DP-DGD, $\epsilon=0.1$", "-"),
            (dp_results[1.0].gaps, r"DP-DGD, $\epsilon=1$", "--"),
            (dp_results[10.0].gaps, r"DP-DGD, $\epsilon=10$", "-."),
            (baseline.gaps, "DGD (no noise)", ":"),
        ],
        RESULTS_DIR / "part3_dgd_dp.pdf",
        ylabel="Average optimality gap",
        title="Part III: DGD with Laplacian noise",
    )

    return {
        "summary_lines": [
            "",
            "[Part III]",
            "Schedules: alpha_k = 0.002/(1+0.001k), gamma_k = 1/(1+0.001k)^0.9, nu_k = (0.01/epsilon)/(1+0.001k)^0.1",
            f"DGD (no noise) final gap = {baseline.gaps[-1]:.6e}",
            f"epsilon = 0.1 final gap = {dp_results[0.1].gaps[-1]:.6e}",
            f"epsilon = 1 final gap = {dp_results[1.0].gaps[-1]:.6e}",
            f"epsilon = 10 final gap = {dp_results[10.0].gaps[-1]:.6e}",
        ],
        "metrics": {"baseline": baseline, "dp_results": dp_results},
    }
