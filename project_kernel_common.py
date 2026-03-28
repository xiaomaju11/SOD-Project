from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

_CACHE_ROOT = Path(".cache")
_CACHE_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str((_CACHE_ROOT / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SEED = 314
SIGMA = 0.5
NU = 1.0
BASE_AGENTS = 5
BASE_N = 100
BASE_M = 10
FIT_GRID_SIZE = 250
PART1_ITERS = 4000
PART1_SCALING_ITERS = 1500
PART2_ROUNDS = 3000
PART3_ITERS = 4000
RESULTS_DIR = Path("results")


@dataclass
class ConsensusProblem:
    x: np.ndarray
    y: np.ndarray
    x_centers: np.ndarray
    center_indices: np.ndarray
    agent_indices: list[np.ndarray]
    kernel_mm: np.ndarray
    local_features: list[np.ndarray]
    local_targets: list[np.ndarray]
    local_hessians: list[np.ndarray]
    local_linear_terms: list[np.ndarray]
    local_inverses: list[np.ndarray]
    global_hessian: np.ndarray
    global_linear_term: np.ndarray
    alpha_star: np.ndarray
    num_agents: int
    dim: int


@dataclass
class FederatedProblem:
    x_centers: np.ndarray
    kernel_mm: np.ndarray
    client_features: list[np.ndarray]
    client_targets: list[np.ndarray]
    client_sizes: np.ndarray
    global_hessian: np.ndarray
    global_linear_term: np.ndarray
    alpha_star: np.ndarray
    all_x: np.ndarray
    all_y: np.ndarray
    num_clients: int
    dim: int


@dataclass
class RunResult:
    final_alphas: np.ndarray
    mean_alpha: np.ndarray
    gaps: np.ndarray
    objective_errors: np.ndarray
    consensus: np.ndarray


@dataclass
class FedAvgResult:
    final_alpha: np.ndarray
    objective_errors: np.ndarray
    communications: np.ndarray


def set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.6),
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


def load_pickle(path: str):
    with open(path, "rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return pickle.load(handle)


def gaussian_kernel(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    return np.exp(-((x1[:, None] - x2[None, :]) ** 2))


def save_figure(fig: plt.Figure, filename: Path) -> None:
    fig.tight_layout()
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_consensus_problem(
    n: int,
    m: int,
    num_agents: int = BASE_AGENTS,
    seed: int = SEED,
) -> ConsensusProblem:
    x_raw, y_raw = load_pickle("first_database.pkl")
    x = np.asarray(x_raw[:n], dtype=float)
    y = np.asarray(y_raw[:n], dtype=float)
    rng = np.random.default_rng(seed + n + m)

    center_indices = np.sort(rng.choice(n, size=m, replace=False))
    x_centers = x[center_indices]
    kernel_mm = gaussian_kernel(x_centers, x_centers)
    agent_indices = [chunk for chunk in np.array_split(rng.permutation(n), num_agents)]

    local_features: list[np.ndarray] = []
    local_targets: list[np.ndarray] = []
    local_hessians: list[np.ndarray] = []
    local_linear_terms: list[np.ndarray] = []
    local_inverses: list[np.ndarray] = []

    for indices in agent_indices:
        features = gaussian_kernel(x[indices], x_centers)
        targets = y[indices]
        hessian = (
            (SIGMA**2 / num_agents) * kernel_mm
            + features.T @ features
            + (NU / num_agents) * np.eye(m)
        )
        linear_term = features.T @ targets
        local_features.append(features)
        local_targets.append(targets)
        local_hessians.append(hessian)
        local_linear_terms.append(linear_term)
        local_inverses.append(np.linalg.inv(hessian))

    global_hessian = np.sum(local_hessians, axis=0)
    global_linear_term = np.sum(local_linear_terms, axis=0)
    alpha_star = np.linalg.solve(global_hessian, global_linear_term)

    return ConsensusProblem(
        x=x,
        y=y,
        x_centers=x_centers,
        center_indices=center_indices,
        agent_indices=agent_indices,
        kernel_mm=kernel_mm,
        local_features=local_features,
        local_targets=local_targets,
        local_hessians=local_hessians,
        local_linear_terms=local_linear_terms,
        local_inverses=local_inverses,
        global_hessian=global_hessian,
        global_linear_term=global_linear_term,
        alpha_star=alpha_star,
        num_agents=num_agents,
        dim=m,
    )


def build_federated_problem(m: int = BASE_M) -> FederatedProblem:
    clients_x, clients_y = load_pickle("second_database.pkl")
    x_centers = np.linspace(-1.0, 1.0, m)
    kernel_mm = gaussian_kernel(x_centers, x_centers)

    client_features: list[np.ndarray] = []
    client_targets: list[np.ndarray] = []
    client_hessians: list[np.ndarray] = []
    client_linear_terms: list[np.ndarray] = []
    client_sizes = []
    all_x = []
    all_y = []

    for x_local, y_local in zip(clients_x, clients_y):
        x_array = np.asarray(x_local, dtype=float)
        y_array = np.asarray(y_local, dtype=float)
        features = gaussian_kernel(x_array, x_centers)
        size = x_array.size
        hessian = (
            (SIGMA**2) * kernel_mm
            + NU * np.eye(m)
            + (features.T @ features) / size
        )
        linear_term = (features.T @ y_array) / size
        client_features.append(features)
        client_targets.append(y_array)
        client_hessians.append(hessian)
        client_linear_terms.append(linear_term)
        client_sizes.append(size)
        all_x.append(x_array)
        all_y.append(y_array)

    global_hessian = np.mean(client_hessians, axis=0)
    global_linear_term = np.mean(client_linear_terms, axis=0)
    alpha_star = np.linalg.solve(global_hessian, global_linear_term)

    return FederatedProblem(
        x_centers=x_centers,
        kernel_mm=kernel_mm,
        client_features=client_features,
        client_targets=client_targets,
        client_sizes=np.asarray(client_sizes, dtype=float),
        global_hessian=global_hessian,
        global_linear_term=global_linear_term,
        alpha_star=alpha_star,
        all_x=np.concatenate(all_x),
        all_y=np.concatenate(all_y),
        num_clients=len(client_features),
        dim=m,
    )


def build_edges(graph_name: str, num_agents: int) -> list[tuple[int, int]]:
    if graph_name == "line":
        return [(idx, idx + 1) for idx in range(num_agents - 1)]
    if graph_name == "small_world":
        edges = [(idx, (idx + 1) % num_agents) for idx in range(num_agents)]
        edges.extend((idx, (idx + 2) % num_agents) for idx in range(num_agents // 2))
        return sorted({tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]})
    if graph_name == "complete":
        return [(i, j) for i in range(num_agents) for j in range(i + 1, num_agents)]
    raise ValueError(f"Unknown graph '{graph_name}'.")


def lazy_metropolis_weights(edges: list[tuple[int, int]], num_agents: int) -> np.ndarray:
    degrees = np.zeros(num_agents, dtype=int)
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1

    weights = np.zeros((num_agents, num_agents), dtype=float)
    for left, right in edges:
        value = 1.0 / (2.0 * (1 + max(degrees[left], degrees[right])))
        weights[left, right] = value
        weights[right, left] = value

    np.fill_diagonal(weights, 1.0 - weights.sum(axis=1))
    return weights


def directed_ring_weights(num_agents: int) -> np.ndarray:
    weights = np.zeros((num_agents, num_agents), dtype=float)
    for idx in range(num_agents):
        weights[idx, idx] = 0.6
        weights[idx, (idx - 1) % num_agents] = 0.4
    return weights


def incidence_matrix(edges: list[tuple[int, int]], num_agents: int) -> np.ndarray:
    matrix = np.zeros((len(edges), num_agents), dtype=float)
    for edge_id, (left, right) in enumerate(edges):
        matrix[edge_id, left] = 1.0
        matrix[edge_id, right] = -1.0
    return matrix


def graph_laplacian(edges: list[tuple[int, int]], num_agents: int) -> np.ndarray:
    incidence = incidence_matrix(edges, num_agents)
    return incidence.T @ incidence


def edge_map(edges: list[tuple[int, int]], num_agents: int) -> list[list[tuple[int, int]]]:
    mapping: list[list[tuple[int, int]]] = [[] for _ in range(num_agents)]
    for edge_id, (left, right) in enumerate(edges):
        mapping[left].append((edge_id, 0))
        mapping[right].append((edge_id, 1))
    return mapping


def local_gradients(problem: ConsensusProblem, alphas: np.ndarray) -> np.ndarray:
    return np.vstack(
        [
            problem.local_hessians[idx] @ alphas[idx] - problem.local_linear_terms[idx]
            for idx in range(problem.num_agents)
        ]
    )


def single_local_gradient(problem: ConsensusProblem, agent_id: int, alpha: np.ndarray) -> np.ndarray:
    return problem.local_hessians[agent_id] @ alpha - problem.local_linear_terms[agent_id]


def global_objective(problem: ConsensusProblem, alpha: np.ndarray) -> float:
    return float(
        0.5 * alpha @ problem.global_hessian @ alpha - problem.global_linear_term @ alpha
    )


def federated_objective(problem: FederatedProblem, alpha: np.ndarray) -> float:
    return float(
        0.5 * alpha @ problem.global_hessian @ alpha - problem.global_linear_term @ alpha
    )


def average_optimality_gap(alphas: np.ndarray, alpha_star: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(alphas - alpha_star[None, :], axis=1)))


def consensus_error(alphas: np.ndarray) -> float:
    mean_alpha = alphas.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(alphas - mean_alpha, axis=1)))


def initialize_run_history(problem: ConsensusProblem, alphas: np.ndarray) -> dict[str, list[float]]:
    mean_alpha = alphas.mean(axis=0)
    optimum = global_objective(problem, problem.alpha_star)
    return {
        "gaps": [average_optimality_gap(alphas, problem.alpha_star)],
        "objective_errors": [global_objective(problem, mean_alpha) - optimum],
        "consensus": [consensus_error(alphas)],
    }


def append_run_history(
    history: dict[str, list[float]],
    problem: ConsensusProblem,
    alphas: np.ndarray,
) -> None:
    mean_alpha = alphas.mean(axis=0)
    optimum = global_objective(problem, problem.alpha_star)
    history["gaps"].append(average_optimality_gap(alphas, problem.alpha_star))
    history["objective_errors"].append(global_objective(problem, mean_alpha) - optimum)
    history["consensus"].append(consensus_error(alphas))


def finalize_run(
    problem: ConsensusProblem,
    alphas: np.ndarray,
    history: dict[str, list[float]],
) -> RunResult:
    return RunResult(
        final_alphas=alphas,
        mean_alpha=alphas.mean(axis=0),
        gaps=np.asarray(history["gaps"]),
        objective_errors=np.asarray(history["objective_errors"]),
        consensus=np.asarray(history["consensus"]),
    )


def local_smoothness(problem: ConsensusProblem) -> float:
    return max(float(np.linalg.eigvalsh(hessian).max()) for hessian in problem.local_hessians)


def predict_consensus(problem: ConsensusProblem, alpha: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    return gaussian_kernel(x_grid, problem.x_centers) @ alpha


def predict_federated(problem: FederatedProblem, alpha: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    return gaussian_kernel(x_grid, problem.x_centers) @ alpha


def plot_gap_curves(
    curves: list[tuple[np.ndarray, str, object]],
    filename: Path,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots()
    iterations = np.arange(1, len(curves[0][0]) + 1)
    for values, label, style in curves:
        if isinstance(style, str):
            ax.loglog(iterations, np.maximum(values, 1e-14), style, label=label)
        else:
            ax.loglog(iterations, np.maximum(values, 1e-14), linestyle=style, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_figure(fig, filename)


def write_summary(lines: list[str]) -> Path:
    summary_path = RESULTS_DIR / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path
