"""Truthful QUBO solvers. All shipped execution is classical or local simulation.

Braket local QAOA is experimental. Managed simulators and physical QPUs are
disabled pending durable budget enforcement and provider cost reconciliation.
No backend silently substitutes simulated annealing for quantum execution.
Formulations, hedge logic, routing, and scheduling remain private to CPZAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

__all__ = [
    "QUBOSolver",
    "ClassicalBruteForceSolver",
    "SimulatedAnnealingSolver",
    "BraketSolver",
    "QaoaBraketSolver",
    "get_solver",
]


class QUBOSolver(Protocol):
    def solve(self, Q: np.ndarray, **kwargs: Any) -> np.ndarray: ...


def _matrix(Q: Any, maximum: int = 5000) -> np.ndarray:
    q = np.asarray(Q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not 1 <= q.shape[0] <= maximum:
        raise ValueError(f"Q must be square with 1..{maximum} variables")
    if not np.isfinite(q).all():
        raise ValueError("Q contains non-finite coefficients")
    symmetric = q / 2.0 + q.T / 2.0
    return symmetric


def _bqm_from_qubo(Q: np.ndarray) -> Any:
    import dimod

    q = _matrix(Q)
    # x'Qx counts each symmetric off-diagonal term TWICE.
    return dimod.BinaryQuadraticModel(
        {i: float(q[i, i]) for i in range(len(q))},
        {
            (i, j): float(q[i, j] * 2)
            for i in range(len(q))
            for j in range(i + 1, len(q))
            if q[i, j] != 0
        },
        0.0,
        dimod.BINARY,
    )


class ClassicalBruteForceSolver:
    """Exact enumeration on the user's CPU, at most 20 binary variables."""

    def solve(self, Q: np.ndarray, **kwargs: Any) -> np.ndarray:
        if kwargs:
            raise TypeError("brute_force accepts no solve options")
        q = _matrix(Q, 20)
        n = len(q)
        best = None
        best_energy = float("inf")
        for mask in range(1 << n):
            x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
            energy = float(x @ q @ x)
            if not np.isfinite(energy):
                raise ValueError("QUBO energy overflow")
            if energy < best_energy:
                best, best_energy = x, energy
        if best is None:
            raise RuntimeError("Exact solver produced no solution")
        return best


class SimulatedAnnealingSolver:
    """Classical CPU heuristic, never quantum hardware."""

    def __init__(self, num_reads: int = 1000, num_sweeps: int = 1000, seed: Optional[int] = None):
        for key, value in [("num_reads", num_reads), ("num_sweeps", num_sweeps)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        self.num_reads, self.num_sweeps, self.seed = num_reads, num_sweeps, seed

    def solve(self, Q: np.ndarray, **kwargs: Any) -> np.ndarray:
        if kwargs:
            raise TypeError("simulated_annealing accepts no solve options")
        try:
            import neal
        except ImportError as exc:
            raise ImportError("Install cpz-quant[quantum] for simulated annealing") from exc
        q = _matrix(Q)
        result = neal.SimulatedAnnealingSampler().sample(
            _bqm_from_qubo(q), num_reads=self.num_reads, num_sweeps=self.num_sweeps, seed=self.seed
        )
        sample = result.first.sample
        return np.array([sample[i] for i in range(len(q))], dtype=float)


@dataclass(frozen=True)
class BraketTaskMeta:
    device_arn: str
    shots: int
    n_tasks: int
    cost_usd: float = 0.0
    cost_source: str = "free"
    task_ids: tuple = ()
    queue_time_ms: None = None
    execution_time_ms: None = None
    notes: tuple = ("Local CPU state-vector simulation; no physical QPU used",)


class BraketSolver:
    """Experimental gate-model QAOA on the Braket LocalSimulator.

    No AWS credentials are read. device/device_arn other than "local" raises,
    instead of returning classical annealing labeled as a Braket QPU result.
    Classical seed controls initialization and sampling of simulator probabilities.
    Local compute still consumes the caller's CPU resources; provider cost is zero.
    """

    def __init__(
        self,
        device: str = "local",
        shots: int = 1000,
        p: int = 1,
        max_iter: int = 50,
        seed: Optional[int] = None,
        device_arn: Optional[str] = None,
        s3_folder: Any = None,
        on_before_submit: Any = None,
        on_after_result: Any = None,
    ):
        if device_arn is not None and device != "local":
            raise ValueError("Pass device OR device_arn")
        selected = device_arn if device_arn is not None else device
        if selected != "local":
            raise NotImplementedError(
                "Paid Braket devices are disabled pending budget enforcement and "
                "cost reconciliation. Use device='local' explicitly for simulation."
            )
        if s3_folder is not None or on_before_submit is not None:
            raise ValueError("Local simulation requires no S3 folder or paid-submission hook")
        for key, value, cap in [
            ("shots", shots, 100000),
            ("p", p, 10),
            ("max_iter", max_iter, 1000),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= cap:
                raise ValueError(f"{key} must be an integer in [1,{cap}]")
        if max_iter < 2 * p + 2:
            raise ValueError("max_iter must be at least 2*p+2 for COBYLA")
        self.device_arn, self.device_kind = "local", "local"
        self.shots, self.p, self.max_iter, self.seed = shots, p, max_iter, seed
        self.on_after_result = on_after_result
        self.last_meta: Optional[BraketTaskMeta] = None

    @property
    def estimated_cost(self) -> float:
        """Zero provider fee, not zero local compute cost."""
        return 0.0

    def solve(self, Q: np.ndarray, **kwargs: Any) -> np.ndarray:
        if kwargs:
            raise TypeError("BraketSolver.solve accepts no options")
        q = _matrix(Q, 20)
        try:
            import dimod  # noqa: F401
            from braket.circuits import Circuit
            from braket.devices import LocalSimulator
            from scipy.optimize import minimize
        except ImportError as exc:
            raise ImportError("Install cpz-quant[quantum-braket] for local QAOA") from exc
        h, couplings, _ = _bqm_from_qubo(q).to_ising()
        scale = max([abs(v) for v in h.values()] + [abs(v) for v in couplings.values()] + [0.0])
        if scale == 0:
            self.last_meta = BraketTaskMeta("local", self.shots, 0)
            return np.zeros(len(q))
        rng = np.random.default_rng(self.seed)
        simulator = LocalSimulator()
        n, evaluations = len(q), 0
        best, best_energy = None, float("inf")

        def objective(params):
            nonlocal evaluations, best, best_energy
            if evaluations >= self.max_iter:
                raise RuntimeError("QAOA evaluation budget exceeded")
            evaluations += 1
            circuit = Circuit()
            for i in range(n):
                circuit.h(i)
            for layer in range(self.p):
                gamma, beta = params[layer], params[self.p + layer]
                # dimod spin s=2x-1 is -Z for the measured computational bit.
                for i, value in h.items():
                    if value:
                        circuit.rz(i, -2 * gamma * value / scale)
                for (i, j), value in couplings.items():
                    if value:
                        circuit.zz(i, j, 2 * gamma * value / scale)
                for i in range(n):
                    circuit.rx(i, 2 * beta)
            probabilities = np.asarray(
                simulator.run(circuit.probability(), shots=0).result().values[0]
            )
            if (
                probabilities.shape != (1 << n,)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or abs(probabilities.sum() - 1) > 1e-6
            ):
                raise RuntimeError("Invalid simulator probability distribution")
            indices = rng.choice(
                len(probabilities), size=self.shots, p=probabilities / probabilities.sum()
            )
            samples = ((indices[:, None] >> (n - 1 - np.arange(n))) & 1).astype(float)
            energies = np.einsum("si,ij,sj->s", samples, q, samples)
            if not np.isfinite(energies).all():
                raise ValueError("QUBO energy overflow")
            k = int(np.argmin(energies))
            if energies[k] < best_energy:
                best, best_energy = samples[k].copy(), float(energies[k])
            return float(energies.mean())

        params = np.concatenate([rng.uniform(0, np.pi, self.p), rng.uniform(0, np.pi / 2, self.p)])
        minimize(
            objective, params, method="COBYLA", options={"maxiter": self.max_iter, "rhobeg": 0.5}
        )
        if best is None:
            raise RuntimeError("No QAOA sample obtained")
        self.last_meta = BraketTaskMeta("local", self.shots, evaluations)
        if self.on_after_result is not None:
            self.on_after_result(self.last_meta)
        return best


QaoaBraketSolver = BraketSolver


def get_solver(backend: str, **kwargs: Any) -> QUBOSolver:
    if backend == "brute_force":
        if kwargs:
            raise TypeError("brute_force takes no constructor options")
        return ClassicalBruteForceSolver()
    if backend == "simulated_annealing":
        return SimulatedAnnealingSolver(**kwargs)
    if backend in ("braket", "braket_qaoa"):
        return BraketSolver(**kwargs)
    raise ValueError(
        f"Unknown backend '{backend}'. Use brute_force, simulated_annealing, or braket."
    )
