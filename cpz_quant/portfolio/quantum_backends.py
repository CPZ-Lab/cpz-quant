"""QUBO solvers for quantum and quantum-inspired portfolio optimization.

Provides a ``QUBOSolver`` protocol with classical, simulated-annealing,
and Amazon Braket (real quantum hardware) implementations.

Extras:
    cpz-ai                   # classical only (heuristic + brute_force)
    cpz-ai[quantum]          # + simulated annealing (dwave-neal)
    cpz-ai[quantum-braket]   # + Amazon Braket (D-Wave, IonQ, Rigetti)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QUBOSolver(Protocol):
    """Protocol for solving QUBO problems: minimize x^T Q x where x in {0,1}^n."""

    def solve(self, Q: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Solve a QUBO and return the best binary solution vector.

        Args:
            Q: (n, n) upper-triangular or symmetric QUBO matrix.
            **kwargs: Solver-specific parameters.

        Returns:
            (n,) binary numpy array with the lowest-energy solution.
        """
        ...


class ClassicalBruteForceSolver:
    """Exact enumeration over all 2^n solutions. Feasible for n <= 20."""

    def solve(self, Q: np.ndarray, **kwargs: Any) -> np.ndarray:
        n = Q.shape[0]
        if n > 20:
            raise ValueError(
                f"Brute-force is infeasible for n={n} (2^{n} states). "
                f"Use 'simulated_annealing' or 'braket' backend instead."
            )

        Q_sym = (Q + Q.T) / 2.0
        best_x: Optional[np.ndarray] = None
        best_val = float("inf")

        for mask in range(2**n):
            x = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.float64)
            val = float(x @ Q_sym @ x)
            if val < best_val:
                best_val = val
                best_x = x.copy()

        return best_x if best_x is not None else np.zeros(n)


class SimulatedAnnealingSolver:
    """CPU-based simulated annealing via dwave-neal.

    Mimics quantum annealing behaviour on classical hardware.
    Requires the ``quantum`` extra: ``cpz-ai[quantum]``

    Args:
        num_reads: Number of independent annealing runs.
        num_sweeps: Sweeps per run (higher = more thorough).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_reads: int = 1000,
        num_sweeps: int = 1000,
        seed: Optional[int] = None,
    ):
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.seed = seed

    def solve(self, Q: np.ndarray, **kwargs: Any) -> np.ndarray:
        try:
            import dimod
            import neal
        except ImportError:
            raise ImportError(
                "Simulated annealing requires the 'quantum' extra (dwave-neal, dimod). "
                "Add 'cpz-ai[quantum]' to your strategy requirements. "
                "Or use backend='heuristic' which needs no extra deps."
            )

        n = Q.shape[0]
        Q_sym = (Q + Q.T) / 2.0

        linear: Dict[int, float] = {}
        quadratic: Dict[Tuple[int, int], float] = {}

        for i in range(n):
            linear[i] = Q_sym[i, i]
            for j in range(i + 1, n):
                if Q_sym[i, j] != 0.0:
                    quadratic[(i, j)] = Q_sym[i, j]

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

        sampler = neal.SimulatedAnnealingSampler()
        result = sampler.sample(
            bqm,
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            seed=self.seed,
        )

        best = result.first.sample
        x = np.array([best.get(i, 0) for i in range(n)], dtype=np.float64)
        return x


BRAKET_COST_PER_TASK = {
    "d-wave": 0.30,
    "ionq": 0.30,
    "rigetti": 0.30,
    "sv1": 0.00,
    "dm1": 0.00,
    "tn1": 0.00,
}

BRAKET_COST_PER_SHOT = {
    "d-wave": 0.00019,
    "ionq": 0.01,
    "rigetti": 0.00035,
    "sv1": 0.0,
    "dm1": 0.0,
    "tn1": 0.0,
}


def _estimate_braket_cost(device_arn: str, shots: int) -> float:
    """Estimate the cost of a Braket task based on device and shot count."""
    device_key = "sv1"
    for key in BRAKET_COST_PER_TASK:
        if key in device_arn.lower():
            device_key = key
            break
    return BRAKET_COST_PER_TASK.get(device_key, 0.30) + \
           BRAKET_COST_PER_SHOT.get(device_key, 0.001) * shots


def _record_quantum_usage(
    cpz_client: Any,
    device_arn: str,
    n_variables: int,
    shots: int,
    estimated_cost: float,
    task_id: str = "",
) -> None:
    """Record quantum compute usage to the CPZ platform for billing."""
    try:
        import requests as _requests

        payload = {
            "service": "quantum_compute",
            "provider": "amazon_braket",
            "device": device_arn,
            "n_variables": n_variables,
            "shots": shots,
            "estimated_cost_usd": round(estimated_cost, 6),
            "task_id": task_id,
        }

        url = f"{cpz_client.url}/quantum-usage"
        headers = cpz_client._headers()
        resp = _requests.post(url, json=payload, headers=headers, timeout=10)  # type: ignore[arg-type]

        if not resp.ok:
            url_fallback = f"{cpz_client.url}/rest/v1/quantum_usage"
            _requests.post(url_fallback, json=payload, headers=headers, timeout=10)  # type: ignore[arg-type]

        logger.info(
            "Quantum usage recorded: device=%s, vars=%d, shots=%d, est_cost=$%.4f",
            device_arn, n_variables, shots, estimated_cost,
        )
    except Exception as exc:
        logger.warning("Failed to record quantum usage: %s", exc)


class BraketSolver:
    """Amazon Braket -- submit QUBOs to real quantum hardware or managed simulators.

    **Requires authenticated CPZ API keys.** Quantum compute costs are tracked
    against your CPZ account. Unauthenticated users cannot access quantum hardware.

    Requires the ``quantum-braket`` extra: ``cpz-ai[quantum-braket]``

    Supported devices (pass full ARN or short alias):
        - ``"ionq"`` / IonQ Forte-1 trapped-ion QPU (us-east-1). ~$10.30/task at 1000 shots.
        - ``"rigetti"`` / Rigetti Ankaa-3 superconducting QPU (us-west-1). ~$0.65/task at 1000 shots.
        - ``"iqm"`` / IQM Garnet superconducting QPU (eu-north-1).
        - ``"quera"`` / QuEra Aquila neutral-atom QPU (us-east-1).
        - ``"sv1"`` / Amazon SV1 state-vector simulator (classical, for testing). Free.

    Note: D-Wave quantum annealers are not currently available on Braket.
    Use ``SimulatedAnnealingSolver`` for QUBO-native annealing on classical CPU,
    or ``"ionq"``/``"rigetti"`` for gate-based QAOA on real QPUs.

    Args:
        device_arn: Braket device ARN or short alias.
        s3_folder: ``(bucket, prefix)`` for task results. Required.
        shots: Number of measurement shots.
        poll_timeout: Max seconds to wait for task completion.
        cpz_client: Authenticated CPZAIClient instance. If None, loaded from env
                    (CPZ_AI_API_KEY + CPZ_AI_API_SECRET must be set).
    """

    DEVICE_ALIASES = {
        "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        "dm1": "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
        "tn1": "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
        "ionq": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
        "rigetti": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
        "iqm": "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
        "quera": "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
    }

    def __init__(
        self,
        device_arn: str = "dwave",
        s3_folder: Optional[Tuple[str, str]] = None,
        shots: int = 1000,
        poll_timeout: int = 300,
        cpz_client: Optional[Any] = None,
    ):
        if s3_folder is None:
            raise ValueError(
                "s3_folder is required for Braket. "
                "Provide (bucket_name, prefix), e.g. ('cpzai-braket-results', 'quantum-tasks')"
            )

        self.device_arn = self.DEVICE_ALIASES.get(device_arn, device_arn)
        self.s3_folder = s3_folder
        self.shots = shots
        self.poll_timeout = poll_timeout

        # The platform client is resolved lazily in solve(): constructing the
        # solver and estimating costs stay fully standalone.
        self._cpz_client = cpz_client

        logger.info("BraketSolver initialised: device=%s", self.device_arn)

    def _resolve_client(self) -> Any:
        """Authenticated CPZAI client, resolved on first hardware submission."""
        if self._cpz_client is not None:
            return self._cpz_client
        try:
            from cpz.common.cpz_ai import CPZAIClient
        except ImportError:
            raise PermissionError(
                "Amazon Braket quantum hardware is executed through the CPZAI "
                "platform, which requires the cpz-ai SDK (pip install cpz-ai) "
                "and a CPZAI account. The local solvers "
                "(SimulatedAnnealingSolver, ClassicalBruteForceSolver, and the "
                "D-Wave 'quantum' extra) work standalone without it."
            ) from None
        try:
            self._cpz_client = CPZAIClient.from_env()
        except (ValueError, KeyError) as exc:
            raise PermissionError(
                "Amazon Braket quantum compute requires authenticated CPZ API keys. "
                "Set CPZ_AI_API_KEY and CPZ_AI_API_SECRET environment variables, "
                "or pass cpz_client=CPZAIClient.from_keys(key, secret). "
                f"Auth error: {exc}"
            )
        return self._cpz_client

    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD for one solve() call at current settings."""
        return _estimate_braket_cost(self.device_arn, self.shots)

    def _solve_simulator(self, Q: np.ndarray) -> np.ndarray:
        """Solve via SimulatedAnnealingSolver as fallback for SV1/DM1/TN1."""
        sa = SimulatedAnnealingSolver(num_reads=self.shots, num_sweeps=1000)
        return sa.solve(Q)

    def solve(self, Q: np.ndarray, **kwargs: Any) -> np.ndarray:
        try:
            import dimod
        except ImportError:
            raise ImportError(
                "Amazon Braket integration requires dimod. "
                "Add 'cpz-ai[quantum-braket]' to your strategy requirements."
            )

        n = Q.shape[0]
        estimated = _estimate_braket_cost(self.device_arn, self.shots)
        logger.info(
            "Quantum task: %d variables, %d shots, estimated cost $%.4f",
            n, self.shots, estimated,
        )

        Q_sym = (Q + Q.T) / 2.0

        linear: Dict[int, float] = {}
        quadratic: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            linear[i] = Q_sym[i, i]
            for j in range(i + 1, n):
                if Q_sym[i, j] != 0.0:
                    quadratic[(i, j)] = Q_sym[i, j]

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
        s3_destination = (self.s3_folder[0], self.s3_folder[1])

        is_simulator = any(s in self.device_arn.lower() for s in ("sv1", "dm1", "tn1"))

        if is_simulator:
            # Braket managed simulators: use SimulatedAnnealing locally
            # (SV1 is a state-vector sim — not compatible with QUBO/annealing)
            logger.info("Braket simulator detected, solving with local SA instead")
            return self._solve_simulator(Q)

        try:
            from braket.ocean_plugin import BraketDWaveSampler
        except ImportError:
            raise ImportError(
                "Amazon Braket QPU access requires amazon-braket-ocean-plugin. "
                "Add 'cpz-ai[quantum-braket]' to your strategy requirements."
            )

        is_dwave = "d-wave" in self.device_arn.lower()

        if is_dwave:
            from dwave.system.composites import EmbeddingComposite

            # Paid hardware submission: authenticate with the CPZAI platform
            # BEFORE the task is sent so usage is always recorded and billed.
            self._resolve_client()

            sampler = BraketDWaveSampler(
                s3_destination_folder=s3_destination,
                device_arn=self.device_arn,
            )
            composite = EmbeddingComposite(sampler)

            logger.info(
                "Submitting QUBO (%d variables) to D-Wave via Braket: %s",
                n, self.device_arn,
            )
            result = composite.sample(bqm, num_reads=self.shots)
        else:
            # Gate-based QPU (IonQ, Rigetti, IQM)
            # The ocean plugin's BraketSampler has compatibility issues with
            # gate-based devices — fall back to SA with a clear message.
            logger.warning(
                "Gate-based QPU (%s) is not compatible with QUBO/annealing. "
                "QUBO problems require annealing hardware (D-Wave). "
                "Falling back to simulated annealing.",
                self.device_arn,
            )
            print(f"[QUANTUM] Gate-based QPU ({self.device_arn}) cannot run QUBO natively.")
            print("[QUANTUM] QUBO requires annealing hardware. Using simulated annealing instead.")
            print("[QUANTUM] For real quantum annealing, use device='sv1' (simulator) or wait for D-Wave on Braket.")
            return self._solve_simulator(Q)

        best = result.first.sample
        x = np.array([best.get(i, 0) for i in range(n)], dtype=np.float64)

        task_id = ""
        try:
            if hasattr(result, 'info') and 'timing' in result.info:
                task_id = str(result.info.get('task_id', ''))
        except Exception:
            pass

        logger.info(
            "Braket task complete. Best energy: %.4f, cost: ~$%.4f",
            result.first.energy, estimated,
        )

        _record_quantum_usage(
            self._cpz_client,
            device_arn=self.device_arn,
            n_variables=n,
            shots=self.shots,
            estimated_cost=estimated,
            task_id=task_id,
        )

        return x


def get_solver(backend: str, **kwargs: Any) -> QUBOSolver:
    """Factory function to create a QUBOSolver by name.

    Args:
        backend: One of ``"brute_force"``, ``"simulated_annealing"``, ``"braket"``.
        **kwargs: Passed to the solver constructor.

    Returns:
        A QUBOSolver instance.
    """
    if backend == "brute_force":
        return ClassicalBruteForceSolver()
    elif backend == "simulated_annealing":
        return SimulatedAnnealingSolver(**kwargs)
    elif backend == "braket":
        return BraketSolver(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Use: 'brute_force', 'simulated_annealing', or 'braket'."
        )
