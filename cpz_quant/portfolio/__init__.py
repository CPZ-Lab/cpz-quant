"""CPZ Portfolio — institutional-grade portfolio construction.

Covers every stage of Paleologo's pipeline: covariance estimation,
factor models, optimisation (10 methods + alpha-risk-cost framework),
transaction costs (Almgren-Chriss), and performance attribution.
"""

from .attribution import (
    AlphaBetaResult,
    BrinsonResult,
    FactorAttrResult,
    RiskAttrResult,
    alpha_beta_decomposition,
    brinson_fachler,
    factor_attribution,
    risk_attribution,
    rolling_attribution,
)

# NOTE: scikit-learn estimator wrappers live in cpz_quant.portfolio.sklearn_estimators
# and are intentionally NOT imported here — they are an optional extra
# (`pip install 'cpz-quant[sklearn]'`) so the base package stays scikit-learn-free.
from .convex import (
    cardinality_constrained_cvx,
    has_cvxpy,
    mean_cvar_cvx,
    mean_variance_cvx,
    robust_mean_variance_cvx,
)
from .copula import (
    ClaytonCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
    fit_copula,
    pseudo_observations,
    synthetic_returns,
)
from .costs import (
    CapacityEstimate,
    CostEstimate,
    CostModel,
    TurnoverResult,
    almgren_chriss,
    alpha_decay_capacity,
    linear_impact,
    spread_cost,
    sqrt_impact,
    total_cost,
    turnover_analysis,
)
from .covariance import (
    FactorCovResult,
    LedoitWolfResult,
    denoise_mp,
    detone_cov,
    ewma_cov,
    factor_model_cov,
    gerber_cov,
    ledoit_wolf,
    oracle_approximating,
    sample_cov,
)
from .entropy_pooling import (
    EntropyPoolingResult,
    entropy_pooling,
    mean_view_rows,
    posterior_moments,
)
from .factors import (
    FactorRiskDecomp,
    FundFactorModel,
    StatFactorModel,
    factor_exposure,
    factor_risk_decomposition,
    fundamental_factors,
    rolling_factor_exposure,
    statistical_factor_model,
)
from .model_selection import (
    CombinatorialPurgedCV,
    CVResult,
    GridSearchResult,
    WalkForward,
    cross_validate,
    grid_search,
)
from .optimization import (
    ARCResult,
    BLResult,
    Constraints,
    OptResult,
    QUBOResult,
    alpha_risk_cost_optimize,
    black_litterman,
    build_permutation_qubo,
    build_portfolio_qubo,
    equal_weight,
    hierarchical_equal_risk_contribution,
    hierarchical_risk_parity,
    max_diversification,
    max_sharpe,
    mean_cvar,
    mean_variance,
    min_tracking_error,
    min_variance,
    nested_clustered_optimization,
    quantum_inspired_hrp,
    qubo_portfolio_selection,
    risk_parity,
    robust_mvo,
    schur_complementary_allocation,
    turnover_penalized,
)
from .preselection import (
    drop_highly_correlated,
    drop_zero_variance,
    select_complete_assets,
    select_k_extremes,
    select_non_dominated,
)
from .quantum_backends import (
    BraketSolver,
    ClassicalBruteForceSolver,
    QUBOSolver,
    SimulatedAnnealingSolver,
    get_solver,
)
from .risk_measures import (
    MeanRiskResult,
    RiskMeasure,
    all_risk_measures,
    compute_risk,
    mean_risk_optimize,
)

__all__ = [
    # Covariance
    "sample_cov", "ledoit_wolf", "oracle_approximating", "ewma_cov",
    "factor_model_cov", "denoise_mp", "gerber_cov", "detone_cov",
    # Risk measures + mean-risk optimization
    "RiskMeasure", "compute_risk", "all_risk_measures", "mean_risk_optimize", "MeanRiskResult",
    # Entropy pooling (Meucci fully-flexible views)
    "entropy_pooling", "EntropyPoolingResult", "mean_view_rows", "posterior_moments",
    # Copulas + synthetic scenarios
    "pseudo_observations", "GaussianCopula", "StudentTCopula", "ClaytonCopula",
    "GumbelCopula", "fit_copula", "synthetic_returns",
    # Clustering / nested allocators
    "hierarchical_equal_risk_contribution",
    "nested_clustered_optimization",
    "schur_complementary_allocation",
    # Pre-selection transformers
    "drop_zero_variance", "select_complete_assets", "drop_highly_correlated",
    "select_k_extremes", "select_non_dominated",
    # Model selection / cross-validation
    "WalkForward", "CombinatorialPurgedCV", "cross_validate", "grid_search",
    "CVResult", "GridSearchResult",
    "LedoitWolfResult", "FactorCovResult",
    # Factors
    "statistical_factor_model", "fundamental_factors",
    "factor_exposure", "factor_risk_decomposition", "rolling_factor_exposure",
    "StatFactorModel", "FundFactorModel", "FactorRiskDecomp",
    # Optimization
    "mean_variance", "min_variance", "max_sharpe", "risk_parity", "equal_weight",
    "black_litterman", "hierarchical_risk_parity", "mean_cvar",
    "robust_mvo", "max_diversification", "min_tracking_error", "turnover_penalized",
    "alpha_risk_cost_optimize",
    "quantum_inspired_hrp", "qubo_portfolio_selection",
    "build_portfolio_qubo", "build_permutation_qubo",
    "OptResult", "BLResult", "ARCResult", "QUBOResult", "Constraints",
    # Convex backend (optional, `pip install cpz-quant[cvx]`)
    "has_cvxpy", "mean_variance_cvx", "mean_cvar_cvx",
    "robust_mean_variance_cvx", "cardinality_constrained_cvx",
    # Quantum backends
    "QUBOSolver", "ClassicalBruteForceSolver", "SimulatedAnnealingSolver",
    "BraketSolver", "get_solver",
    # Costs
    "almgren_chriss", "linear_impact", "sqrt_impact", "spread_cost",
    "total_cost", "turnover_analysis",
    "alpha_decay_capacity", "CapacityEstimate",
    "CostEstimate", "TurnoverResult", "CostModel",
    # Attribution
    "brinson_fachler", "factor_attribution", "risk_attribution",
    "alpha_beta_decomposition", "rolling_attribution",
    "BrinsonResult", "FactorAttrResult", "RiskAttrResult", "AlphaBetaResult",
]
