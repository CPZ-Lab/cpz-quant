# Convex backend

```bash
pip install "cpz-quant[cvx]"       # cvxpy programs
pip install "cpz-quant[cvx-mip]"   # + SCIP for cardinality (mixed-integer)
```

The native scipy optimisers handle most workflows. The convex backend (`cpz_quant.portfolio.convex`) exists for formulations that need a disciplined convex solver to be *exact*: hard constraint satisfaction, robust worst-case objectives, and mixed-integer cardinality.

Fail-loud policy: if cvxpy or a required mixed-integer solver is missing, these functions raise with install instructions. They never silently substitute an approximation.

## Exact mean-variance with hard constraints

```python
from cpz_quant.portfolio import Constraints
from cpz_quant.portfolio.convex import mean_variance_cvx

res = mean_variance_cvx(
    returns,
    constraints=Constraints(long_only=True, max_weight=0.30, max_turnover=0.10),
    prev_weights=current_weights,   # turnover measured against these
    l2_reg=0.001,                   # ridge penalty stabilises weights
)
```

Gross exposure and turnover are hard constraints of the quadratic program, not penalties.

## CVaR as a linear program

`mean_cvar_cvx` implements the Rockafellar-Uryasev formulation: minimise CVaR at confidence `alpha` subject to a return floor, exposure, and turnover limits. `info` reports the daily VaR and CVaR at the optimum.

## Robust mean-variance

Estimation error in expected returns is the classic cause of extreme Markowitz weights. `robust_mean_variance_cvx` optimises against the worst case inside an uncertainty set:

```python
from cpz_quant.portfolio.convex import robust_mean_variance_cvx

res = robust_mean_variance_cvx(
    returns,
    uncertainty="ellipsoidal",   # or "box"
    kappa=1.5,                   # set size; 0 recovers plain mean-variance
)
```

- **Ellipsoidal**: worst case over an ellipsoid scaled by the estimation-error covariance `cov / T`; a second-order-cone penalty.
- **Box**: worst case over per-asset intervals of `kappa` standard errors.

## Cardinality constraints

Hold at most `max_assets` names, each with at least `min_position` weight (semi-continuous, so no dust positions), solved as an exact mixed-integer quadratic program:

```python
from cpz_quant.portfolio.convex import cardinality_constrained_cvx

res = cardinality_constrained_cvx(returns, max_assets=10, min_position=0.02)
print(res.info["n_selected"], res.info["solver"])
```

Requires a mixed-integer-capable solver; `pip install "cpz-quant[cvx-mip]"` installs the open-source SCIP solver. Commercial solvers (GUROBI, MOSEK, CPLEX, XPRESS) are used automatically when installed.
