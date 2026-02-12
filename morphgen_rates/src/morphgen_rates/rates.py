from pyomo.environ import *
import numpy as np

_Bif_Var_Penalty = 1.0


def _f_var_term(bi, gi, Zj, Zi, Vi, bin_size):    
    if np.isclose(gi, 0):
        val = 2 * bi * Zi * bin_size
    else:
        val = (2 * bi - gi) * Zj * (Zj - Zi) / (Zi * gi) + Vi * ((Zj / Zi)**2 - 1)
    return val


def _f_var(beta, gamma, Z, V, bin_size, jmax):
    Vi = V[0]
    for j in range(1, jmax + 1):
        Vi = _f_var_term(beta[j - 1], gamma[j - 1], Z[j], Z[j - 1], Vi, bin_size)
    return Vi


def _mk_objective(beta, gamma, Z, V, bin_size):
    terms = []            
    for i in range(1, gamma.size + 1):
        terms.append(_f_var(beta, gamma, Z, V, bin_size, i)**2)
    return terms


def _mk_bif_mean_constraint(beta, Z, gamma, bin_size, n_bif):
    """
    Add the mean constraint:
      sum_{i=1..len(Z)-1} term_i  ==  n_bif[0]
    where
      term_i = b[i-1] * Z[i-1] * bin_size              if gamma[i-1] ~ 0
             = b[i-1] * (Z[i] - Z[i-1]) / gamma[i-1]  otherwise
    """
    mean_terms = []
    for i in range(1, len(Z)):
        if np.isclose(gamma[i - 1], 0.0):
            term = beta[i - 1] * Z[i - 1] * bin_size
        else:
            term = beta[i - 1] * (Z[i] - Z[i - 1]) / gamma[i - 1]

        mean_terms.append(term)

    return mean_terms



def _mk_bif_var_terms(beta, Z, gamma, V, bin_size):
    """
    Build and return the variance terms for i=1..len(Z)-1

    For each i:
      - If gamma[i-1] ~ 0, use the "zero-gamma" formula
      - Else, use the "nonzero-gamma" formula

    Returns
      var_terms: list of term expressions (same order as i=1..)
    """
    var_terms = []

    for i in range(1, len(Z)):
        gi = gamma[i - 1]
        bi = beta[i - 1]
        
        vf = _f_var(beta, gamma, Z, V, bin_size, i)

        if np.isclose(gi, 0.0):
            term1 = bi**2 * ((2.0 / 3.0) * bi * Z[i - 1] * bin_size**3 + vf * bin_size**2)
            term2 = bi * Z[i - 1] * bin_size
            term = term1 + term2
        else:
            inner1 = (2 * bi - gi) * (Z[i]**2 - 2 * gi * bin_size * Z[i] * Z[i - 1] - Z[i - 1]**2) / (Z[i - 1] * gi**3)

            inner2 = vf * (Z[i]**2 - 2 * Z[i - 1] + Z[i - 1]) / (Z[i - 1]**2 * gi**2)

            term1 = bi**2 * (inner1 + inner2)
            term2 = bi * (Z[i] - Z[i - 1]) / gi
            term = term1 + term2

        var_terms.append(term)

    return var_terms



def _mk_bif_covar_terms(beta, Z, gamma, V, bin_size):
    """
    Build covariance terms for all (i, j) with 1 <= i < j <= len(Z)-1

    Each term:
      b[i-1] * b[j-1] * term1(i) * term2(j) * (Z[j-1]/Z[i-1]) * variance_function(..., i-1)

    where
      term1(i) = bin_size                              if gamma[i-1] ~ 0
               = (Z[i] - Z[i-1]) / gamma[i-1]         otherwise
      term2(j) = bin_size                              if gamma[j-1] ~ 0
               = (Z[j] - Z[j-1]) / gamma[j-1]         otherwise
    """
    cov_terms = []

    for j in range(1, len(Z)):
        for i in range(1, j):
            gj = gamma[j - 1]
            bj = beta[j - 1]
            
            gi = gamma[i - 1]
            bi = beta[i - 1]

            term1 = bin_size if np.isclose(gi, 0.0) else (Z[i] - Z[i - 1]) / gi
            
            term2 = bin_size if np.isclose(gj, 0.0) else (Z[j] - Z[j - 1]) / gj
            
            term3 = Z[j - 1] / Z[i - 1]
            
            vf = _f_var(beta, gamma, Z, V, bin_size, i)

            cov_terms.append(bi * bj * term1 * term2 * term3 * vf)

    return cov_terms



def compute_rates(data, max_step_size):
    """
    Compute bifurcation and annihilation rates from summary statistics.

    The estimator expects Sholl-plot summary statistics (mean and variance per
    radial bin) and summary statistics of bifurcation counts (mean and variance).
    These quantities are used to infer the event rates of a branching-and-
    annihilating process.

    Parameters
    ----------
    data : dict
      Input container with the following structure:

      data = {
        "sholl_plot": {
          "bin_size": float,
          "mean": numpy.ndarray,   # shape (K,)
          "var":  numpy.ndarray,   # shape (K,)
        },
        "bifurcation_count": {
          "mean": float,
          "var":  float,
        },
      }

      Where:
      - `data["sholl_plot"]["bin_size"]` is the spatial bin size used to build the Sholl plot
      - `data["sholl_plot"]["mean"][i]` is the mean Sholl intersection count in bin i
      - `data["sholl_plot"]["var"][i]` is the variance of the Sholl intersection count in bin i
      - `data["bifurcation_count"]["mean"]` is the mean number of bifurcations
      - `data["bifurcation_count"]["var"]` is the variance of the number of bifurcations

    max_step_size : float
      Maximum advancement (in distance from the soma) allowed for a single
      elongation step in the model. This value bounds the radial increment used
      by the estimator and should be expressed in the same spatial units as the
      Sholl binning.

    Returns
    -------
    dict
      Dictionary containing the estimated rates and any additional derived values
      produced by the implementation. At minimum, the returned dictionary is
      expected to include:

      - "bifurcation_rate"
      - "annihilation_rate"

    Notes
    -----
    - `data["sholl_plot"]["mean"]` and `data["sholl_plot"]["var"]` must be 1D arrays of equal length
    - Variances must be non-negative
    - Ensure `bin_size` and `max_step_size` use consistent spatial units
    """    

##    Solves a QP problem using Pyomo with vector-style variable indexing.
##
##    Minimize: 0.5*x[0]^2 + x[1]^2 + x[0]*x[1] + 3*x[0]
##    Subject to: x[0] + x[1] >= 1, x[i] >= 0
##
##    Returns:
##        np.ndarray: [x[0], x[1], objective_value]

    global _Bif_Var_Penalty

    # Keep data only up to the first zero in either mean or std (whichever occurs earlier)
    bin_size = data['sholl_plot']['bin_size']
    mean = np.array(data['sholl_plot']['mean'])
    std  = np.array(data['sholl_plot']['std'])

    iz = np.minimum(
        np.flatnonzero(mean == 0)[0] if (mean == 0).any() else mean.size,
        np.flatnonzero(std  == 0)[0] if (std  == 0).any() else std.size
    )

    Z = mean[:iz]
    V = std[:iz] ** 2


    # Use bifurcation mean and variance if available; otherwise set to None
    bc = data.get('bifurcation_count')
    n_bif = [bc['mean'], bc['std'] ** 2] if bc is not None else None



    # calculate the gamma
    gamma = np.log(Z[1:] / Z[:-1]) / bin_size
    
    # Initial value for b: copy gamma, clamp negatives to 0, then map index -> value
    init_b = dict(enumerate(gamma.clip(min=0).copy()))

    # create the model 
    model = ConcreteModel()

    # Define index set and variables
    model.b = Var(range(gamma.size), domain=NonNegativeReals, initialize=init_b)

    # define 1 slack variables for eventual constraints of variance of bifurcations
    model.s = Var(range(2), domain=Reals)
            
    # Constraint: 
    model.constraints = ConstraintList()

    # create the minimum required constraints
    for i in range(gamma.size):
        model.constraints.add(model.b[i] >= gamma[i])
        model.constraints.add((2 * model.b[i] - gamma[i]) * max_step_size <= 1)

    # if we have number of bifurcations, use it as contraints
    if n_bif:        
        # constraint the average number of bifurcations
        if n_bif[0]:
            mean_terms = _mk_bif_mean_constraint(model.b, Z, gamma, bin_size, n_bif[0])
            model.constraints.add(sum(mean_terms) == n_bif[0])
        
        # constrain the variance for the number of bifurcations
        if n_bif[1]:
            var_terms = _mk_bif_var_terms(model.b, Z, gamma, V, bin_size)
            covar_terms = _mk_bif_covar_terms(model.b, Z, gamma, V, bin_size)               
            model.constraints.add(sum(var_terms + covar_terms) + model.s[0] == n_bif[1])


    # Objective
    model.obj = Objective(
        expr=sum(_mk_objective(model.b, gamma, Z, V, bin_size)) + _Bif_Var_Penalty * model.s[0] ** 2,
        sense=minimize
    )

    
    # Solve
    solver = SolverFactory('ipopt')
    solver.solve(model, tee=False)
    
    # Extract bifurcation rates as array
    b = np.array([value(model.b[i]) for i in model.b])
    b[b < 0] = 0.

    # calculate annihilation rates
    a = - gamma + b   
    a[a < 0] = 0.

    # implement barrier at the end of the sholl plots
    b = np.append(b, 0.)
    a = np.append(a, np.inf)

    return { 'bifurcation_rate':b, 'annihilation_rate':a }
