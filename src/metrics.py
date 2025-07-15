"""
Hydrological performance metrics used in Alzhanov et al.

Functions
---------
calculate_all_metrics(qobs, qsim)  →  (nse, kge, lnse, fhv, fms, flv)
"""

import numpy as np

def calculate_all_metrics(qobs, qsim, eps=1e-6):
    """
    Calculates a suite of standard hydrological model performance metrics.
    
    Negative simulated flows are set to zero before calculations.

    Returns
    -------
    nse : float   # Nash-Sutcliffe Efficiency
    kge : float   # Kling-Gupta Efficiency
    lnse: float   # Nash–Sutcliffe Efficiency on ln(q)
    fhv : float   # High-flow bias   (top 2% of FDC)      [%]
    fms : float   # Mid-section bias (central 68% of FDC) [%]
    flv : float   # Low-flow bias    (bottom 30% of FDC)  [%]
    """
    # 0. Guard: qobs & qsim must be 1-D np.ndarray of equal length
    qobs = np.asarray(qobs).ravel()
    qsim = np.asarray(qsim).ravel()
    assert qobs.shape == qsim.shape, "qobs and qsim must be the same length"

    # 1. Clip negative simulated flows to zero
    qsim_clip = np.where(qsim < 0, 0.0, qsim)

    # 2. Calculate overall performance metrics (NSE, KGE, logNSE)
    
    # --- NSE ---
    numerator_nse = np.sum(np.square(qobs - qsim_clip))
    denominator_nse = np.sum(np.square(qobs - np.mean(qobs)))
    if denominator_nse < eps:
        nse = -np.inf
    else:
        nse = 1.0 - (numerator_nse / denominator_nse)
        
    # --- KGE ---
    mean_obs = np.mean(qobs)
    mean_sim = np.mean(qsim_clip)
    std_obs = np.std(qobs)
    std_sim = np.std(qsim_clip)
    
    r = np.corrcoef(qobs, qsim_clip)[0, 1]
    beta = mean_sim / mean_obs if mean_obs > eps else -np.inf
    gamma = std_sim / std_obs if std_obs > eps else -np.inf
    
    if np.isinf(beta) or np.isinf(gamma):
        kge = -np.inf
    else:
        kge = 1.0 - np.sqrt((r - 1.0)**2 + (beta - 1.0)**2 + (gamma - 1.0)**2)

    # --- logNSE ---
    log_obs = np.log(qobs + eps)
    log_sim = np.log(qsim_clip + eps)
    denominator_lnse = np.sum(np.square(log_obs - np.mean(log_obs)))
    
    if denominator_lnse < eps:
        lnse = -np.inf
    else:
        lnse = 1.0 - np.sum(np.square(log_obs - log_sim)) / denominator_lnse

    # 3. Calculate FDC biases
    sort_idx = np.argsort(qobs)[::-1]
    o_sorted = qobs[sort_idx]
    s_sorted = qsim_clip[sort_idx]

    n = len(o_sorted)
    high_cut = int(np.ceil(0.02 * n))
    mid_cut = int(np.ceil(0.70 * n))

    def _bias(obs_section, sim_section):
        denominator_bias = obs_section.sum()
        if denominator_bias < eps:
            return -np.inf # Or some other indicator for zero observed flow
        return 100.0 * (sim_section.sum() - denominator_bias) / denominator_bias

    fhv = _bias(o_sorted[:high_cut], s_sorted[:high_cut])
    fms = _bias(o_sorted[high_cut:mid_cut], s_sorted[high_cut:mid_cut])
    flv = _bias(o_sorted[mid_cut:], s_sorted[mid_cut:])

    return nse, kge, lnse, fhv, fms, flv
