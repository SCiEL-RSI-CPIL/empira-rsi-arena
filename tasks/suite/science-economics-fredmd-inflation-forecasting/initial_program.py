"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def forecast_target(history, horizons):
    """Seed: the benchmark direct autoregression with BIC lag selection (scores exactly 1)."""
    from autoresearch_suite.finance_macro import ar_bic
    return ar_bic(history, horizons)
# EVOLVE-BLOCK-END
