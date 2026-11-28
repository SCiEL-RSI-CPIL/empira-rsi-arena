"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def hindcast_waves(train, test):
    """Per buoy, least-squares fit of wave height to the square of the concurrent wind speed."""
    out = {}
    for buoy, past in train.items():
        ok = np.isfinite(past["wvht"]) & np.isfinite(past["wspd"])
        x = np.column_stack([np.ones(ok.sum()), past["wspd"][ok] ** 2])
        a, b = np.linalg.lstsq(x, past["wvht"][ok], rcond=None)[0]
        now = test[buoy]
        index = now["target_index"].astype(int)
        wind = now["wspd"][index]
        wind = np.where(np.isfinite(wind), wind, np.nanmean(past["wspd"]))
        out[buoy] = np.clip(a + b * wind ** 2, 0.0, 20.0)
    return out
# EVOLVE-BLOCK-END
