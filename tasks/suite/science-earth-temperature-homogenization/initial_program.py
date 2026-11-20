"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def snht(z):
    """Largest SNHT statistic and the first index after the most likely shift (Alexandersson 1986)."""
    n = len(z)
    k = np.arange(1, n)
    c = np.cumsum(z)[:-1]
    t = k * (c / k) ** 2 + (n - k) * ((z.sum() - c) / (n - k)) ** 2
    i = int(np.argmax(t))
    return i + 1, float(t[i])


def breakpoints(diff, threshold=10.0, min_years=5):
    """Binary segmentation of an annual difference series; returns sorted start indices of segments."""
    found, stack = set(), [(0, len(diff))]
    while stack:
        a, b = stack.pop()
        idx = np.flatnonzero(np.isfinite(diff[a:b])) + a
        if len(idx) < 2 * min_years:
            continue
        x = diff[idx]
        sd = x.std()
        if sd == 0:
            continue
        pos, stat = snht((x - x.mean()) / sd)
        if stat > threshold and min_years <= pos <= len(x) - min_years:
            cut = int(idx[pos])
            found.add(cut)
            stack += [(a, cut), (cut, b)]
    return sorted(found)


def homogenize(networks, calendar):
    out = []
    for obs in networks:
        n_st, n_mo = obs.shape
        months = n_mo // 12
        clim = np.array([[np.nanmean(obs[s, m::12]) for m in range(12)] for s in range(n_st)])
        anom = obs - np.tile(clim, months)
        result = obs.copy()
        for s in range(n_st):
            others = np.delete(anom, s, axis=0)
            reference = np.nanmedian(others, axis=0)
            d = anom[s] - reference
            mad = np.nanmedian(np.abs(d - np.nanmedian(d))) + 1e-9
            bad = np.abs(d - np.nanmedian(d)) > 6 * mad
            result[s, bad] = (np.tile(clim[s], months) + reference + np.nanmedian(d))[bad]
            d[bad] = np.nan
            yearly = np.array([np.nanmean(d[12 * y:12 * y + 12]) if np.isfinite(d[12 * y:12 * y + 12]).sum() >= 10 else np.nan
                               for y in range(months)])
            cuts = [0] + breakpoints(yearly) + [months]
            last = np.nanmean(yearly[cuts[-2]:cuts[-1]])
            for a, b in zip(cuts[:-2], cuts[1:-1]):
                shift = last - np.nanmean(yearly[a:b])
                if np.isfinite(shift):
                    result[s, 12 * a:12 * b] += shift
        out.append(result)
    return out
# EVOLVE-BLOCK-END
