"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 28
# EVOLVE-BLOCK-START
def construct():
    # Odd-even merge sort (Knuth, TAOCP 5.2.2, Algorithm M): a valid network for any N.
    out = []
    t = max(1, (N - 1).bit_length())
    p = 1 << (t - 1)
    while p > 0:
        q, r, d = 1 << (t - 1), 0, p
        while d > 0:
            for i in range(0, N - d):
                if i & p == r:
                    out.append((i, i + d))
            d, q, r = q - p, q >> 1, p
        p >>= 1
    return out
# EVOLVE-BLOCK-END
