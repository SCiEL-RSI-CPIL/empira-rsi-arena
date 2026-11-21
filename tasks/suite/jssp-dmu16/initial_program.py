"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def solve(inst):
    # Greedy list scheduling: always start the operation that can begin earliest, ties to the job with most work left.
    jobs = inst["jobs"]
    n = len(jobs)
    nxt, job_ready, mach_ready = [0] * n, [0] * n, {}
    remaining = [sum(d for _, d in j) for j in jobs]
    starts = [[0] * len(j) for j in jobs]
    for _ in range(sum(len(j) for j in jobs)):
        best = None
        for j in range(n):
            k = nxt[j]
            if k >= len(jobs[j]):
                continue
            m, d = jobs[j][k]
            s = max(job_ready[j], mach_ready.get(m, 0))
            key = (s, -remaining[j])
            if best is None or key < best[0]:
                best = (key, j, s)
        _, j, s = best
        m, d = jobs[j][nxt[j]]
        starts[j][nxt[j]] = s
        job_ready[j] = mach_ready[m] = s + d
        remaining[j] -= d
        nxt[j] += 1
    return starts
# EVOLVE-BLOCK-END
