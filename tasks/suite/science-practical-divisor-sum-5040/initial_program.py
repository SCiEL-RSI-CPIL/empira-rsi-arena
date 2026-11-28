"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Deliberately weak baseline: a subset-sum table over the divisors in increasing
    # order, read back so that a divisor is used only when the smaller ones cannot
    # finish. Always valid, but it spends many small divisors.
    divisors=[d for d in range(1,5041) if 5040%d==0]
    reachable=[{0}]
    for d in divisors:
        reachable.append(reachable[-1]|{s+d for s in reachable[-1] if s+d<=4979})
    remainder=4979;chosen=[]
    for i in range(len(divisors)-1,-1,-1):
        if remainder not in reachable[i]:
            chosen.append(divisors[i]);remainder-=divisors[i]
    assert remainder==0
    return chosen
# EVOLVE-BLOCK-END
