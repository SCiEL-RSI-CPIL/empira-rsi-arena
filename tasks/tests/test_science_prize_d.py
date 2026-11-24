"""Independent small-instance oracles and invalid-certificate controls."""
from __future__ import annotations

import itertools
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"suite/_lib"))
sys.path.insert(0,str(ROOT/"tests"))
from autoresearch_suite import runtime
from autoresearch_suite import science_prize_d as v
from suite_test_support import specifications

# The specification list of the generator script is not part of this library: the same fields are read from the packages.
SPECS = specifications("prize_d")


def fan(n):
    return {"edges": [(0,i) for i in range(1,n)]+[(i,i+1) for i in range(1,n-1)],
            "cycles": [list(range(k)) for k in range(3,n+1)]}


class GaussianPaths(unittest.TestCase):
    def test_two_point_exact_score(self):
        self.assertEqual(v.gaussian_path([(1,1),(59,0)])[0],3365)

    def test_all_small_gaussian_primes_against_gaussian_division(self):
        # A nonunit Gaussian integer is composite iff it has a nonunit divisor
        # whose norm is no larger than its square-root norm. This checks the
        # rational-prime characterization through a separate definition.
        for x,y in itertools.product(range(-6,7),repeat=2):
            norm=x*x+y*y
            if norm<=1:
                expected=False
            else:
                expected=True
                for a,b in itertools.product(range(-4,5),repeat=2):
                    d=a*a+b*b
                    if 1<d and d*d<=norm and (x*a+y*b)%d==0 and (y*a-x*b)%d==0:
                        expected=False
                        break
            if (x,y)==(1,1):
                continue
            path=[(1,1),(x,y)]
            if expected:
                v.gaussian_path(path,bound=6,target=0)
            else:
                with self.assertRaises(runtime.CandidateError):
                    v.gaussian_path(path,bound=6,target=0)

    def test_invalid_paths(self):
        for path in [[],[(1,1),(1,1),(59,0)],[(1,1),(53,0)],[(1,1),(6,6)],
                     [(1,1),(59.0,0)],[(True,1),(59,0)],[(1,1),(67,0)],[(2,1),(59,0)],[(1,1),(3,0)]]:
            with self.subTest(path=path),self.assertRaises(runtime.CandidateError):
                v.gaussian_path(path)


class CoprimalityGaps(unittest.TestCase):
    def test_cyclic_scan_against_every_start(self):
        for primes in itertools.combinations([2,3,5,7,11],2):
            n=math.prod(primes)
            largest=0
            for start in range(n):
                length=0
                while math.gcd(start+length,n)>1:
                    length+=1
                largest=max(largest,length)
            self.assertEqual(v.jacobsthal(n,maximum=200,prime_count=2)[0],largest)

    def test_published_five_prime_value(self):
        score,details=v.jacobsthal(2*3*5*7*11)
        self.assertEqual(score,13)
        self.assertEqual(details['jacobsthal_function'],14)

    def test_bad_moduli(self):
        for n in [True,2730.0,0,1,30,4*3*5*7*13,1000001]:
            with self.assertRaises(runtime.CandidateError):v.jacobsthal(n)


class PancyclicGraphs(unittest.TestCase):
    def test_fan_cycles_for_several_orders(self):
        for n in range(3,12):
            self.assertEqual(v.pancyclic(fan(n),n)[0],n-3)

    def test_incomplete_or_false_certificate(self):
        candidates=[]
        a=fan(6);a['cycles'].pop();candidates.append(a)
        a=fan(6);a['edges'].remove((0,2));candidates.append(a)
        a=fan(6);a['cycles'][0]=[0,1,1];candidates.append(a)
        a=fan(6);a['edges'].append((1,0));candidates.append(a)
        a=fan(6);a['edges'][0]=(False,1);candidates.append(a)
        for a in candidates:
            with self.assertRaises(runtime.CandidateError):v.pancyclic(a,6)


class TriangleAreas(unittest.TestCase):
    def test_square_has_four_equal_triangles(self):
        self.assertEqual(v.equal_area([(0,0),(0,1),(1,0),(1,1)],4)[0],4)

    def test_shoelace_oracle(self):
        rng=random.Random(724)
        for _ in range(20):
            pts=rng.sample(list(itertools.product(range(5),repeat=2)),7)
            counts={}
            for a,b,c in itertools.permutations(pts,3):
                area=abs(a[0]*b[1]+b[0]*c[1]+c[0]*a[1]-a[1]*b[0]-b[1]*c[0]-c[1]*a[0])
                if area:counts[area]=counts.get(area,0)+1
            self.assertEqual(v.equal_area(pts,7)[0],max(counts.values())//6)

    def test_degenerate_duplicate_and_inexact_points(self):
        for pts in [[(0,0),(1,1),(2,2)],[(0,0),(0,0),(1,1)],[(0,0),(0,1),(1.0,0)]]:
            with self.assertRaises(runtime.CandidateError):v.equal_area(pts,3)


class ProjectiveBlocking(unittest.TestCase):
    def test_projective_plane_axioms(self):
        for q in [2,3,7]:
            pts,lines=v.projective_plane(q)
            self.assertEqual(len(pts),q*q+q+1)
            self.assertEqual(len(set(lines)),len(pts))
            self.assertTrue(all(len(line)==q+1 for line in lines))
            for a,b in itertools.combinations(lines,2):
                self.assertEqual(len(set(a)&set(b)),1)
            for p in range(len(pts)):
                self.assertEqual(sum(p in line for line in lines),q+1)

    def test_fano_plane_all_subsets_independent_lines(self):
        # For q=2 all nonzero bit vectors are their own representatives;
        # each line is a triple {u,v,u xor v}.
        pts,_=v.projective_plane(2)
        vectors=[4*x+2*y+z for x,y,z in pts]
        lines={frozenset((i,j,vectors.index(vectors[i]^vectors[j]))) for i,j in itertools.combinations(range(7),2)}
        for mask in range(1,128):
            chosen={i for i in range(7) if mask>>i&1}
            counts=[len(chosen&line) for line in lines]
            if min(counts):self.assertEqual(v.blocking_set(sorted(chosen),2)[0],max(counts))
            else:
                with self.assertRaises(runtime.CandidateError):v.blocking_set(sorted(chosen),2)

    def test_bad_blocking_sets(self):
        for chosen in [[],[0],[0,0],[57],[True],[0.0]]:
            with self.assertRaises(runtime.CandidateError):v.blocking_set(chosen)


class Tournaments(unittest.TestCase):
    def test_all_order_four_tournaments_against_degree_oracle(self):
        n=4
        pairs=list(itertools.combinations(range(n),2))
        for bits in itertools.product((0,1),repeat=len(pairs)):
            arcs={(i,j) if b==0 else (j,i) for (i,j),b in zip(pairs,bits)}
            best=0
            for mask in range(1,1<<n):
                subset=[i for i in range(n) if mask>>i&1]
                degrees=sorted(sum((i,j) in arcs for j in subset) for i in subset)
                if degrees==list(range(len(subset))):best=max(best,len(subset))
            self.assertEqual(v.transitive_tournament(bits,n)[0],best)

    def test_cycle_and_total_order(self):
        self.assertEqual(v.transitive_tournament([0,1,0],3)[0],2)
        self.assertEqual(v.transitive_tournament([0]*190,20)[0],20)

    def test_invalid_bits(self):
        for bits in [[],[0]*189,[0]*189+[True],[0]*189+[1.0],[0]*189+[2]]:
            with self.assertRaises(runtime.CandidateError):v.transitive_tournament(bits)


class PrimeRoots(unittest.TestCase):
    def test_against_generated_multiplicative_group(self):
        for p in [3,5,7,11,13,17,19,23,29,31]:
            for g in range(2,50):
                prime=all(g%d for d in range(2,g))
                valid=prime and g%p and len({pow(g,k,p) for k in range(1,p)})==p-1
                if valid:self.assertEqual(v.prime_primitive_roots([g],[p],50)[0],g)
                else:
                    with self.assertRaises(runtime.CandidateError):v.prime_primitive_roots([g],[p],50)

    def test_modulus_itself_composite_and_lossy_input(self):
        for root in [11,6,True,2.0,100001]:
            with self.assertRaises(runtime.CandidateError):v.prime_primitive_roots([root],[11])


class FareyPairs(unittest.TestCase):
    def test_sequence_against_sorted_reduced_rationals(self):
        for order in range(2,15):
            expected=sorted({Fraction(a,b) for b in range(1,order+1) for a in range(b+1)})
            self.assertEqual(v.farey(order),tuple((f.numerator,f.denominator) for f in expected))

    def test_rank_gap_and_false_pairs(self):
        self.assertEqual(v.opposite_farey_pair([[1,4],[2,3]],4)[0],3)
        for pair in [[[2,8],[2,3]],[[2,3],[1,4]],[[1,4],[1,3]],[[0,1],[1,1]],[[1.0,4],[2,3]],[[1,5],[2,3]]]:
            with self.assertRaises(runtime.CandidateError):v.opposite_farey_pair(pair,4)


class AntiRamseyColorings(unittest.TestCase):
    def test_all_three_cycles_checked_against_permutations(self):
        pairs=list(itertools.combinations(range(4),2))
        for bits in itertools.product(range(4),repeat=6):
            matrix={edge:color for edge,color in zip(pairs,bits)}
            has_rainbow=any(len({matrix[tuple(sorted((a,b)))] for a,b in zip(cycle,cycle[1:]+cycle[:1])})==4
                            for cycle in itertools.permutations(range(4)))
            if has_rainbow:
                with self.assertRaises(runtime.CandidateError):v.rainbow_c4_avoidance(bits,4)
            else:self.assertEqual(v.rainbow_c4_avoidance(bits,4)[0],len(set(bits)))

    def test_invalid_palette(self):
        for colors in [[],[0]*5+[True],[0]*5+[1.0],[0]*5+[6]]:
            with self.assertRaises(runtime.CandidateError):v.rainbow_c4_avoidance(colors,4)


class PermutationOrders(unittest.TestCase):
    def test_cycle_type_counts_against_every_small_permutation(self):
        for n in range(1,8):
            counts=Counter()
            for perm in itertools.permutations(range(n)):
                visited=set();order=1
                for start in range(n):
                    if start in visited:continue
                    current=start;length=0
                    while current not in visited:
                        visited.add(current);length+=1;current=perm[current]
                    order=math.lcm(order,length)
                counts[order]+=1
            self.assertEqual(v.permutation_orders(n),dict(counts))
            self.assertEqual(sum(v.permutation_orders(n).values()),math.factorial(n))
            for k,count in counts.items():self.assertEqual(v.frequent_permutation_order(k,n)[0],count)

    def test_invalid_or_unattainable_orders(self):
        for k in [True,1.0,0,100,17]:
            with self.assertRaises(runtime.CandidateError):v.frequent_permutation_order(k,8)


class InverseCovers(unittest.TestCase):
    def test_small_cover_and_term_count(self):
        rows={}
        # Fermat powers are a separate inverse oracle from pow(x,-1,p).
        for length in range(1,5):
            for terms in itertools.combinations(range(1,5),length):
                target=sum(pow(x,5,7) for x in terms)%7
                rows.setdefault(target,terms)
        cover=[rows[r] for r in range(7)]
        self.assertEqual(v.modular_inverse_cover(cover,7,4)[0],max(map(len,cover)))
        for bad in [[1,1],[],[0],[5],[1.0],[True]]:
            modified=cover[:];modified[0]=bad
            with self.assertRaises(runtime.CandidateError):v.modular_inverse_cover(modified,7,4)
        with self.assertRaises(runtime.CandidateError):v.modular_inverse_cover(cover[:-1],7,4)


class AllRegisteredSeeds(unittest.TestCase):
    def test_every_seed_executes_and_has_finite_score(self):
        for s in SPECS:
            with self.subTest(task=s['id']):
                namespace={'N':s['parameters'].get('n')}
                exec(s['seed'],namespace)
                value,details=getattr(v,s['key'])(namespace['construct'](),**s['parameters'])
                self.assertTrue(math.isfinite(value))
                self.assertFalse(details['lean_verified'])


if __name__ == '__main__':unittest.main()
