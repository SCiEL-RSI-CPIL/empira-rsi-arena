"""Independent small-instance oracles, adversarial controls, and all 30 seeds."""
from __future__ import annotations

import copy
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, permutations
import json
from math import gcd, isqrt, lcm, log, prod
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'suite/_lib'),str(ROOT/'tests')]
from autoresearch_suite import science_prize_a as v, runtime
from suite_test_support import specifications

# The generator script's table (name -> direction, contract, seed) is not part of this library: it is read from the packages.
SPECS={spec['key']:(spec['direction'],spec['contract'],spec['seed']) for spec in specifications('prize_a')}


def accepted(function, candidate, **params):
    try:
        return function(candidate,**params)[0]
    except runtime.CandidateError:
        return None


def subsets(n):
    for mask in range(1,1<<n):
        yield [x+1 for x in range(n) if mask>>x&1]


def ap(row):
    return len(row)<3 or len({row[i+1]-row[i] for i in range(len(row)-1)})==1


def prime(x):
    return x>=2 and not any(x%d==0 for d in range(2,x))


def mutate_first_integer(x,new):
    if type(x) is int:return new,True
    if type(x) in (list,tuple):
        row=list(x)
        for i,value in enumerate(row):
            row[i],changed=mutate_first_integer(value,new)
            if changed:return row,True
    return x,False


class SeedsAndContractTypes(unittest.TestCase):
    def test_all_thirty_seeds_and_known_scores(self):
        # practical_divisor_sum (fifth entry): the evaluator audit replaced its seed, which attained the optimum 5,
        # by a deliberately weak one that scores 49.
        expected=[2048,2.713797159083181,65,89,49,44,33,12,9,17,152,367,9,15,13,125,13,3276,21,372,13,7,7,6,19,
                  40/256,log(9)/log(6),5/25,1/16,249/254]
        self.assertEqual(len(SPECS),30)
        for (name,(_,_,source)),score in zip(SPECS.items(),expected):
            with self.subTest(family=name):
                namespace={};exec(source,namespace)
                actual,details=getattr(v,name)(namespace['construct']())
                self.assertEqual(actual,score)
                self.assertIs(details['lean_verified'],False)

    def test_every_family_rejects_bool_float_string_and_missing_certificate(self):
        for name,(_,_,source) in SPECS.items():
            ns={};exec(source,ns);candidate=ns['construct']()
            for bad in [True,1.0,'1']:
                malformed,changed=mutate_first_integer(copy.deepcopy(candidate),bad)
                self.assertTrue(changed)
                with self.subTest(family=name,bad=bad),self.assertRaises(runtime.CandidateError):
                    getattr(v,name)(malformed)
            with self.subTest(family=name,bad=None),self.assertRaises(runtime.CandidateError):
                getattr(v,name)(None)

    def test_all_catalog_ids_and_sources_are_distinct_and_match_registrations(self):
        records=json.loads((ROOT/'catalog/science/prize_a.json').read_text())
        self.assertEqual(len(records),30)
        self.assertEqual(len({r['id'] for r in records}),30)
        self.assertEqual({r['family'] for r in records},{'science_prize_a_'+x for x in SPECS})
        for r in records:
            self.assertTrue(any(s['type']=='paper' for s in r['sources']))
            self.assertTrue(any('bcf1866ea9a4ae82b32bd95b82ffd812b9ee80d2' in s['url'] for s in r['sources']))
            self.assertFalse(r['provenance']['upstream_evaluator_reused'])


class ArithmeticOracles(unittest.TestCase):
    def test_goldbach_against_trial_division(self):
        for n in range(4,42,2):
            for a in range(1,n):
                b=n-a
                good=prime(a) and prime(b)
                self.assertEqual(accepted(v.goldbach_decompositions,[[a,b]],start=n,count=1),1 if good else None)
        self.assertEqual(v.goldbach_decompositions([[2,2],None],start=4,count=2)[0],0.5)
        self.assertIsNone(accepted(v.goldbach_decompositions,[None],start=4,count=1))
        self.assertIsNone(accepted(v.goldbach_decompositions,[[2,2]],start=6,count=1))

    def test_abc_radical_by_independent_prime_enumeration(self):
        for a in range(1,20):
            for b in range(a,25):
                c=a+b
                value=accepted(v.abc_quality,[a,b,c],maximum=50)
                if gcd(a,b)!=1:
                    self.assertIsNone(value);continue
                radical=prod(p for p in range(2,c+1) if prime(p) and (a*b*c)%p==0)
                self.assertEqual(value,log(c)/log(radical))
        self.assertIsNone(accepted(v.abc_quality,[1,8,10]))
        self.assertIsNone(accepted(v.abc_quality,[0,1,1]))
        self.assertIsNone(accepted(v.abc_quality,[8,1,9]))
        self.assertEqual(v.abc_quality([1,8,9])[1]['radical'],6)

    def test_hadamard_orthogonality_against_determinant_bound(self):
        def determinant(matrix):
            return sum((-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))*prod(matrix[i][p[i]] for i in range(4)) for p in permutations(range(4)))
        rng=random.Random(13)
        for _ in range(256):
            matrix=[[rng.choice([-1,1]) for _ in range(4)] for _ in range(4)]
            good=abs(determinant(matrix))==16
            self.assertEqual(accepted(v.hadamard_orders,[matrix],maximum=4),1 if good else None)
        h=[[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
        self.assertEqual(v.hadamard_orders([h],maximum=4)[0],1)
        self.assertEqual(v.hadamard_orders([h,None],maximum=8)[0],0.5)
        h[0][0]=-1
        self.assertIsNone(accepted(v.hadamard_orders,[h],maximum=4))
        self.assertIsNone(accepted(v.hadamard_orders,[None],maximum=4))

    def test_lonely_runner_exact_rational_distances(self):
        speeds=[1,2,5]
        for q in range(2,35):
            for p in range(1,q):
                distances=[]
                for speed in speeds:
                    x=Fraction(speed*p,q)
                    distances.append(min(abs(x-k) for k in range(speed+1)))
                expected=min(distances)
                score,details=v.lonely_runner([p,q],speeds=speeds)
                self.assertEqual(Fraction(details['numerator'],details['denominator']),expected)
                self.assertEqual(score,float(expected))
        for row in [[0,2],[2,2],[3,2],[1,1],[1,1000001]]:
            self.assertIsNone(accepted(v.lonely_runner,row))

    def test_three_distinct_egyptian_fractions_against_fraction_oracle(self):
        for n in range(3,11):
            for triple in combinations(range(1,18),3):
                good=sum((Fraction(1,x) for x in triple),Fraction())==Fraction(4,n)
                self.assertEqual(accepted(v.egyptian_four,[triple],start=n,count=1),1 if good else None)
        self.assertEqual(v.egyptian_four([[1,4,12]],start=3,count=1)[0],1)
        self.assertIsNone(accepted(v.egyptian_four,[[3,3,3]],start=4,count=1))
        self.assertIsNone(accepted(v.egyptian_four,[None],start=3,count=1))
        self.assertIsNone(accepted(v.egyptian_four,[[1,4,13]],start=3,count=1))

    def test_distinct_subset_sums_exhaustively(self):
        for row in combinations(range(1,10),4):
            sums=[sum(row[i] for i in range(4) if mask>>i&1) for mask in range(16)]
            expected=max(row) if len(set(sums))==16 else None
            self.assertEqual(accepted(v.distinct_subset_sums,row,n=4,maximum=9),expected)

    def test_harmonic_and_cardinality_ap_constraints(self):
        for row in subsets(7):
            good=not any(ap(t) for t in combinations(row,3))
            score=accepted(v.harmonic_progression_free,row,n=7,k=3)
            self.assertEqual(score,float(sum((Fraction(1,x) for x in row),Fraction())) if good else None)
            self.assertEqual(accepted(v.progression_free,row,n=7,k=3),len(row) if good else None)

    def test_property_p_including_repeated_summands(self):
        for row in subsets(7):
            good=all((b+c)%a!=0 for a in row for b in row for c in row if a<b and a<c)
            self.assertEqual(accepted(v.property_p,row,n=7),len(row) if good else None)
        self.assertIsNone(accepted(v.property_p,[2,3],n=3))

    def test_cluster_primes_exhaustively_small(self):
        for n in range(3,80):
            good=prime(n) and all(any(prime(q) and prime(q-d) for q in range(d+2,n+1)) for d in range(2,n-2,2))
            self.assertEqual(accepted(v.cluster_prime,n,maximum=80),n if good else None)

    def test_practical_divisor_sum_minimum_independently(self):
        divisors=[d for d in range(1,25) if 24%d==0]
        solutions=[r for count in range(1,len(divisors)+1) for r in combinations(divisors,count) if sum(r)==17]
        scores=[v.practical_divisor_sum(r,n=24,target=17)[0] for r in solutions]
        self.assertEqual(min(scores),3)
        for bad in [[1,4,12,12],[1,3,13],[1,4,8]]:
            self.assertIsNone(accepted(v.practical_divisor_sum,bad,n=24,target=17))

    def test_b3_exhaustive_repeated_triple_collisions(self):
        for row in subsets(7):
            sums={}
            good=True
            for i in range(len(row)):
                for j in range(len(row)):
                    for k in range(len(row)):
                        total=row[i]+row[j]+row[k];triple=tuple(sorted((i,j,k)))
                        if total in sums and sums[total]!=triple:good=False
                        sums[total]=triple
            self.assertEqual(accepted(v.b3_set,row,n=7),len(row) if good else None)
        self.assertIsNone(accepted(v.b3_set,[1,2,3],n=7))

    def test_pair_sum_prime_diversity(self):
        for row in combinations(range(1,9),4):
            product=1
            for a,b in combinations(row,2):product*=a+b
            factors={p for p in range(2,17) if prime(p) and product%p==0}
            self.assertEqual(v.pair_sum_prime_diversity(row,n=4,maximum=8)[0],len(factors))
        self.assertEqual(v.pair_sum_prime_diversity([1,2],n=2,maximum=2)[0],1)

    def test_nondividing_k_exact_other_element_convention(self):
        for row in subsets(7):
            good=all(sum(t)%a for t in combinations(row,3) for a in row if a not in t)
            self.assertEqual(accepted(v.nondividing_k_sum,row,n=7,k=3),len(row) if good else None)
        self.assertIsNone(accepted(v.nondividing_k_sum,[2,3,4,5],n=7,k=3))

    def test_antichain_certificates(self):
        self.assertEqual(v.antichain_representations([[1],[2],[3],[4],[5]],targets=[1,2,3,4,5])[0],5)
        self.assertEqual(v.antichain_representations([[3,4],[5,6]],targets=[7,11])[0],4)
        self.assertEqual(v.antichain_representations([[12]],targets=[12])[0],1)
        self.assertEqual(v.antichain_representations([[3,4,5]],targets=[12])[0],3)
        self.assertIsNone(accepted(v.antichain_representations,[None,[5,6]],targets=[7,11]))
        for row in [[1,2,4],[7],[2,2,3],[1,6]]:
            self.assertIsNone(accepted(v.antichain_representations,[row],targets=[7]))
        self.assertIsNone(accepted(v.antichain_representations,[None],targets=[7]))
        allowed={1,2,3,4,5,6,8,9,10,12,15,16}
        for size in range(1,5):
            for row in combinations(range(1,17),size):
                target=sum(row)
                if target>16:continue
                good=set(row)<=allowed and all(gcd(a,b)!=min(a,b) for a,b in combinations(row,2))
                self.assertEqual(accepted(v.antichain_representations,[row],targets=[target]),size if good else None)

    def test_sum_product_against_two_dimensional_table(self):
        for row in combinations(range(1,9),4):
            sums=set();products=set()
            for i in range(4):
                for j in range(i,4):sums.add(row[i]+row[j]);products.add(row[i]*row[j])
            self.assertEqual(v.sum_product(row,n=4,maximum=8)[0],max(len(sums),len(products)))

    def test_squarefree_gap_oracle(self):
        sf=[x for x in range(1,201) if all(x%(p*p) for p in range(2,x+1) if prime(p))]
        valid=set(zip(sf,sf[1:]))
        for a in range(1,100):
            for b in range(a+1,min(a+9,201)):
                self.assertEqual(accepted(v.squarefree_gap,[a,b],maximum=200),b-a if (a,b) in valid else None)

    def test_two_square_gap_by_exact_table(self):
        values=sorted({x*x+y*y for x in range(15) for y in range(15) if 1<=x*x+y*y<=200})
        valid=set(zip(values,values[1:]))
        for a in range(1,100):
            for b in range(a+1,min(a+9,201)):
                self.assertEqual(accepted(v.two_square_gap,[a,b],maximum=200),b-a if (a,b) in valid else None)
        self.assertEqual(v.two_square_gap([2,4],maximum=4)[0],2) # zero square at right endpoint


class CombinatorialOracles(unittest.TestCase):
    def test_sunflowers_exhaustive_small_families(self):
        ground=list(combinations(range(1,6),2))
        for family in combinations(ground,3):
            sets=[set(s) for s in family]
            common=set.intersection(*sets)
            sunflower=all(a&b==common for a,b in combinations(sets,2))
            self.assertEqual(accepted(v.sunflower_free,family,n=5,k=2),None if sunflower else 3)
        self.assertIsNone(accepted(v.sunflower_free,[[1,2],[3,4],[5,6]],n=6,k=2))
        self.assertIsNone(accepted(v.sunflower_free,[[1,2],[1,2]],n=5,k=2))

    def test_basis_counts_and_prime_complements(self):
        for row in subsets(7):
            a=[x-1 for x in row]
            counts={t:sum(x+y==t for x,y in combinations_with_replacement(a,2)) for t in range(6,13)}
            expected=max(counts.values()) if all(counts.values()) else None
            self.assertEqual(accepted(v.bounded_basis,a,n=6),expected)
            good=all(any(prime(t-x) for x in a) for t in range(6,13))
            self.assertEqual(accepted(v.prime_complement,a,n=6),len(a) if good else None)

    def test_paired_sidon_independent_positive_difference_criterion(self):
        choices=list(subsets(5))
        for a in choices:
            for b in choices:
                differences=[[y-x for x,y in combinations(row,2)] for row in (a,b)]
                good=all(len(d)==len(set(d)) for d in differences) and not set(differences[0])&set(differences[1])
                self.assertEqual(accepted(v.paired_sidon,[a,b],n=5),len(a)+len(b) if good else None)

    def test_vdw_and_at_least_three_color_condition(self):
        rng=random.Random(142)
        aps=[t for t in combinations(range(8),4) if ap(t)]
        for _ in range(80):
            colors=[rng.randrange(2) for _ in range(8)]
            good=all(len({colors[x] for x in t})>1 for t in aps)
            self.assertEqual(accepted(v.van_der_waerden,colors,maximum=8),8 if good else None)
            colors=[rng.randrange(4) for _ in range(8)]
            good=all(len({colors[x] for x in t})>=3 for t in aps)
            self.assertEqual(accepted(v.three_color_progressions,colors,n=8),len(set(colors)) if good else None)

    def test_triple_free_every_subset(self):
        for row in subsets(9):
            good=not any(b==2*a and c==3*a for a,b,c in combinations(row,3))
            self.assertEqual(accepted(v.triple_free,row,n=9),len(row) if good else None)

    def test_discrepancy_against_all_progression_subsets(self):
        rng=random.Random(171)
        all_aps=[t for k in range(1,9) for t in combinations(range(8),k) if ap(t)]
        for _ in range(100):
            signs=[rng.choice((-1,1)) for _ in range(8)]
            expected=max(abs(sum(signs[x] for x in t)) for t in all_aps)
            self.assertEqual(v.progression_discrepancy(signs,n=8)[0],expected)
        self.assertEqual(v.progression_discrepancy([1]*8,n=8)[0],8)

    def test_many_short_progressions_exhaustive(self):
        for row in subsets(8):
            good=not any(ap(t) for t in combinations(row,4))
            count=sum(ap(t) for t in combinations(row,3))
            self.assertEqual(accepted(v.many_short_progressions,row,n=8),count if good else None)

    def test_nonaveraging_bitsets_against_subsets(self):
        for row in subsets(9):
            good=all(sum(t)!=len(t)*a for k in range(2,len(row)) for t in combinations(row,k) for a in row if a not in t)
            self.assertEqual(accepted(v.nonaveraging,row,n=9),len(row) if good else None)
        self.assertIsNone(accepted(v.nonaveraging,[1,2,4,9],n=9)) # 4=(1+2+9)/3

    def test_permutation_ap_both_directions_and_nonadjacent_positions(self):
        for row in permutations(range(1,6)):
            expected=max(k for k in range(1,6) for indexes in combinations(range(5),k) if ap([row[i] for i in indexes]))
            self.assertEqual(v.permutation_progression(row,n=5)[0],expected)
        self.assertEqual(v.permutation_progression([5,4,3,2,1],n=5)[0],5)
        self.assertEqual(v.permutation_progression([1,5,2,4,3],n=5)[0],3)

    def test_congruence_classes_by_period_enumeration(self):
        rng=random.Random(189)
        for _ in range(100):
            moduli=rng.sample(range(2,17),3)
            classes=[[rng.randrange(m),m] for m in moduli]
            period=lcm(*moduli)
            good=all(sum(x%m==r for r,m in classes)<=1 for x in range(period))
            self.assertEqual(accepted(v.disjoint_congruences,classes,maximum=16),3 if good else None)
        self.assertIsNone(accepted(v.disjoint_congruences,[[0,2],[0,3]],maximum=6))
        self.assertIsNone(accepted(v.disjoint_congruences,[[0,2],[1,2]],maximum=6))


if __name__=='__main__':unittest.main()
