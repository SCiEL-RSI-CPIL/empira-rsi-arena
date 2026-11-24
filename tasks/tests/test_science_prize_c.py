"""Independent small oracles, published mathematical witnesses and hostile inputs.

Published witness credit: Alon (1985) Fano plane; Bui–Pratt–Zaharescu
(2024) t_2,t_3,t_5; Cipollini (2026) two-interval family; Gaitanas (2019)
powers-of-two complements. No implementation from those sources is reused.
"""
from __future__ import annotations

import copy
import importlib.util
from itertools import combinations, permutations, product
import json
from math import gcd, isfinite, isqrt, prod
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'suite/_lib'))
sys.path.insert(0,str(ROOT/'tests'))
from autoresearch_suite import runtime, science_prize_c as v
from suite_test_support import registered, source_file

# The case table of the generator script is not part of this library: seed and parameters of each case are read from
# its package, whose seed starts with the same parameters as P.
CASES={family.removeprefix('science_prize_c_'):{'seed':spec['seed'],'params':spec['parameters']} for family,spec in registered('prize_c').items()}


def seed(number):
    case=CASES[str(number)]
    ns={'P':copy.deepcopy(case['params'])}
    exec(case['seed'],ns)
    return ns['construct']()


def score(number,candidate,params):
    return v.evaluate(number,candidate,params)[0]


def mutate_first_int(x,replacement):
    if type(x) is int:return replacement,True
    if isinstance(x,list):
        for i,a in enumerate(x):
            b,done=mutate_first_int(a,replacement)
            if done:x[i]=b;return x,True
    if isinstance(x,dict):
        for key,a in x.items():
            b,done=mutate_first_int(a,replacement)
            if done:x[key]=b;return x,True
    return x,False


def powerset(values):
    return [list(s) for k in range(len(values)+1) for s in combinations(values,k)]


class PrizeC(unittest.TestCase):
    def reject(self,j,x,p):
        with self.assertRaises(runtime.CandidateError):v.evaluate(j,x,p)

    def test_all_29_seeds_and_strict_types(self):
        self.assertEqual(len(CASES),29)
        self.assertEqual(set(map(int,CASES)),set(v.VERIFIERS))
        for number,case in CASES.items():
            j=int(number);p=case['params'];x=seed(j)
            with self.subTest(problem=j):
                value,details=v.evaluate(j,x,p)
                self.assertTrue(isfinite(value));self.assertIsInstance(details,dict)
                self.assertFalse(details['lean_verified'])
                self.assertTrue(details['finite_instance_only'])
                self.assertEqual((value,details),v.evaluate(j,seed(j),p))
                for invalid in (None,{},'false'):
                    self.reject(j,invalid,p)
                for replacement in (True,1.0,-10**100,10**100):
                    bad,changed=mutate_first_int(copy.deepcopy(x),replacement)
                    self.assertTrue(changed);self.reject(j,bad,p)

    def test_catalog_provenance_evidence_and_family_ownership(self):
        records=json.loads((ROOT/'catalog/science/prize_c.json').read_text())
        evidence=json.loads(source_file(self,'validation/prize-expansion/c-evidence.json').read_text())
        audit=json.loads(source_file(self,'validation/prize-feasibility-audit.json').read_text())['tasks']
        by_id={r['id']:r for r in audit}
        families=[]
        spec=importlib.util.spec_from_file_location('science_evidence_c_test',source_file(self,'scripts/science_evidence.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for r in records:
            self.assertTrue(r['id'].startswith('science-'));self.assertEqual(r['provenance']['category'],'independent_reimplementation')
            self.assertFalse(r['provenance']['upstream_evaluator_reused'])
            self.assertEqual(r['requirements']['dependencies'],[])
            row=by_id[r['provenance']['prize_problem_id']]
            self.assertTrue(row['new_ready']);family=row['planning_family'];families.append(family)
            earliest=min(x['id'] for x in audit if x['new_ready'] and x['planning_family']==family)
            self.assertTrue('JSP-000501'<=earliest<='JSP-000750')
            self.assertTrue(any(s['type']=='paper' for s in r['sources']))
            self.assertEqual(module.assess(r,overrides=evidence)['status'],evidence[r['id']]['status'])
        self.assertEqual(len(records),29);self.assertEqual(len(set(families)),29)

    def test_501_augmentation_triangles_and_diameter(self):
        p={'n':5,'edges':[[0,1],[1,2],[2,3],[3,4]],'diameter':2}
        self.assertEqual(score(501,[[0,3],[1,4]],p),2)
        self.reject(501,[[0,2]],p);self.reject(501,[],p)
        self.reject(501,[[0,1]],p)

    def test_506_homogeneous_partitions_and_independent_chromatic_oracle(self):
        es=list(combinations(range(4),2))
        for mask in range(1<<len(es)):
            chosen=[e for i,e in enumerate(es) if (mask>>i)&1]
            expected=next(k for k in range(1,5) if any(all(c[a]!=c[b] for a,b in chosen) for c in product(range(k),repeat=4)))
            self.assertEqual(v.chromatic_number(set(chosen),4),expected)
        p={'n':4,'edges':[[0,1],[1,2],[2,3],[0,3]]}
        self.assertEqual(score(506,[[0,2],[1,3]],p),2)
        self.reject(506,[[0,1,2],[3]],p);self.reject(506,[[0,2],[1]],p)

    def test_518_induced_degrees(self):
        p={'n':5,'edges':[[0,1],[1,2],[2,3],[3,4],[0,4]],'homogeneous_cap':3,'minimum_vertices':3}
        self.assertEqual(score(518,[0,1,2],p),2)
        self.reject(518,[0,1],p);self.reject(518,[0,1,1],p)

    def test_521_shared_vertex_disjoint_cycle_packing(self):
        p={'n':5,'edges':list(combinations(range(5),2))}
        a=[0,1,2,3,4];b=[0,2,4,1,3]
        self.assertEqual(score(521,[a,b],p),2)
        self.reject(521,[a,a],p);self.reject(521,[a,[0,1,2]],p)

    def test_554_consecutive_gap_and_least_factor_oracle(self):
        for n in range(2,1001):
            # Independent sieve-based least divisor rather than square-root search.
            expected=next(d for d in range(2,n+1) if n%d==0)
            self.assertEqual(v.least_prime_factor(n),expected)
        p={'gaps':[[3,5],[7,11]]}
        self.assertEqual(score(554,[4,None],p),1)
        self.reject(554,[4,9],p);self.reject(554,[3,None],p)
        self.reject(554,[None,None],p)
        self.reject(554,[9],{'gaps':[[5,11]]})

    def test_572_all_small_families_against_bit_masks(self):
        triples=list(combinations(range(5),3))
        masks=[sum(1<<i for i in t) for t in triples]
        for selection in range(1,1<<len(triples)):
            indices=[i for i in range(len(triples)) if selection>>i&1]
            good=all((masks[a]&masks[b]).bit_count()!=1 for a,b in combinations(indices,2))
            candidate=[list(triples[i]) for i in indices]
            if good:self.assertEqual(score(572,candidate,{'n':5,'r':3}),len(candidate))
            else:self.reject(572,candidate,{'n':5,'r':3})

    def test_578_multiple_span_and_invalid_matching(self):
        p={'divisors':[2,3,5],'maximum':30}
        self.assertEqual(score(578,[4,3,5],p),2)
        # Three distinct integer representatives cannot span less than two.
        possibilities=[x for x in product(range(2,31,2),range(3,31,3),range(5,31,5)) if len(set(x))==3]
        self.assertEqual(min(max(x)-min(x) for x in possibilities),2)
        self.reject(578,[6,6,5],p);self.reject(578,[4,5,6],p)

    def test_580_exact_matching_length_against_assignment_enumeration(self):
        for n in range(1,5):
            for m in range(7):
                def possible(length):
                    choices=[[x for x in range(m+1,m+length+1) if x%d==0] for d in range(1,n+1)]
                    return any(len(set(row))==n for row in product(*choices))
                reference=next(length for length in range(1,n*n+1) if possible(length))
                self.assertEqual(v.matching_window(n,m),reference)
        self.reject(580,-1,{'n':3,'maximum_start':10})

    def test_585_clique_subdivision(self):
        p={'n':6,'edges':[[0,1],[1,2],[2,3],[3,4],[4,5],[0,5]]}
        x={'branches':[0,2,4],'paths':[[0,1,2],[0,5,4],[2,3,4]]}
        self.assertEqual(score(585,x,p),3)
        bad=copy.deepcopy(x);bad['paths'][2]=[2,1,0,5,4];self.reject(585,bad,p)
        bad=copy.deepcopy(x);bad['paths'][2]=[2,4];self.reject(585,bad,p)

    def test_592_mols_and_pair_overlay(self):
        squares=[[[((a*i)+j)%3 for j in range(3)] for i in range(3)] for a in (1,2)]
        self.assertEqual(score(592,squares,{'n':3}),2)
        self.reject(592,[squares[0],squares[0]],{'n':3})
        bad=copy.deepcopy(squares);bad[0][1]=bad[0][0][:];self.reject(592,bad,{'n':3})

    def test_610_diameter_criticality_independent_floyd_oracle(self):
        def distance(n,es):
            d=[[0 if i==j else 999 for j in range(n)] for i in range(n)]
            for a,b in es:d[a][b]=d[b][a]=1
            for k in range(n):
                for i in range(n):
                    for j in range(n):d[i][j]=min(d[i][j],d[i][k]+d[k][j])
            return max(map(max,d))
        all_edges=list(combinations(range(5),2));best=0
        for mask in range(1<<len(all_edges)):
            chosen=[e for i,e in enumerate(all_edges) if mask>>i&1]
            good=distance(5,chosen)==2 and all(distance(5,chosen[:i]+chosen[i+1:])>2 for i in range(len(chosen)))
            if good:best=max(best,score(610,chosen,{'n':5}))
            else:self.reject(610,chosen,{'n':5})
        self.assertEqual(best,6)
        self.assertEqual(score(610,[[i,j] for i in range(8) for j in range(8,16)],{'n':16}),64)

    def test_614_hamiltonian_prefix_independent_permutation_oracle(self):
        order=list(combinations(range(4),2));p={'n':4,'edge_order':order}
        def earliest(cycle):
            for t in range(1,len(order)+1):
                es=set(order[:t])
                if all(tuple(sorted(e)) in es for e in zip(cycle,cycle[1:]+cycle[:1])):return t
        answers=[]
        for cycle in permutations(range(4)):
            cycle=list(cycle);answers.append(score(614,cycle,p));self.assertEqual(answers[-1],earliest(cycle))
        self.assertEqual(min(answers),5);self.reject(614,[0,1,2,2],p)

    def test_615_perfect_matching_arrival(self):
        order=list(combinations(range(6),3));p={'n':6,'edge_order':order};answers=[]
        for first in combinations(range(6),3):
            second=tuple(v for v in range(6) if v not in first)
            answers.append(score(615,[list(first),list(second)],p))
            self.assertEqual(answers[-1],max(order.index(first)+1,order.index(second)+1))
        self.assertEqual(min(answers),11)
        self.reject(615,[[0,1,2],[2,3,4]],p)

    def test_619_cycle_lengths_degree_and_girth(self):
        k33=[[u,w] for u in range(3) for w in range(3,6)]
        value,details=v.evaluate(619,k33,{'n':6,'minimum_degree':3,'girth':4})
        self.assertEqual(value,2);self.assertEqual(details['cycle_lengths'],[4,6])
        self.reject(619,list(combinations(range(4),2)),{'n':4,'minimum_degree':3,'girth':4})
        self.reject(619,[[0,1],[1,2]],{'n':4,'minimum_degree':2,'girth':4})
        # Independent permutation enumeration, including cycles with chords.
        graph=[[0,1],[0,2],[1,2],[1,3],[2,3],[3,4],[0,4]];es={tuple(e) for e in graph}
        lengths={k for k in range(3,6) if any(all(tuple(sorted(e)) in es for e in zip(c,c[1:]+c[:1])) for c in permutations(range(5),k))}
        self.assertEqual(v.cycle_lengths(v.adjacency(es,5)),lengths)

    def test_637_comparable_pairs_in_boolean_lattice(self):
        self.assertEqual(score(637,powerset(range(3)),{'n':3,'m':8}),3**3-2**3)
        self.assertEqual(score(637,[[],[0],[0,1],[0,1,2]],{'n':3,'m':4}),6)
        self.reject(637,[[],[],[0],[1]],{'n':3,'m':4})

    def test_652_three_distinct_primitive_oracle(self):
        for values in powerset(range(2,10))[1:]:
            good=all((b*c)%a for a,b,c in permutations(values,3))
            if good:self.assertEqual(score(652,values,{'n':10}),len(values))
            else:self.reject(652,values,{'n':10})
        self.assertEqual(score(652,[2,4],{'n':10}),2)

    def test_653_subset_products_oracle(self):
        for values in powerset(range(2,9))[1:]:
            products=[prod(s) for s in powerset(values)]
            if len(set(products))==len(products):self.assertEqual(score(653,values,{'n':10,'maximum_size':8}),len(values))
            else:self.reject(653,values,{'n':10,'maximum_size':8})

    def test_655_bicolored_cycle_detection(self):
        p={'n':4,'edges':[[0,1],[1,2],[2,3],[0,3]]}
        self.assertEqual(score(655,[0,1,0,2],p),3)
        self.reject(655,[0,1,0,1],p);self.reject(655,[0,0,1,2],p)
        self.assertEqual(score(655,[0,1,0,1],{'n':4,'edges':p['edges'][:3]}),2)

    def test_664_exact_edge_biclique_partition(self):
        p={'n':4,'edges':[[0,1],[1,2],[2,3],[0,3]]};biclique=[[0,2],[1,3]]
        self.assertEqual(score(664,[biclique],p),1)
        self.reject(664,[biclique,biclique],p)
        self.reject(664,[[[0],[1,3]]],p)
        self.reject(664,[[[0,1],[2,3]]],p)

    def test_665_restricted_sum_product(self):
        x={'values':[1,2,3,4],'edges':[[0,1],[0,2],[1,3]]};p={'n':4,'m':3,'maximum':10}
        self.assertEqual(score(665,x,p),3)
        self.assertEqual(v.graph_sum_product(x,p)[1],{'distinct_sums':3,'distinct_products':3})
        bad=copy.deepcopy(x);bad['values'][1]=1;self.reject(665,bad,p)
        bad=copy.deepcopy(x);bad['edges'].append([0,3]);self.reject(665,bad,p)

    def test_673_equal_degree_paths_and_noninduced_paths(self):
        self.assertEqual(score(673,[[i,j] for i in range(2) for j in range(2,5)],{'n':5}),6)
        self.reject(673,[[0,1],[1,2],[2,3]],{'n':4})
        self.reject(673,list(combinations(range(4),2)),{'n':4})

    def test_679_sigma_and_rational_error(self):
        for n in range(1,201):self.assertEqual(v.sigma(n),sum(d for d in range(1,n+1) if n%d==0))
        p={'maximum':1000,'target':[3,2]}
        value,details=v.evaluate(679,[11,6],p)
        self.assertEqual((details['error_numerator'],details['error_denominator']),(1,3))
        self.assertEqual(value,1/3);self.assertEqual(score(679,[1,1],p),.5)
        self.reject(679,[2,3],p)

    def test_688_published_fano_plane_and_lower_color_controls(self):
        es=[list(e) for e in combinations(range(7),3) if (e[0]+1)^(e[1]+1)^(e[2]+1)==0]
        self.assertEqual(len(es),7)
        # Independent explicit coloring enumeration supplies a valid upper certificate.
        colors=next(c for c in product(range(3),repeat=7) if all(len({c[i] for i in e})>1 for e in es))
        p={'n':9,'r':3,'colors':3};x={'edges':es,'colors':list(colors)+[0,0]}
        self.assertEqual(score(688,x,p),7)
        bad=copy.deepcopy(x);bad['edges']=es[:-1];self.reject(688,bad,p)
        bad=copy.deepcopy(x);bad['colors']=[0]*9;self.reject(688,bad,p)

    def test_697_published_small_values_and_bruteforce_optima(self):
        for n,multipliers,value in [(2,[3,6],4),(3,[6,8],5),(5,[8,10],5)]:
            p={'n':n,'maximum_window':10,'maximum_size':10}
            self.assertEqual(score(697,multipliers,p),value)
            for window in range(1,value):
                self.assertFalse(any(isqrt(n*prod(row))**2==n*prod(row) for row in powerset(range(n+1,n+window+1))[1:]))
            self.reject(697,[n+1],p);self.reject(697,[2*n,2*n],p)

    def test_717_published_interval_construction_and_small_oracle(self):
        self.assertEqual(score(717,list(range(16,33))+list(range(64,129)),{'n':128}),82)
        for values in powerset(range(1,9))[1:]:
            aset=set(values)
            good=not any({a+b,a+c,b+c}<=aset for a,b,c in combinations(values,3))
            if good:self.assertEqual(score(717,values,{'n':8}),len(values))
            else:self.reject(717,values,{'n':8})
        self.reject(717,[1,2,3,4,5],{'n':8})

    def test_725_cardinality_collision_oracle(self):
        for values in powerset(range(1,9))[1:]:
            groups={}
            for subset in powerset(values):groups.setdefault(sum(subset),set()).add(len(subset))
            good=all(len(c)==1 for c in groups.values())
            if good:self.assertEqual(score(725,values,{'n':8,'maximum_size':8}),len(values))
            else:self.reject(725,values,{'n':8,'maximum_size':8})
        self.assertEqual(score(725,[3,4,5,6],{'n':8,'maximum_size':8}),4)

    def test_730_pairwise_coprime_sum(self):
        best=0
        for values in powerset(range(1,11))[1:]:
            # Pairwise prime-support disjointness is independent of gcd implementation.
            supports=[{d for d in range(2,n+1) if n%d==0 and all(d%k for k in range(2,d))} for n in values]
            good=all(not a&b for a,b in combinations(supports,2))
            if good:best=max(best,score(730,values,{'n':10}));self.assertEqual(score(730,values,{'n':10}),sum(values))
            else:self.reject(730,values,{'n':10})
        self.assertEqual(best,30)  # {1,5,7,8,9}
        self.reject(730,[1,1],{'n':10})

    def test_733_published_construction_and_subset_divisibility_oracle(self):
        for z in range(3,9):
            values=[2**z-2**i for i in range(z)]
            self.assertEqual(score(733,values,{'n':256,'maximum_size':30}),z)
        for values in powerset(range(1,8))[1:]:
            sums={sum(row) for row in powerset(values)[1:]}
            good=all(a%b and b%a for a,b in combinations(sums,2))
            if good:self.assertEqual(score(733,values,{'n':8,'maximum_size':8}),len(values))
            else:self.reject(733,values,{'n':8,'maximum_size':8})

    def test_748_simple_path_not_cycle(self):
        p={'n':4,'edges':[[0,1],[1,2],[2,3]]}
        self.assertEqual(score(748,[0,1,2,3],p),4)
        self.reject(748,[0,2],p);self.reject(748,[0,1,0],p)


if __name__=='__main__':unittest.main()
