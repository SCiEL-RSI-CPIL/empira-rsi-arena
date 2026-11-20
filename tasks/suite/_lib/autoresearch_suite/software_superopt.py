"""Superoptimisation of the 25 Hacker's Delight kernels (Gulwani et al., PLDI 2011; STOKE's p01-p25).

The reference programs are STOKE's C files (examples/hacker/pNN/fxn.cc,
Apache-2.0), parsed into a straight-line 32-bit IR: one operation per
instruction, C types decide signed or unsigned shifts, divisions and
comparisons, and C's undefined cases (division by zero, shifts by 32 or more)
take the SMT-LIB bit-vector semantics. Four STOKE transcriptions differ from
the intended functions and are corrected (CORRECTIONS): p12 swaps x and y
(STOKE's file tests nlz(y) <= nlz(x)), p13 needs an unsigned shift of -x
(otherwise every nonzero x maps to -1), p20 must use x ^ (x + s)
with unsigned shift and division (Gosper's next-permutation step), and p24
must return o11 + 1 (o10 + 1 does not round up to a power of two).

A candidate program counts when Z3 proves it equal to the reference for all
inputs within a fixed resource limit; otherwise the reference length counts.
The score is the total number of instructions over the 25 kernels.
"""
from __future__ import annotations

import json
import random
import re

from . import software_data, software_sandbox
from .runtime import CandidateError

MASK = (1 << 32) - 1
NAMES = [f"p{i:02d}" for i in range(1, 26)]
SOURCES = tuple(f"stoke-{name}" for name in NAMES)
ARITY = {"add": 2, "sub": 2, "mul": 2, "and": 2, "or": 2, "xor": 2, "shl": 2, "lshr": 2, "ashr": 2,
         "udiv": 2, "sdiv": 2, "urem": 2, "srem": 2, "eq": 2, "ne": 2, "ult": 2, "ule": 2, "ugt": 2, "uge": 2,
         "slt": 2, "sle": 2, "sgt": 2, "sge": 2, "not": 1, "neg": 1, "select": 3}
DESCRIPTIONS = {
    "p01": "clear the lowest set bit", "p02": "test whether x has the form 2^n - 1 (zero exactly then)",
    "p03": "isolate the lowest set bit", "p04": "mask of the lowest set bit and the zeros below it",
    "p05": "propagate the lowest set bit to the right", "p06": "set the lowest clear bit",
    "p07": "isolate the lowest clear bit", "p08": "mask of the trailing zeros", "p09": "absolute value",
    "p10": "test whether x and y have the same number of leading zeros", "p11": "test whether nlz(x) < nlz(y)",
    "p12": "test whether nlz(x) <= nlz(y)", "p13": "sign function (-1, 0 or 1)", "p14": "floor of the average without overflow",
    "p15": "ceiling of the average without overflow", "p16": "unsigned maximum", "p17": "clear the lowest run of set bits",
    "p18": "test whether x is a power of two", "p19": "exchange two bit fields of x selected by mask m and shift k",
    "p20": "next larger number with the same number of set bits", "p21": "cycle through three values a, b, c",
    "p22": "parity", "p23": "population count", "p24": "round up to the next power of two",
    "p25": "high 32 bits of the signed 64-bit product",
}
# STOKE transcription errors corrected to the Hacker's Delight functions of Gulwani et al. (see module docstring).
CORRECTIONS = {
    ("p12", "int32_t o1 = ~y"): "int32_t o1 = ~x",
    ("p12", "int32_t o2 = x & o1"): "int32_t o2 = y & o1",
    ("p12", "return (uint32_t) o2 <= (uint32_t) y"): "return (uint32_t) o2 <= (uint32_t) x",
    ("p13", "int32_t o3 = o2 >> 31"): "int32_t o3 = (uint32_t) o2 >> 31",
    ("p20", "int32_t o4 = x ^ o2"): "int32_t o4 = x ^ o3",
    ("p20", "int32_t o5 = o4 >> 2"): "uint32_t o5 = (uint32_t) o4 >> 2",
    ("p20", "int32_t o6 = o5 / o2"): "uint32_t o6 = o5 / (uint32_t) o2",
    ("p24", "return o10 + 1"): "return o11 + 1",
}
MAX_INSTRUCTIONS = 64
RLIMIT = 20_000_000


def _signed(v):
    return v - (1 << 32) if v & 0x80000000 else v


def _udiv(a, b):
    return MASK if b == 0 else a // b


def _sdiv(a, b):
    sa, sb = _signed(a), _signed(b)
    q = _udiv(abs(sa) & MASK, abs(sb) & MASK)
    return (-q if (sa < 0) != (sb < 0) else q) & MASK


def _srem(a, b):
    sa = _signed(a)
    r = (abs(sa) & MASK) if b == 0 else (abs(sa) % (abs(_signed(b)) & MASK or 1))
    return (-r if sa < 0 else r) & MASK


OPS = {
    "add": lambda a, b: (a + b) & MASK, "sub": lambda a, b: (a - b) & MASK, "mul": lambda a, b: (a * b) & MASK,
    "and": lambda a, b: a & b, "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b,
    "shl": lambda a, b: (a << b) & MASK if b < 32 else 0, "lshr": lambda a, b: a >> b if b < 32 else 0,
    "ashr": lambda a, b: (_signed(a) >> min(b, 31)) & MASK, "udiv": _udiv, "sdiv": _sdiv,
    "urem": lambda a, b: a if b == 0 else a % b, "srem": _srem,
    "eq": lambda a, b: int(a == b), "ne": lambda a, b: int(a != b), "ult": lambda a, b: int(a < b),
    "ule": lambda a, b: int(a <= b), "ugt": lambda a, b: int(a > b), "uge": lambda a, b: int(a >= b),
    "slt": lambda a, b: int(_signed(a) < _signed(b)), "sle": lambda a, b: int(_signed(a) <= _signed(b)),
    "sgt": lambda a, b: int(_signed(a) > _signed(b)), "sge": lambda a, b: int(_signed(a) >= _signed(b)),
    "not": lambda a: ~a & MASK, "neg": lambda a: -a & MASK, "select": lambda c, a, b: a if c else b,
}


def run(program, inputs, values):
    env = {name: v & MASK for name, v in zip(inputs, values)}
    for dest, op, *args in program:
        env[dest] = OPS[op](*[env[a] if isinstance(a, str) else a & MASK for a in args])
    return env[program[-1][0]]


def check_program(program, inputs):
    """Structural validation: SSA straight-line code over known operations."""
    if not isinstance(program, list) or not 1 <= len(program) <= MAX_INSTRUCTIONS:
        raise CandidateError(f"a program must be a list of 1 to {MAX_INSTRUCTIONS} instructions")
    defined = set(inputs)
    for i, instruction in enumerate(program):
        if not isinstance(instruction, list) or len(instruction) < 2:
            raise CandidateError(f"instruction {i} must be [dest, op, args...]")
        dest, op, *args = instruction
        if op not in ARITY or len(args) != ARITY[op]:
            raise CandidateError(f"instruction {i}: unknown operation or wrong arity ({op!r})")
        if not isinstance(dest, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,30}", dest) or dest in defined:
            raise CandidateError(f"instruction {i}: destination must be a new identifier")
        for a in args:
            if isinstance(a, str):
                if a not in defined:
                    raise CandidateError(f"instruction {i}: {a!r} is not defined before use")
            elif type(a) is not int or not -(1 << 31) <= a <= MASK:
                raise CandidateError(f"instruction {i}: constants must be ints in [-2^31, 2^32 - 1]")
        defined.add(dest)
    return program


# ---------------------------------------------------------------- C parsing (preparation time)

TOKEN = re.compile(r"\s*(0x[0-9a-fA-F]+|\d+|[A-Za-z_]\w*|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^~!<>?:()=;,{}])")


class _Parser:
    def __init__(self, text, types):
        self.tokens, self.pos, self.types, self.program, self.count = TOKEN.findall(text), 0, types, [], 0

    def peek(self, k=0):
        return self.tokens[self.pos + k] if self.pos + k < len(self.tokens) else None

    def take(self, expected=None):
        token = self.tokens[self.pos]
        if expected is not None and token != expected:
            raise ValueError(f"expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    def emit(self, op, *args, dest=None):
        if dest is None:
            self.count += 1
            dest = f"t{self.count}"
        self.program.append([dest, op, *args])
        return dest

    def expression(self):
        cond = self.binary(0)
        if self.peek() != "?":
            return cond
        self.take("?")
        a = self.expression()
        self.take(":")
        b = self.expression()
        return (self.emit("select", cond[0], a[0], b[0]), a[1])

    LEVELS = [["||"], ["&&"], ["|"], ["^"], ["&"], ["==", "!="], ["<", "<=", ">", ">="], ["<<", ">>"], ["+", "-"], ["*", "/", "%"]]

    def binary(self, level):
        if level == len(self.LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek() in self.LEVELS[level]:
            op = self.take()
            right = self.binary(level + 1)
            left = self.combine(op, left, right)
        return left

    def combine(self, op, left, right):
        (a, ta), (b, tb) = left, right
        common = "u" if "u" in (ta, tb) else "i"
        if op in ("&&", "||"):
            x, y = self.emit("ne", a, 0), self.emit("ne", b, 0)
            return self.emit("and" if op == "&&" else "or", x, y), "i"
        if op in ("<<", ">>"):
            return self.emit("shl" if op == "<<" else ("lshr" if ta == "u" else "ashr"), a, b), ta
        simple = {"&": "and", "|": "or", "^": "xor", "+": "add", "-": "sub", "*": "mul", "==": "eq", "!=": "ne"}
        if op in simple:
            return self.emit(simple[op], a, b), ("i" if op in ("==", "!=") else common)
        if op in ("/", "%"):
            return self.emit(("u" if common == "u" else "s") + ("div" if op == "/" else "rem"), a, b), common
        name = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[op]
        return self.emit(("u" if common == "u" else "s") + name, a, b), "i"

    def unary(self):
        token = self.peek()
        if token == "(" and self.peek(1) in ("uint32_t", "int32_t") and self.peek(2) == ")":
            kind = "u" if self.peek(1) == "uint32_t" else "i"
            self.pos += 3
            value, _ = self.unary()
            return value, kind
        if token in ("-", "~", "!"):
            self.take()
            value, kind = self.unary()
            if token == "!":
                return self.emit("eq", value, 0), "i"
            return self.emit("neg" if token == "-" else "not", value), kind
        if token == "(":
            self.take("(")
            value = self.expression()
            self.take(")")
            return value
        self.take()
        if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", token):
            return int(token, 0), "i"
        return token, self.types[token]


def parse_c(name, text):
    body = re.sub(r"//[^\n]*", "", text)
    head = re.search(r"int32_t\s+" + name + r"\s*\(([^)]*)\)\s*\{(.*)\}", body, re.S)
    inputs = [p.split()[-1] for p in head.group(1).split(",")]
    types = {p: "i" for p in inputs}
    parser = _Parser("", types)
    for statement in [s.strip() for s in head.group(2).split(";") if s.strip()]:
        statement = re.sub(r"\s+", " ", statement)
        statement = CORRECTIONS.get((name, statement), statement)
        declared = re.fullmatch(r"(u?int32_t)\s+(\w+)\s*=\s*(.*)", statement, re.S)
        source = declared.group(3) if declared else re.fullmatch(r"return\s+(.*)", statement, re.S).group(1)
        sub = _Parser(source, types)
        sub.program, sub.count = parser.program, parser.count
        value, _ = sub.expression()
        if sub.pos != len(sub.tokens):
            raise ValueError(f"{name}: unparsed tokens in {statement!r}")
        parser.count = sub.count
        if declared:
            types[declared.group(2)] = "u" if declared.group(1) == "uint32_t" else "i"
            if parser.program and parser.program[-1][0] == value:
                parser.program[-1][0] = declared.group(2)
            else:
                raise ValueError(f"{name}: declaration without an operation")
        elif not parser.program or parser.program[-1][0] != value:
            raise ValueError(f"{name}: return value is not the last instruction")
    return {"name": name, "inputs": inputs, "reference": parser.program, "description": DESCRIPTIONS.get(name, "")}


def derive(raw):
    texts = {name: re.sub(r"\s+", " ", re.sub(r"//[^\n]*", "", raw[f"stoke-{name}"].decode())) for name in NAMES}
    for name, statement in CORRECTIONS:
        if texts[name].count(statement + ";") != 1:
            raise RuntimeError(f"{name}: correction anchor {statement!r} not found exactly once")
    problems = [parse_c(name, raw[f"stoke-{name}"].decode()) for name in NAMES]
    rng = random.Random(0)
    for problem in problems:  # the IR must be well formed and deterministic on random inputs
        check_program(problem["reference"], problem["inputs"])
        run(problem["reference"], problem["inputs"], [rng.getrandbits(32) for _ in problem["inputs"]])
    return {"superopt/problems.json": json.dumps(problems, sort_keys=True, separators=(",", ":")).encode()}


def z3_term(program, inputs, variables):
    """Z3 bit-vector term of a program (SMT-LIB semantics, identical to ``run``)."""
    import z3
    env = dict(zip(inputs, variables))
    one, zero = z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)
    flag = lambda condition: z3.If(condition, one, zero)
    table = {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "and": lambda a, b: a & b,
        "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b, "shl": lambda a, b: a << b, "lshr": z3.LShR,
        "ashr": lambda a, b: a >> b, "udiv": z3.UDiv, "sdiv": lambda a, b: a / b, "urem": z3.URem, "srem": z3.SRem,
        "eq": lambda a, b: flag(a == b), "ne": lambda a, b: flag(a != b), "ult": lambda a, b: flag(z3.ULT(a, b)),
        "ule": lambda a, b: flag(z3.ULE(a, b)), "ugt": lambda a, b: flag(z3.UGT(a, b)), "uge": lambda a, b: flag(z3.UGE(a, b)),
        "slt": lambda a, b: flag(a < b), "sle": lambda a, b: flag(a <= b), "sgt": lambda a, b: flag(a > b),
        "sge": lambda a, b: flag(a >= b), "not": lambda a: ~a, "neg": lambda a: -a,
        "select": lambda c, a, b: z3.If(c != zero, a, b),
    }
    for dest, op, *args in program:
        env[dest] = table[op](*[env[a] if isinstance(a, str) else z3.BitVecVal(a & MASK, 32) for a in args])
    return env[program[-1][0]]


SPECIAL = [0, 1, 2, 3, 31, 32, 33, 0x7FFFFFFF, 0x80000000, 0x80000001, MASK, MASK - 1, 0x55555555, 0xAAAAAAAA]


def equivalent(reference, program, inputs, *, tests=300, seed=0):
    """'proven', 'counterexample' or 'unproven' (Z3 resource limit reached)."""
    import z3
    rng = random.Random(seed)
    for t in range(tests):
        values = [rng.choice(SPECIAL) if t % 3 == 0 else rng.getrandbits(32) for _ in inputs]
        if run(reference, inputs, values) != run(program, inputs, values):
            return "counterexample"
    variables = [z3.BitVec(name, 32) for name in inputs]
    solver = z3.Solver()
    solver.set("rlimit", RLIMIT)
    solver.add(z3_term(reference, inputs, variables) != z3_term(program, inputs, variables))
    result = solver.check()
    return "proven" if result == z3.unsat else ("counterexample" if result == z3.sat else "unproven")


DRIVER = '''

def _artask_superopt(problems):
    return [optimize(problem) for problem in problems]
'''


def public_problem(problem):
    return {"name": problem["name"], "description": problem["description"], "inputs": list(problem["inputs"]),
            "reference": [list(i) for i in problem["reference"]], "operations": dict(ARITY), "max_instructions": MAX_INSTRUCTIONS}


def superopt_score(module, *, cpu_seconds=120):
    from pathlib import Path
    path = software_sandbox.require_entry(module, "optimize")
    problems = software_data.load_json("superopt/problems.json")
    source = Path(path).read_text(encoding="utf-8") + DRIVER
    try:
        programs = software_sandbox.execute(source, "_artask_superopt", [[public_problem(p) for p in problems]],
                                            cpu_seconds=cpu_seconds, memory_mb=2048, filename="candidate.py")
    except software_sandbox.SandboxFailure as failure:
        raise CandidateError(f"candidate optimize() failed in the sandbox ({failure.kind}): {failure}"[:2000]) from None
    if not isinstance(programs, list) or len(programs) != len(problems):
        raise CandidateError("optimize() must return one program per kernel")
    total, reference_total, per_kernel = 0, 0, {}
    for problem, program in zip(problems, programs):
        check_program(program, problem["inputs"])
        status = equivalent(problem["reference"], program, problem["inputs"])
        length = len(program) if status == "proven" else len(problem["reference"])
        total += length
        reference_total += len(problem["reference"])
        per_kernel[problem["name"]] = {"status": status, "instructions": length, "reference": len(problem["reference"])}
    return float(total), {"kernels": len(problems), "reference_instructions": reference_total,
                          "improved_kernels": sorted(n for n, k in per_kernel.items() if k["instructions"] < k["reference"]),
                          "not_proven": sorted(n for n, k in per_kernel.items() if k["status"] != "proven"), "per_kernel": per_kernel}
