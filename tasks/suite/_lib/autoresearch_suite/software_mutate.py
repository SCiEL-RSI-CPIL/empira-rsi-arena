"""First-order AST mutants of a Python module (operators in the style of mutmut and MutPy).

Operators: arithmetic (AOR), relational (ROR), logical connector (LCR) and
augmented-assignment (ASR) replacement, deletion of ``not`` and unary minus
(UOD), constant replacement (CRP: n -> n + 1, x -> x + 1.0, s -> 'XX' + s +
'XX', True <-> False) and break <-> continue (BCR). Only code inside function
and method bodies is mutated (docstrings excluded). A mutant is stored as a
replacement of the source lines of the top-level function or method that
contains it, ``(first line, last line, new text)``, produced by ``ast.unparse``.
Mutants that do not compile, duplicates and mutants whose compiled function
is identical to the compiled original (trivial compiler equivalence,
Papadakis et al., ICSE 2015) are dropped.
"""
from __future__ import annotations

import ast
import copy
import types

AOR = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Div,
       ast.Mod: ast.Div, ast.Pow: ast.Mult, ast.LShift: ast.RShift, ast.RShift: ast.LShift,
       ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd, ast.BitXor: ast.BitAnd}
ROR = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
       ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}


def _docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            ids.add(id(body[0].value))
    return ids


def _sites(function, docstrings):
    """(node, operator name, apply(node)) for every mutation site inside the function body."""
    sites = []
    for stmt in function.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.BinOp) and type(node.op) in AOR:
                sites.append((node, "AOR", lambda n: setattr(n, "op", AOR[type(n.op)]())))
            elif isinstance(node, ast.AugAssign) and type(node.op) in AOR:
                sites.append((node, "ASR", lambda n: setattr(n, "op", AOR[type(n.op)]())))
            elif isinstance(node, ast.Compare):
                for i, op in enumerate(node.ops):
                    if type(op) in ROR:
                        sites.append((node, "ROR", lambda n, i=i: n.ops.__setitem__(i, ROR[type(n.ops[i])]())))
            elif isinstance(node, ast.BoolOp):
                sites.append((node, "LCR", lambda n: setattr(n, "op", ast.Or() if isinstance(n.op, ast.And) else ast.And())))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
                sites.append((node, "UOD", lambda n: setattr(n, "op", ast.UAdd() if isinstance(n.op, ast.USub) else _NOT_DELETED)))
            elif isinstance(node, ast.Constant) and id(node) not in docstrings:
                value = node.value
                if type(value) is bool:
                    sites.append((node, "CRP", lambda n: setattr(n, "value", not n.value)))
                elif type(value) is int:
                    sites.append((node, "CRP", lambda n: setattr(n, "value", n.value + 1)))
                elif type(value) is float:
                    sites.append((node, "CRP", lambda n: setattr(n, "value", n.value + 1.0)))
                elif type(value) is str:
                    sites.append((node, "CRP", lambda n: setattr(n, "value", "XX" + n.value + "XX")))
            elif isinstance(node, (ast.Break, ast.Continue)):
                sites.append((node, "BCR", None))
    return sites


class _NotDeleted(ast.unaryop):
    """Marker: the UnaryOp is replaced by its operand when the mutant is unparsed."""


_NOT_DELETED = _NotDeleted()


class _Finish(ast.NodeTransformer):
    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        return node.operand if isinstance(node.op, _NotDeleted) else node

    def visit_Break(self, node):
        return ast.copy_location(ast.Continue(), node) if getattr(node, "_artask_swap", False) else node

    def visit_Continue(self, node):
        return ast.copy_location(ast.Break(), node) if getattr(node, "_artask_swap", False) else node


def _code_signature(code):
    """Compiled code without line information, for trivial-compiler-equivalence checks."""
    consts = tuple(_code_signature(c) if isinstance(c, types.CodeType) else (type(c).__name__, repr(c)) for c in code.co_consts)
    return (code.co_code, consts, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars)


def _compiled(function_source):
    module = compile(function_source, "<unit>", "exec", optimize=2)
    return tuple(_code_signature(c) for c in module.co_consts if isinstance(c, types.CodeType))


def _units(tree):
    """Top-level functions and methods of top-level classes."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{node.name}.{item.name}", item


def mutants(source):
    """List of {'unit', 'operator', 'line', 'start', 'end', 'replacement'} (deterministic order)."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out, seen = [], set()
    for qualname, unit in _units(tree):
        start = min([unit.lineno] + [d.lineno for d in unit.decorator_list])
        indent = " " * unit.col_offset
        original = _compiled(ast.unparse(unit))
        docstrings = _docstring_ids(unit)
        for index in range(len(_sites(unit, docstrings))):
            mutated = copy.deepcopy(unit)
            node, operator, apply = _sites(mutated, _docstring_ids(mutated))[index]
            if apply is None:
                node._artask_swap = True
            else:
                apply(node)
            text = ast.unparse(ast.fix_missing_locations(_Finish().visit(mutated)))
            key = (qualname, text)
            if key in seen:
                continue
            seen.add(key)
            try:
                if _compiled(text) == original:
                    continue  # trivially equivalent
                replacement = "".join(indent + line if line.strip() else line for line in text.splitlines(keepends=True)) + "\n"
                compile("".join(lines[:start - 1]) + replacement + "".join(lines[unit.end_lineno:]), "<mutant>", "exec")
            except (SyntaxError, ValueError):
                continue
            out.append({"unit": qualname, "operator": operator, "line": node.lineno, "start": start,
                        "end": unit.end_lineno, "replacement": replacement})
    return out
