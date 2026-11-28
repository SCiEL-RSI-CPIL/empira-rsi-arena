"""M3ToolEval adapter (CodeAct's multi-turn tool-use benchmark, text-as-action mode).

The tasks, tools, prompt, output post-processing, parser and answer check are
CodeAct's own code (xingyaoww/code-act, scripts/eval/m3tooleval, MIT), fetched
at a pinned commit by ``scripts/prepare_agent_bench.py`` into
``data/agent_bench/m3tooleval`` of the task cache and imported here with one
replacement: CodeAct's ``calculator`` tool is ``eval(expression)``, which would
run the agent's text as Python inside this (trusted) server process, where the
expected outputs and the answer check live. ``calculator`` below evaluates the
same arithmetic without ``eval``; ``harden`` installs it and refuses any other
tool whose code evaluates text. The episode loop follows CodeAct's
``run_task``: each agent turn is post-processed and parsed; a tool call
('Action: name, arg, ... End Action') is executed and its result returned; a
wrong 'Answer:' is answered with CodeAct's "incorrect, try again" message; the
episode ends on a correct answer (or, for single-tool tasks, a tool result
equal to the expected output) or after 10 turns. Episode score: 1 when solved,
else 0.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import operator
import sys
import types

from arena_task_paths import data_root

N_TURNS = 10
INCORRECT = "Your answer is incorrect. Please try again. Note that you should ONLY output the answer (e.g., single number), without any other text."
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
          ast.Mod: operator.mod, ast.Pow: operator.pow}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCTIONS = {"abs": abs, "round": round, "int": int, "float": float}
MAX_INT_BITS = 4096  # larger integers (about 1,233 digits) are refused: printing has a 4,300-digit limit and powers cost unbounded time


def data_dir():
    return data_root() / "agent_bench"


def _arithmetic(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        value = UNARY[type(node.op)](_arithmetic(node.operand))
    elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = _arithmetic(node.left), _arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int) and abs(left) > 1 and left.bit_length() * right > 2 * MAX_INT_BITS:
            raise OverflowError("integer power too large")  # refused before it is computed; the exact limit is applied below
        value = BINARY[type(node.op)](left, right)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        value = FUNCTIONS[node.func.id](*[_arithmetic(argument) for argument in node.args])
    else:
        raise ValueError("not an arithmetic expression")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise OverflowError("integer too large")
    return value


def calculator(expression):
    """Stand-in for CodeAct's calculator tool (``eval(expression)``): numbers, + - * / // % **, parentheses and abs, round,
    int, float. Everything else, like every expression ``eval`` fails on, gets CodeAct's failure message."""
    try:
        if not isinstance(expression, str):
            raise TypeError("the expression must be a string")
        return _arithmetic(ast.parse(expression.strip(), mode="eval").body)
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError, MemoryError):
        return f"Failed to evaluate expression: {expression}"


def evaluates_text(function):
    """Whether a tool's own code (nested functions included) loads eval, exec, compile or __import__ as a global name."""
    import dis
    pending = [getattr(getattr(function, "__func__", function), "__code__", None)]
    while pending:
        code = pending.pop()
        if code is None:
            continue
        if any(i.opname in ("LOAD_GLOBAL", "LOAD_NAME") and i.argval in ("eval", "exec", "compile", "__import__") for i in dis.get_instructions(code)):
            return True
        pending += [constant for constant in code.co_consts if hasattr(constant, "co_code")]
    return False


def harden(task):
    """Give a task the arithmetic calculator; a tool that evaluates text under any other name stops the adapter."""
    tools = {}
    for name, tool in task.tools.items():
        if evaluates_text(tool.function):
            if name != "calculator":
                raise RuntimeError(f"tool {name!r} of task {task.name} evaluates text as Python; it needs a safe replacement")
            tool = tool._replace(function=calculator)
        tools[name] = tool
    task.tools = tools
    task.reset()


def load_tasks():
    """Import CodeAct's task package (the IPython-based REPL of the code-as-action mode is not needed)."""
    root = data_dir() / "m3tooleval"
    if not (root / "tasks" / "base.py").is_file():
        raise FileNotFoundError(f"{root} missing; run python scripts/prepare_agent_bench.py m3tooleval")
    stub = types.ModuleType("tasks.repl")
    stub.PythonREPL = None
    sys.modules["tasks.repl"] = stub
    try:  # CodeAct imports tqdm for a progress bar that is not used here; the package does not require it
        import tqdm  # noqa: F401
    except ImportError:
        progress = types.ModuleType("tqdm")
        progress.tqdm = lambda iterable=None, *args, **kwargs: iterable
        sys.modules["tqdm"] = progress
    sys.path.insert(0, str(root))
    with contextlib.redirect_stdout(io.StringIO()):  # CodeAct prints every task with its expected output when it is built
        import tasks  # noqa: F401 - registers every task iterator
        from tasks import base
    out = {}
    for iterator, _ in base.task_iterators:
        for task in iterator:
            harden(task)
            out[task.name] = task
    return base, out


def postprocess(generation):
    """CodeAct's postprocess_fn: keep the first Answer/Action block of a generation."""
    import re
    generation = generation.lstrip()
    matches = list(re.finditer(r"(Answer:|Action:)", generation))
    if len(matches) <= 1:
        return generation
    trimmed = generation[:matches[1].start()].strip()
    first_action_end = trimmed.find("Action:") + len("Action:")
    end_action = trimmed.find("End Action", first_action_end)
    if end_action != -1:
        trimmed = trimmed[:end_action + len("End Action")].strip()
    next_thought = trimmed.find("Thought:", first_action_end)
    if next_thought != -1:
        trimmed = trimmed[:next_thought].strip()
    return trimmed


def ordered(names, seed):
    return sorted(names, key=lambda name: hashlib.sha256(f"{seed}|{name}".encode()).hexdigest())


class Adapter:
    def __init__(self, params):
        self.params = params
        self.base, self.tasks = load_tasks()
        self.task = None

    def episodes(self):
        names = sorted(self.tasks)
        domains = sorted({name.split("/")[0] for name in names})
        per_domain = int(self.params.get("per_domain", 4))
        chosen = []
        for domain in domains:
            chosen += ordered([n for n in names if n.split("/")[0] == domain], self.params.get("seed", 0))[:per_domain]
        return [{"id": name} for name in chosen]

    def reset(self, spec):
        self.task = self.tasks[spec["id"]]
        self.task.reset()
        self.turns, self.correct = 0, False
        prompt = self.task.get_prompt(self.base.ActionMode.TEXT_AS_ACTION)
        return {"instruction": prompt, "observation": "", "valid_actions": list(self.task.tools), "done": False}

    def step(self, generation):
        with contextlib.redirect_stdout(io.StringIO()):  # tool calls print their arguments and results
            return self._step(generation)

    def _step(self, generation):
        task = self.task
        generation = postprocess(generation)
        self.turns += 1
        parsed = task.parse_generation(generation)
        extra = parsed.get("extra_info") or ""
        if parsed["type"] == "action":
            result = task.execute_action(parsed["content"], self.base.ActionMode.TEXT_AS_ACTION)
            observation = str(result) + ("\n*Extra reminder: " + extra if extra else "")
            if task.is_single_tool_task and task.check_answer(result):
                self.correct = True
        elif parsed["type"] == "answer":
            self.correct = bool(task.check_answer(parsed["content"]))
            observation = "Correct." if self.correct else INCORRECT + ("\n*Extra reminder: " + extra if extra else "")
        else:
            observation = parsed["content"]
        done = self.correct or self.turns >= N_TURNS
        return {"observation": observation, "reward": float(self.correct), "done": done, "valid_actions": list(task.tools)}

    def score(self):
        return {"score": float(self.correct), "details": {"success": self.correct, "turns": self.turns, "domain": self.task.name.split("/")[0]}}
