"""Deterministic toy environment for the agent_bench tests (no model, no external packages).

The agent must say the hidden target number of each episode; ``higher`` and
``lower`` hints follow wrong guesses. Episode score: 1 when the target is said
within the step cap, 0.5 when it was never said but the last guess was within
one of it, else 0. The target lives only in this server process.
"""
from __future__ import annotations


class Adapter:
    def __init__(self, params):
        self.targets = list(params.get("targets", [3, 7]))
        self.target, self.last, self.solved = None, None, False

    def episodes(self):
        return [{"id": f"toy-{i}", "index": i} for i in range(len(self.targets))]

    def reset(self, spec):
        self.target, self.last, self.solved = self.targets[spec["index"]], None, False
        return {"instruction": "Guess the hidden integer between 0 and 9. Reply with the number only.", "observation": "Start guessing.",
                "valid_actions": [str(i) for i in range(10)], "done": False}

    def step(self, action):
        try:
            self.last = int(action.strip())
        except ValueError:
            return {"observation": "Not a number.", "reward": 0.0, "done": False, "valid_actions": [str(i) for i in range(10)]}
        self.solved = self.last == self.target
        hint = "correct" if self.solved else ("higher" if self.last < self.target else "lower")
        return {"observation": hint, "reward": float(self.solved), "done": self.solved, "valid_actions": [str(i) for i in range(10)]}

    def score(self):
        near = self.last is not None and abs(self.last - self.target) <= 1
        return {"score": 1.0 if self.solved else (0.5 if near else 0.0), "details": {"success": self.solved}}
