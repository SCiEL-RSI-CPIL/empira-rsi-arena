"""Prepared Aider Polyglot exercises: what the coding agent sees and what the hidden test run receives.

``scripts/prepare_polyglot.py`` extracts the Aider polyglot-benchmark (225
Exercism practice exercises in C++, Go, Java, JavaScript, Python and Rust,
pinned commit) into ``data_dir()/exercises/<language>/<slug>/`` and writes
``manifest.json`` plus DGM's exercise subsets (``subsets/small.json``,
``subsets/medium.json``). Nothing here touches the network.

Visibility follows the source harnesses. The agent receives the exercise
instructions (``.docs/introduction.md``, ``instructions.md`` and
``instructions.append.md`` in Aider's order) and the stub files that the
exercise's ``.meta/config.json`` lists as its solution; as read-only context it
also receives the small build files that DGM's harness leaves in the agent's
repository (Java ``build.gradle``, C++ ``CMakeLists.txt``, JavaScript
``package.json``, ``.eslintrc`` and ``.npmrc``). Test files, test data, helper
("editor") files, ``.meta`` (reference solutions) and ``.docs`` are never
shown. The hidden test run receives the whole exercise directory except
``.meta``, ``.docs``, ``.approaches`` and ``.articles``, with the solution files
replaced by the agent's versions.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path

from arena_task_paths import data_root
from .runtime import CandidateError

LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")
BENCHMARK_REPOSITORY = "https://github.com/Aider-AI/polyglot-benchmark"
BENCHMARK_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
DGM_REPOSITORY = "https://github.com/jennyzzt/dgm"
DGM_COMMIT = "a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2"
EXCLUDED_DIRS = (".approaches", ".articles", ".git")  # never prepared
HIDDEN_DIRS = (".meta", ".docs")  # prepared, but never shown to the agent or copied into a test run
CONTEXT_FILES = {"java": ("build.gradle",), "cpp": ("CMakeLists.txt",), "javascript": ("package.json", ".eslintrc", ".npmrc")}
EXECUTABLE = {"gradlew"}
DISABLED_LINE = re.compile(r"@Disabled\([^)]*\)\s*\n")


def data_dir():
    return data_root() / "polyglot"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root, files):
    """SHA-256 over sorted ``<relative path> <file sha256>`` lines."""
    root = Path(root)
    lines = [f"{rel} {file_sha256(root / rel)}" for rel in sorted(files)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def list_files(root):
    """Relative POSIX paths of the regular files below ``root`` (sorted)."""
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


@functools.lru_cache(maxsize=1)
def manifest():
    path = data_dir() / "manifest.json"
    if not path.is_file():
        raise CandidateError(f"prepared data missing: {path}; run `python scripts/prepare_polyglot.py`")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("benchmark", {}).get("commit") != BENCHMARK_COMMIT:
        raise CandidateError("prepared polyglot data come from a different benchmark commit; rerun scripts/prepare_polyglot.py")
    return data


def subset_ids(name):
    """Exercise ids of a named subset: DGM's ``small`` (10) or ``medium`` (50, one shared with small), ``heldout``
    (the 166 exercises in neither DGM list) or ``all`` (225)."""
    data = manifest()
    if name == "all":
        return sorted(data["exercises"])
    if name == "heldout":
        used = {i for ids in data["subsets"].values() for i in ids}
        return sorted(i for i in data["exercises"] if i not in used)
    if name not in data["subsets"]:
        raise CandidateError(f"unknown exercise subset {name!r}; known: all, heldout, " + ", ".join(sorted(data["subsets"])))
    return list(data["subsets"][name])


class Exercise:
    """One prepared exercise (``<language>__<slug>``)."""

    def __init__(self, exercise_id, record=None, root=None):
        record = record or manifest()["exercises"].get(exercise_id)
        if record is None:
            raise CandidateError(f"unknown exercise {exercise_id!r}")
        self.id, self.language, self.slug = exercise_id, record["language"], record["slug"]
        self.record = record
        self.root = Path(root) if root else data_dir() / "exercises" / self.language / self.slug
        if not self.root.is_dir():
            raise CandidateError(f"prepared exercise missing: {self.root}; run `python scripts/prepare_polyglot.py`")

    @property
    def files(self):
        return self.record["files"]

    def verify(self):
        """The prepared directory must match the digest recorded in the manifest."""
        digest = tree_digest(self.root, list_files(self.root))
        if digest != self.record["sha256"]:
            raise RuntimeError(f"prepared exercise {self.id} does not match the manifest; rerun scripts/prepare_polyglot.py")
        return digest

    def read(self, rel):
        return (self.root / rel).read_text(encoding="utf-8")

    def instructions(self):
        parts = []
        for name in ("introduction.md", "instructions.md", "instructions.append.md"):
            path = self.root / ".docs" / name
            if path.is_file():
                parts.append(path.read_text(encoding="utf-8").strip("\n") + "\n")
        return "\n".join(parts)

    def solution_paths(self):
        return list(self.files.get("solution", []))

    def stubs(self):
        return {rel: self.read(rel) for rel in self.solution_paths() if (self.root / rel).is_file()}

    def context(self):
        return {rel: self.read(rel) for rel in CONTEXT_FILES.get(self.language, ()) if (self.root / rel).is_file()}

    def visible(self):
        """The agent's input (JSON-serialisable)."""
        return {"id": self.id, "language": self.language, "instructions": self.instructions(), "files": self.stubs(),
                "context": self.context()}

    def bundle(self, solution=None, remove=()):
        """``{relative path: (bytes, mode)}`` for the hidden test run: the exercise without hidden dirs, solution overridden."""
        solution = dict(solution or {})
        out = {}
        for rel in list_files(self.root):
            if rel.split("/", 1)[0] in HIDDEN_DIRS or rel in remove:
                continue
            data = (self.root / rel).read_bytes()
            if rel in solution:
                data = solution[rel].encode("utf-8")
            elif self.language == "java" and rel.startswith("src/test/") and rel.endswith(".java"):
                # Aider's harness removes @Disabled so that every test runs (DGM's harness does not).
                data = DISABLED_LINE.sub("", data.decode("utf-8")).encode("utf-8")
            out[rel] = (data, 0o755 if Path(rel).name in EXECUTABLE else 0o644)
        for rel, text in solution.items():
            if rel not in out:
                out[rel] = (text.encode("utf-8"), 0o644)
        return out

    def reference_solution(self):
        """Exercism's reference ("example") files mapped onto solution paths, and stub files to drop (trusted checks only)."""
        solution_paths = self.solution_paths()
        mapped, remove = {}, []
        for rel in self.files.get("example", []):
            text = self.read(rel)
            name = Path(rel).name
            if self.language == "java":
                mapped["src/main/java/" + name] = text
            elif self.language == "rust":
                mapped["src/lib.rs" if name.endswith(".rs") else rel] = text
            elif self.language == "cpp":
                suffix = Path(name).suffix
                target = next((p for p in solution_paths if p.endswith(suffix)), None)
                if target:
                    mapped[target] = text
            else:
                target = next((p for p in solution_paths if Path(p).suffix == Path(name).suffix), None)
                if target:
                    mapped[target] = text
        if self.language == "rust" and (self.root / ".meta" / "Cargo-example.toml").is_file():
            mapped["Cargo.toml"] = self.read(".meta/Cargo-example.toml")
        if self.language == "cpp":
            remove = [p for p in solution_paths if p.endswith(".cpp") and p not in mapped]
        return mapped, remove
