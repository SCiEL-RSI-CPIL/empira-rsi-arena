"""Toolchain image for the polyglot evaluator: recipe, local build and checks.

The image follows the benchmark image of Aider's harness (``benchmark/Dockerfile``
in Aider-AI/aider, Apache-2.0), which DGM's ``polyglot/dockerfiles.py`` reuses:
``buildpack-deps:jammy`` with CMake, Boost and TBB, OpenJDK 21, Python 3.11,
Go 1.21.5, Rust (rustup) and Node.js 20 with the Exercism JavaScript test
packages in ``/npm-install``. Differences: versions are pinned (base image digest,
Rust, Node.js, jest, pytest), conda and Aider itself are not installed, Boost is
reduced to the headers plus ``date_time`` (the only compiled Boost library an
exercise links), and the dependency caches of every exercise in the benchmark
(the Gradle 8.7 distribution and Maven artifacts, the crates named in any
``Cargo.toml``, Go's standard-library build cache) are filled during the build
so that test runs work with ``--network none``.

The image is built locally from this recipe by ``python scripts/prepare_polyglot.py
--build-image`` for the Docker host's architecture (arm64 on Apple silicon).
Its tag carries a digest of the recipe, so a changed recipe needs a rebuild.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from . import polyglot_data

IMAGE_REPOSITORY = "artask-polyglot"
RECIPE_LABEL = "org.autoresearch-tasks.polyglot.recipe"
BASE_IMAGE = "buildpack-deps:jammy@sha256:d4772f876f8647b56d4afc4989d526e4c4111fc39177d175c5a70d817c184858"
VERSIONS = {"go": "1.21.5", "rust": "1.90.0", "node": "20.18.1", "pytest": "8.3.4", "jest": "29.7.0", "openjdk": "21 (Ubuntu jammy package)",
            "python": "3.11 (deadsnakes PPA, as in Aider's image)", "gradle": "8.7 (the exercises' wrapper)"}
NPM_PACKAGES = ("jest@29.7.0", "@babel/core@7.25.2", "@exercism/babel-preset-javascript@0.2.1", "@exercism/eslint-config-javascript@0.6.0",
                "@types/jest@29.5.12", "@types/node@20.12.12", "babel-jest@29.6.4", "core-js@3.37.1", "eslint@8.49.0")

DOCKERFILE = r"""FROM {base}
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common cmake libboost-dev \
        libboost-date-time-dev libtbb-dev ca-certificates-java openjdk-21-jdk-headless \
    && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m venv /opt/py311 && /opt/py311/bin/pip install --no-cache-dir pytest=={pytest}
RUN ARCH=$(dpkg --print-architecture) && curl -fsSL "https://go.dev/dl/go{go}.linux-$ARCH.tar.gz" -o /tmp/go.tgz \
    && tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz
RUN curl -fsSL https://sh.rustup.rs -o /tmp/rustup.sh && sh /tmp/rustup.sh -y --profile minimal --default-toolchain {rust} \
    && rm /tmp/rustup.sh
RUN ARCH=$(dpkg --print-architecture | sed 's/amd64/x64/') \
    && curl -fsSL "https://nodejs.org/dist/v{node}/node-v{node}-linux-$ARCH.tar.xz" -o /tmp/node.txz \
    && tar -C /usr/local --strip-components=1 -xJf /tmp/node.txz && rm /tmp/node.txz
ENV PATH=/opt/py311/bin:/root/.cargo/bin:/usr/local/go/bin:$PATH GOTOOLCHAIN=local
RUN mkdir -p /npm-install && cd /npm-install && npm init -y > /dev/null \
    && npm install --no-audit --no-fund {npm}
COPY warm /opt/warm
RUN cd /opt/warm/java && chmod +x gradlew && ./gradlew test --no-daemon -q && rm -rf build .gradle
RUN cd /opt/warm/rust && cargo fetch
RUN cd /opt/warm/go && go test ./... && go build std
ENV CARGO_NET_OFFLINE=true
WORKDIR /work
"""

WARM_FILES = {
    "java/build.gradle": """plugins {
    id "java"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation "io.reactivex.rxjava2:rxjava:2.2.12"
    implementation "org.json:json:20190722"
    testImplementation platform("org.junit:junit-bom:5.10.0")
    testImplementation "org.junit.jupiter:junit-jupiter"
    testImplementation "org.assertj:assertj-core:3.25.1"
}

test {
    useJUnitPlatform()
}
""",
    "java/src/main/java/Warm.java": "public class Warm {\n    public int answer() {\n        return 42;\n    }\n}\n",
    "java/src/test/java/WarmTest.java": """import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

public class WarmTest {
    @Test
    public void answer() {
        assertThat(new Warm().answer()).isEqualTo(42);
    }
}
""",
    # Every crate named in a Cargo.toml or .meta/Cargo-example.toml of the benchmark, pinned so that the
    # offline resolution inside a test run picks these versions.
    "rust/Cargo.toml": """[package]
name = "warm"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "=1.0.98"
counter = "=0.5.7"
itertools = "=0.5.10"
num-bigint = "=0.4.6"
num-traits = "=0.2.19"
permutohedron = "=0.2.4"
rand = "=0.8.5"
rand03 = { package = "rand", version = "=0.3.23" }
regex = "=0.2.11"
thiserror = "=1.0.69"
time = "=0.3.37"
""",
    "rust/src/lib.rs": "pub fn answer() -> u32 {\n    42\n}\n",
    "go/go.mod": "module warm\n\ngo 1.18\n",
    "go/warm.go": "package warm\n\nimport (\n\t\"fmt\"\n\t\"sort\"\n\t\"strings\"\n)\n\nfunc Answer() string {\n\tvalues := []int{3, 1, 2}\n\tsort.Ints(values)\n\treturn strings.TrimSpace(fmt.Sprint(values))\n}\n",
    "go/warm_test.go": "package warm\n\nimport \"testing\"\n\nfunc TestAnswer(t *testing.T) {\n\tif Answer() != \"[1 2 3]\" {\n\t\tt.Fatal(Answer())\n\t}\n}\n",
}
WRAPPER_FILES = ("gradlew", "gradle/wrapper/gradle-wrapper.jar", "gradle/wrapper/gradle-wrapper.properties")


def dockerfile():
    return DOCKERFILE.format(base=BASE_IMAGE, npm=" ".join(NPM_PACKAGES), **{k: v for k, v in VERSIONS.items() if k in ("go", "rust", "node", "pytest")})


def wrapper_source():
    """A prepared Java exercise whose Gradle wrapper (identical in every Java exercise) seeds the image's Gradle cache."""
    records = polyglot_data.manifest()["exercises"]
    first = sorted(i for i, r in records.items() if r["language"] == "java")[0]
    return polyglot_data.Exercise(first).root


def recipe_digest():
    parts = {"Dockerfile": hashlib.sha256(dockerfile().encode()).hexdigest()}
    parts.update({"warm/" + rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in WARM_FILES.items()})
    root = wrapper_source()
    parts.update({"warm/java/" + rel: polyglot_data.file_sha256(root / rel) for rel in WRAPPER_FILES})
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def image_tag():
    return f"{IMAGE_REPOSITORY}:{recipe_digest()[:12]}"


def write_context(directory):
    directory = Path(directory)
    (directory / "Dockerfile").write_text(dockerfile())
    for rel, text in WARM_FILES.items():
        path = directory / "warm" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    root = wrapper_source()
    for rel in WRAPPER_FILES:
        target = directory / "warm" / "java" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)
    return directory


def docker_available():
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def inspect(tag):
    """``docker image inspect`` record of ``tag`` or None."""
    proc = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return None
    records = json.loads(proc.stdout or "[]")
    return records[0] if records else None


def require_image():
    """Return ``{tag, id, architecture}`` of the evaluator image; raise RuntimeError with the fix when it is unusable."""
    if not docker_available():
        raise RuntimeError("Docker is not available (start it, e.g. `colima start`); the polyglot tests run in a local container")
    tag = image_tag()
    record = inspect(tag)
    if record is None:
        raise RuntimeError(f"Docker image {tag} is missing; build it with `python scripts/prepare_polyglot.py --build-image`")
    labels = (record.get("Config") or {}).get("Labels") or {}
    if labels.get(RECIPE_LABEL) != recipe_digest():
        raise RuntimeError(f"Docker image {tag} was not built from the current recipe; rebuild it with `python scripts/prepare_polyglot.py --build-image`")
    return {"tag": tag, "id": record.get("Id"), "architecture": record.get("Architecture")}


def build(context_dir, log=print):
    """Build the image from a fresh context directory (needs the network); returns the image record."""
    write_context(context_dir)
    tag, digest = image_tag(), recipe_digest()
    command = ["docker", "build", "--label", f"{RECIPE_LABEL}={digest}", "-t", tag, str(context_dir)]
    env = dict(os.environ)
    if subprocess.run(["docker", "buildx", "version"], capture_output=True).returncode != 0:
        env["DOCKER_BUILDKIT"] = "0"  # classic builder when the buildx plugin is not installed
    log("building " + tag + ": " + " ".join(command) + (" (DOCKER_BUILDKIT=0)" if env.get("DOCKER_BUILDKIT") == "0" else ""))
    subprocess.run(command, check=True, env=env)
    return require_image()
