# LLVM function inlining for binary size

**Paper:** Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve (https://arxiv.org/pdf/2601.21096)\
**Task id:** `magellan-llvm-function-inlining-for-binary-size`\
**Protocol:** `recreated`\
**Direction:** minimize\

## Task

LLVM function inlining for binary size task evaluated in Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve.

Benchmark named in the paper: LLVM function inlining for binary size. Paper metric: binary size.

## Candidate interface

should_inline(features) -> bool, called by LLVM's ML inline advisor (interactive mode) for every eligible call site; features is a dict of int values with LLVM 23's MLInlineAdvisor feature names: sroa_savings, sroa_losses, load_elimination, call_penalty, call_argument_setup, load_relative_intrinsic, lowered_call_arg_setup, indirect_call_penalty, jump_table_penalty, case_cluster_penalty, switch_default_dest_penalty, switch_penalty, unsimplified_common_instructions, num_loops, dead_blocks, simplified_instructions, constant_args, constant_offset_ptr_args, callsite_cost, cold_cc_penalty, last_call_to_static_bonus, is_multiple_blocks, nested_inlines, nested_inline_cost_estimate, threshold, callee_basic_block_count, callsite_height, node_count, nr_ctant_params, cost_estimate, edge_count, caller_users, caller_conditionally_executed_blocks, caller_basic_block_count, callee_conditionally_executed_blocks, callee_users, is_callee_avail_external, is_caller_avail_external. Legality and mandatory inlining stay with LLVM. Score (minimize): text-section bytes of the sqlite3 shell (SQLite amalgamation, clang -Oz bitcode, opt default<O2>, llc -O2, linked) built with the policy divided by the same build with LLVM's default inlining heuristic. Needs opt, llc, llvm-size and clang of one LLVM >= 17 installation (PATH, LLVM_BIN or Homebrew's llvm keg)

Parameters: `{"archive": "sqlite-amalgamation-3530400.zip", "folder": "sqlite-amalgamation-3530400"}`

## Labels

Scientific contribution: `engineering` (tier 3); artifact `program`, domain `systems`, data `public_benchmark`, verification `simulation`, reference `none`, resources `cpu`. See [docs/LABELS.md](../../docs/LABELS.md).

## Importance to science

A task is important to science when it addresses a substantive scientific or real-world problem, and a cited research paper's main objective is to solve that problem or advance understanding of it. The evidence must explain why the problem matters and how the local task represents it. Using a task to evaluate an agent, framework, or general-purpose model, or citing only a benchmark, dataset, or tool, does not establish scientific importance.

Evidence status: **unreviewed**.


No qualifying paper whose main objective is this underlying scientific or real-world problem has been established in the evidence registry. A task's use in an agent/framework evaluation is insufficient. This label describes the available evidence, not proof that the task has no scientific value.

## Evaluation protocol

Locally implemented or adapted evaluator for the cited task or benchmark. Data sources, generated instances, evaluation budgets and protocol differences are specified in the candidate contract, requirements and notes; scores are not automatically comparable with published results.

- Clearly labelled proxy of Magellan's feature-based setting. Magellan evolves C++ inlining heuristics in two settings: feature-based policies that extend MLModelRunner::evaluateUntyped and compose the 38 MLInlineAdvisor features (4.27% size reduction), and API-level policies implemented in MLInlineAdvisor::getAdviceImpl (5.23%); both run on Google's internal search application with Google's internal clang/LLVM toolchain (which tracks LLVM tip-of-tree) and reward the final linked binary size relative to LLVM's upstream heuristic. This task keeps the decision point (LLVM's ML inline advisor; legality and mandatory inlining remain LLVM's) and the feature-based policy space, but the policy is a Python function answering LLVM's interactive model-runner channel (-enable-ml-inliner=release -inliner-interactive-channel-base); there is no API-level setting and no autotuner. The paper prints its evolved binary-size and performance policies (Appendices A and B); they are not used as seeds or references here.
- Workload: the sqlite3 shell built from the SQLite 3.53.4 amalgamation (sqlite3.c and shell.c, public domain; fetched by scripts/prepare_or_csp.py and checked against the SHA3-256 published on sqlite.org) instead of Google's internal application. Each unit is compiled with clang -Oz -Xclang -disable-llvm-optzns to bitcode, optimised with opt -passes=default<O2> (LLVM 23 rejects the Oz pipeline name; size optimisation is carried by the minsize function attributes), code-generated with llc -O2 and linked with clang. Score: text-section bytes of the linked binary / the same build with LLVM's default inlining heuristic, rebuilt in every evaluation (minimize; below 1 beats the default). The binary is not executed.
- Reference run (Homebrew LLVM 23.1.1, arm64 macOS): default heuristic 692,952 text bytes; the never-inline seed (Magellan's starting point) 700,852 bytes, score 1.0114; a cost_estimate <= threshold policy 692,884 bytes. Sizes depend on the LLVM version, the target and the system headers; MLInlineAdvisor's feature set depends on the LLVM version.
- Verification: answering every decision with LLVM's own default decision (sent only when -inliner-interactive-include-default is given, which candidate runs do not use) reproduces the default heuristic's text size exactly (692,952 bytes over 19,699 decisions on the two units), which checks the channel protocol and that the decisions are applied as given. A policy that raises or returns a non-bool makes the candidate invalid; the compiler is killed rather than left waiting on the pipe.
- The task needs opt, llc, llvm-size and clang from one LLVM >= 17 installation with the interactive ML inliner (found on PATH, in LLVM_BIN or in Homebrew's llvm keg). should_inline and the module's top-level code run in the evaluator's process and can reach the scoring code there, so scores of untrusted candidates are not tamper-proof.
- Hosts LLVM's interactive ML-inliner channel; replaying LLVM's own default decisions through it reproduces the default heuristic's text size exactly.

## Run

```
python evaluator.py initial_program.py
```

The evaluator prints a JSON result with `combined_score` (the value SCiEL maximizes), `score` (the task metric in its own direction), `valid` and `error`. `config.yaml` is a ready SCiEL configuration; copied SCiEL examples keep their upstream configuration.

## Requirements

- `py:numpy`
- `bin:opt`
- `bin:llc`
- `bin:llvm-size`
- `bin:clang`
- `file:data/or_csp/sqlite/sqlite-amalgamation-3530400.zip`

## Sources

- paper: https://arxiv.org/pdf/2601.21096 (Sections 3.1-3.3)
- archive: https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip (SQLite 3.53.4 amalgamation: the program whose sqlite3 shell is built to measure text size)

## Limitations

- No official public runnable implementation verified from the inspected paper. External benchmark code may exist separately.
- No publicly accessible, paper-linked implementation with a verified immutable revision was established.
