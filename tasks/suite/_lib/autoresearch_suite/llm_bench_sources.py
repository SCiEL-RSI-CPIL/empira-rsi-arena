"""Public sources of the llm_bench package and the derivations of its prepared files.

``SOURCES`` maps a source name to its pinned URL (Git commit or Hugging Face
revision) and license; ``DERIVED`` maps each prepared file (``data/llm_bench``
in the task cache) to the sources it needs and a deterministic
function ``derive(paths) -> bytes``.  ``scripts/prepare_llm_bench.py`` downloads
the sources, checks them against ``catalog/llm_bench-data-lock.json`` and
writes the derived files, whose hashes are locked too.
"""
from __future__ import annotations

import json

from . import llm_bench_lcb

ZERO_SHOT_COT = "https://raw.githubusercontent.com/kojima-takeshi188/zero_shot_cot/5ef330fcdeec0cd26aee27943504f91f8ec1c33c"
LLM4OR = "https://raw.githubusercontent.com/LLM4OR/LLM4OR/75962df5ddf531906fda76f8708c658114e63c0e/static/clean_benchmarks"
# Problem directories of the cleaned NLP4LP set (178 problems after removing the samples flagged as erroneous).
NL4LP_IDS = [int(x) for x in """0 1 3 4 6 7 9 10 11 12 14 15 16 17 18 19 20 21 22 23 26 28 30 32 33 35 37 38 40 41 42 43 44 46 47 49 51 52 53 54
 55 58 60 61 62 64 65 66 68 69 70 71 72 75 77 79 80 81 84 85 86 87 89 91 92 93 96 97 98 99 101 106 107 108 109 110 111 118 120 121
 123 124 126 128 129 132 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 152 153 154 155 156 159 160 161 162 163 164
 165 166 167 168 169 171 172 173 174 175 176 179 181 182 184 185 186 188 192 194 195 198 199 200 201 202 205 207 208 209 210 211 212
 214 216 217 220 221 222 224 225 226 227 228 230 232 233 234 238 240 241 242 244 245 246 247 248 251 252 253 254 257 259 261 267""".split()]
MCE = "https://raw.githubusercontent.com/metaevo-ai/mce-artifact/c63300daef93404e22e2569b3d08e72843e7123b/env"
MCE_LICENSE = ("MCE artifact repository (pyproject declares MIT; no LICENSE file). FiNER-139 (Loukas et al., ACL 2022) is CC-BY-SA-4.0; "
               "USPTO-50k (Schneider et al., 2016) derives from Lowe's USPTO reaction extraction (CC0). Not redistributed by this repository.")
JAILBREAK = "https://huggingface.co/datasets/jackhhao/jailbreak-classification/resolve/2f2ceeb39658696fd3f462403562b6eea5306287"
GEPA_IFBENCH = "https://raw.githubusercontent.com/gepa-ai/gepa-artifact/cbefbc1aa0f43dd39874ec4bf42211365dbda42e/gepa_artifact/benchmarks/IFBench"
IFBENCH_LICENSE = ("gepa-artifact repository (MIT); IFBench test prompts and instruction checkers come from allenai/IFBench (Apache-2.0 code, ODC-BY data), "
                   "itself extending google-research IFEval checkers (Apache-2.0). Not redistributed by this repository.")
IFBENCH_CHECKERS = ("instructions.py", "instructions_registry.py", "instructions_util.py", "instructions_ifeval.py",
                    "instructions_registry_ifeval.py", "instructions_util_ifeval.py")
NLTK_DATA = "https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages"
NLTK_PACKAGES = {"tokenizers/punkt_tab.zip": "Punkt tokenizer models (Jan Strunk); NLTK data package, no license field in the NLTK index",
                 "corpora/stopwords.zip": "Stopwords corpus; NLTK data package, no license field in the NLTK index",
                 "taggers/averaged_perceptron_tagger_eng.zip": "Averaged perceptron tagger (MIT License per the NLTK index)"}
NATURAL_PLAN = "https://raw.githubusercontent.com/google-deepmind/natural-plan/ca76db336072ff8931db43bc1ca8d381038cf073"
LEETCODE_CONTEST = "https://huggingface.co/datasets/lbaf23/LeetCode-Contest/resolve/05db6cfbad9648c312e0aff99144e21aa5781514"
CRAVE = "https://huggingface.co/datasets/TuringEnterprises/CRAVE/resolve/4d94975d6c9b70d7bcf611b15aadf3d89fe22870"
IMOBENCH = "https://raw.githubusercontent.com/google-deepmind/superhuman/c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8/imobench"
IMOBENCH_LICENSE = "CC-BY-4.0 data, Apache-2.0 code (google-deepmind/superhuman, IMO-Bench). Not redistributed by this repository."
# Hyperagents' held-out IMO-GradingBench subset: rows with 0, 1, 6 or 7 points, shuffled and sampled (random_state 42),
# then stratified train/val/test splits of 100 each (domains/imo/curate_subsets.py); computed with pandas 2.2.3 and scikit-learn 1.9.1.
GRADINGBENCH_TEST = ["GB-" + x for x in """0083 0539 0687 0760 0309 0495 0730 0088 0253 0194 0031 0667 0597 0487 0901 0817 0679 0899 0447 0706 0931 0256
 0779 0575 0932 0788 0167 0217 0851 0233 0804 0643 0296 0516 0080 0441 0858 0285 0132 0862 0424 0780 0053 0188
 0052 0551 0295 0297 0782 0844 0745 0345 0829 0281 0121 0211 0156 0085 0930 0547 0318 0438 0548 0045 0036 0292
 0163 0483 0918 0195 0654 0803 0661 0087 0484 0709 0629 0418 0210 0086 0883 0724 0293 0635 0051 0935 0435 0482
 0923 0326 0694 0589 0542 0022 0949 0758 0337 0995 0444 0028""".split()]
HYPERAGENTS = "https://raw.githubusercontent.com/facebookresearch/Hyperagents/59a68f672dfb92c74aeb7e61535d776fb36e172d"
# Hyperagents' held-out paper-review split (domains/paper_review/curate_subsets.py: shuffles with random_state 42 and
# balanced 50/50 accept/reject train, val and test splits of 100), computed here with pandas 2.2.3; order as in that split.
PAPER_REVIEW_TEST = """w0b7fCX2nN vMIVqlEWRw wKOoWTBMZe rcXXNFVlEn eTHawKFT4h ta26LtNq2r 9O2sVnEHor wO1NJLitPL DkeeXVdQyu A3YUPeJTNR
 UdxpjKO2F9 RvUVMjfp8i 10BTKkFfhl UvTo3tVBk2 8QTpYC4smR 5ES5Hdlbxw zDJNUDprhW yx8bU8T5ZN EO8xpnW7aX 7BLXhmWvwF
 6ZuDeSHzjj wJ6Bx1IYrQ DqfdhM64LI Ddak3nSqQM brOAVSPPjw On5WIN7xyD RSGNGiB1q4 zWYHsbuedA vf5aUZT0Fz 4jzjexvjI7
 TXoZiUZywf QkLpGxUboF xcHIiZr3DT SxRblm9aMs g3PuaFh5vV 0BqyZSWfzo yyIHdaSDUU RVSQpkfsLq SQrHpTllXa yC3q7vInux
 8aDG51pxFc 0BfQT652sC 5zSCSE0k41 51cjeYcXjs BHXsb69bSx ppJuFSOAnM zKoUV1wHRJ aVh9KRZdRk z7PhIgVmZU abuQMKDVkW
 fxNQJVMwK2 25kAzqzTrz QEHrmQPBdd nzkWhoXUpv cnpkzQZaLU Z91rwXnJsw et5l9qPUhm wWcNhS4g1U OeQE9zsztS cPmLjxedbD
 06ZvHHBR0i PcE0yAGAGW Vi1PJjvEdh R6KJN1AUAR wMRFTQwp1d yrnrvfXFaV FXJDcriMYH zEUDoD9cU9 xtp6QPnwLu vM4CdVScT8
 2CxkRDMIG4 T56j6aV8Oc L3tW9nbcEM Oo7dlLgqQX plAiJUFNja xRi8sKo4XI xAZLCWbsTF 3j2nasmKkP xVOMtecrAS AfZH9EEuRR
 KncRpAnprQ vfEqSWpMfj cc8h3I3V4E eJZ5vJEaaa 4xWQS2z77v o1efpbvR6v aIPwlkdOut z8sxoCYgmd 4mxzxYhMuN vqJZb9SX1T
 2DD4AXOAZ8 o6aUi3ukdd gENfMmUIkT x7NbaU8RSU OEL4FJMg1b veNewXAdHE xTrAA3UKPa qjnl1QUnFA zZUCWkn4PL vnp2LtLlQg""".split()
ARXIVMATH = {"arxivmath-1225": "https://huggingface.co/datasets/MathArena/arxivmath-1225/resolve/90c60aa1eadbb72168640169d82154b079cdeaac",
             "arxivmath-0126": "https://huggingface.co/datasets/MathArena/arxivmath-0126/resolve/bac939ba603c93f905aa48072980ea681a2824cd"}
TRAVELPLANNER = "https://huggingface.co/datasets/osunlp/TravelPlanner/resolve/8736504ecfc31b7f8b7e40122873c337e83fff7c"
TRAVEL_DB = "https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard/resolve/a57daf2e45155a5378dcf75ee9fe374e671ee048/database"
TRAVEL_LICENSE = "CC-BY-4.0 (TravelPlanner dataset and leaderboard-space database; evaluation code MIT). Not redistributed by this repository."
TRAVEL_TABLES = {"travel_accommodations.csv": "accommodations/clean_accommodations_2022.csv", "travel_attractions.csv": "attractions/attractions.csv",
                 "travel_restaurants.csv": "restaurants/clean_restaurant_2022.csv", "travel_distance.csv": "googleDistanceMatrix/distance.csv",
                 "travel_city_states.txt": "background/citySet_with_states.txt"}
PUPA = "https://huggingface.co/datasets/Columbia-NLP/PUPA/resolve/9981b49b6ced0033988a224b6712895ebf119294"
LLM4OR_LICENSE = ("The LLM4OR repository (cleaned benchmarks of Xiao et al., 'A Survey of Optimization Modeling Meets LLMs', IJCAI 2025) declares no license; "
                  "the underlying sets are NLP4LP (OptiMUS, AhmadiTeshnizi et al.) and OptiBench/ReSocratic (Yang et al., ICML 2024, Apache-2.0 repository). Not redistributed by this repository.")

SOURCES = {
    "addsub-raw": {"url": f"{ZERO_SHOT_COT}/dataset/AddSub/AddSub.json",
                   "license": "No license file in the zero_shot_cot repository; AddSub (Hosseini et al., EMNLP 2014) is redistributed there for research. Not redistributed by this repository."},
    "singleeq-raw": {"url": f"{ZERO_SHOT_COT}/dataset/SingleEq/questions.json",
                     "license": "No license file in the zero_shot_cot repository; SingleEq (Koncel-Kedziorski et al., TACL 2015) is redistributed there for research. Not redistributed by this repository."},
    "llm4or-resocratic": {"url": f"{LLM4OR}/ReSocratic.jsonl", "license": LLM4OR_LICENSE},
    "ifbench-test": {"url": f"{GEPA_IFBENCH}/data/IFBench_test.jsonl", "license": IFBENCH_LICENSE},
    **{f"ifbench-{name}": {"url": f"{GEPA_IFBENCH}/utils_ifbench/{name}", "license": IFBENCH_LICENSE} for name in IFBENCH_CHECKERS},
    **{f"nltk-{name.split('/')[1][:-4]}": {"url": f"{NLTK_DATA}/{name}", "license": text} for name, text in NLTK_PACKAGES.items()},
    "natural-plan-meeting": {"url": f"{NATURAL_PLAN}/data/meeting_planning.json",
                             "license": "CC-BY-4.0 data, Apache-2.0 code (google-deepmind/natural-plan). Not redistributed by this repository."},
    "leetcode-contest": {"url": f"{LEETCODE_CONTEST}/test.jsonl",
                         "license": "MIT (lbaf23/LeetCode-Contest, CoCoEvo); problem statements and tests collected from LeetCode contests. Not redistributed by this repository."},
    **llm_bench_lcb.source_specs(),
    "crave-test": {"url": f"{CRAVE}/test.parquet",
                   "license": "MIT (TuringEnterprises/CRAVE); patches and pull-request text come from public GitHub repositories under their own licenses. Not redistributed by this repository."},
    "imobench-gradingbench": {"url": f"{IMOBENCH}/gradingbench.csv", "license": IMOBENCH_LICENSE},
    "imobench-proofbench": {"url": f"{IMOBENCH}/proofbench.csv", "license": IMOBENCH_LICENSE},
    "hyperagents-paper-review": {"url": f"{HYPERAGENTS}/domains/paper_review/dataset.csv",
                                 "license": "CC-BY-NC-SA-4.0 (Hyperagents repository); paper texts and decisions of public OpenReview submissions collected by APReS (Zhao et al., 2026). Not redistributed by this repository."},
    **{name: {"url": f"{base}/data/train-00000-of-00001.parquet",
              "license": "CC-BY-SA-4.0 (MathArena ArXivMath; questions written from recent arXiv papers). Not redistributed by this repository."}
       for name, base in ARXIVMATH.items()},
    "travelplanner-validation": {"url": f"{TRAVELPLANNER}/validation.csv", "license": TRAVEL_LICENSE},
    "travelplanner-flights": {"url": f"{TRAVEL_DB}/flights/clean_Flights_2022.csv", "license": TRAVEL_LICENSE},
    **{f"travelplanner-{name.split('.')[0]}": {"url": f"{TRAVEL_DB}/{path}", "license": TRAVEL_LICENSE} for name, path in TRAVEL_TABLES.items()},
    "pupa-new": {"url": f"{PUPA}/PUPA_New.csv", "license": "MIT (Columbia-NLP/PUPA, PAPILLON). Not redistributed by this repository."},
    "jailbreak-test": {"url": f"{JAILBREAK}/balanced/jailbreak_dataset_test_balanced.csv",
                       "license": "Apache-2.0 (jackhhao/jailbreak-classification; jailbreak prompts from verazuo/jailbreak_llms, benign prompts from OpenOrca and GPTeacher). Not redistributed by this repository."},
    "mce-finer-test": {"url": f"{MCE}/finer/data/test.jsonl", "license": MCE_LICENSE},
    "mce-finer-context": {"url": f"{MCE}/finer/finer_context.py", "license": MCE_LICENSE},
    "mce-uspto-test": {"url": f"{MCE}/uspto/data/test.jsonl", "license": MCE_LICENSE},
    "llm4or-nlp4lp": {"base": f"{LLM4OR}/NL4LP", "files": [f"{i}/{name}" for i in NL4LP_IDS for name in ("description.txt", "solution.json")],
                      "license": LLM4OR_LICENSE},
}


# Local CPU models used by the LMX sentence tasks, fetched into the Hugging Face cache at pinned revisions
# by `python scripts/prepare_llm_bench.py --models`; the weight files are hash-checked.
MODELS = {
    "sentiment": {"repo": "cardiffnlp/twitter-roberta-base-sentiment-latest", "revision": "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
                  "files": ["config.json", "merges.txt", "vocab.json", "special_tokens_map.json", "pytorch_model.bin"],
                  "weights": {"pytorch_model.bin": "4d24a3e32a88ed1c4e5b789fc6644e2e767500554e954b27dccf52a8e762cbae"}, "license": "CC-BY-4.0"},
    "embedding": {"repo": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                  "files": ["modules.json", "config_sentence_transformers.json", "sentence_bert_config.json", "config.json", "model.safetensors",
                            "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "1_Pooling/config.json"],
                  "weights": {"model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"}, "license": "Apache-2.0"},
}


def dumps(value):
    """Canonical JSON bytes so that derived files hash identically on every machine."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode("utf-8")


def _arithmetic(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        solutions = row["lSolutions"]
        if len(solutions) != 1:
            raise ValueError("expected one solution per arithmetic problem")
        out.append({"id": int(row["iIndex"]), "question": str(row["sQuestion"]).strip(), "answer": float(solutions[0])})
    if len({row["id"] for row in out}) != len(out):
        raise ValueError("duplicate problem identifiers")
    return dumps(out)


def _nlp4lp(directory):
    out = []
    for i in NL4LP_IDS:
        solution = json.loads((directory / str(i) / "solution.json").read_text(encoding="utf-8"))
        objective = float(solution["objective"])
        out.append({"id": i, "question": (directory / str(i) / "description.txt").read_text(encoding="utf-8").strip(), "answer": objective})
    return dumps(out)


def _resocratic(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        results = {str(key): float(value) for key, value in row["results"].items()}
        if not results:
            raise ValueError("a ReSocratic item without requested quantities")
        out.append({"id": int(row["index"]), "question": str(row["question"]).strip(), "results": results, "type": row.get("type")})
    if len({row["id"] for row in out}) != len(out):
        raise ValueError("duplicate ReSocratic indices")
    return dumps(out)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finer(test, context_file):
    import ast
    context = None
    for node in ast.parse(context_file.read_text(encoding="utf-8")).body:  # read the literal; never execute the file
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "FINER_CONTEXT":
            context = ast.literal_eval(node.value)
    if not context:
        raise ValueError("FINER_CONTEXT not found")
    items = [{"id": i, "question": row["question"], "answer": row["target"]} for i, row in enumerate(_jsonl(test))]
    return dumps({"context": context, "items": items})


def _uspto(test):
    items = []
    for i, row in enumerate(_jsonl(test)):
        question = row["question"].strip()
        if question.endswith("Answer:"):
            question = question[: -len("Answer:")].rstrip()
        items.append({"id": i, "question": question, "answer": row["target"]})
    return dumps(items)


def _jailbreak(path):
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    labels = {"jailbreak": 1, "benign": 0}
    out = [{"id": i, "text": row["prompt"], "label": labels[row["type"].strip()]} for i, row in enumerate(rows)]
    if len({row["text"] for row in out}) != len(out):
        raise ValueError("duplicate prompts in the jailbreak test split")
    return dumps(out)


def _ifbench(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if len(row["instruction_id_list"]) != len(row["kwargs"]):
                raise ValueError("IFBench row with mismatched instructions and kwargs")
            out.append({"id": str(row["key"]), "prompt": row["prompt"], "instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]})
    if len({row["id"] for row in out}) != len(out):
        raise ValueError("duplicate IFBench keys")
    return dumps(out)


def _copy(name):
    return lambda paths: paths[name].read_bytes()


def _meeting_test(path):
    """Mind Evolution's test split: the last 50 of the 100 problems at each number of people (Appendix B)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = sorted(data, key=lambda key: int(key.rsplit("_", 1)[1]))
    out = []
    for position, key in enumerate(keys):
        row = data[key]
        if row["num_people"] != position // 100 + 1:
            raise ValueError("unexpected ordering of the meeting-planning problems")
        if position % 100 >= 50:
            out.append({"id": int(key.rsplit("_", 1)[1]), "num_people": row["num_people"], "prompt_5shot": row["prompt_5shot"],
                        "constraints": row["constraints"], "dist_matrix": row["dist_matrix"], "golden_plan": row["golden_plan"]})
    return dumps(out)


def _leetcode_contest(path):
    """CoCoEvo's 80 contest problems; each assert 'f(args) == literal' becomes a call and an encoded expected value."""
    import ast
    from .llm_bench_code import encode
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tests = []
        for test in row["tests"]:
            node = ast.parse(test).body[0]
            if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq) and isinstance(node.test.left, ast.Call)):
                raise ValueError(f"unexpected test form: {test[:80]}")
            tests.append({"call": ast.get_source_segment(test, node.test.left),
                          "expected": encode(ast.literal_eval(node.test.comparators[0]))})
        out.append({"id": int(row["index"]), "title": row["title"], "difficulty": row["difficulty"], "contest": row["contest"],
                    "entry_point": row["entry_point"], "prompt": row["prompt"], "tests": tests})
    return dumps(out)


def _crave(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path).to_pylist()
    out = []
    for i, row in enumerate(table):
        if row["label"] not in ("APPROVE", "REQUEST_CHANGES"):
            raise ValueError("unexpected CRAVE label")
        out.append({"id": i, "url": row["original_pull_request_url"], "repo": row["repo"], "title": row["pull_request_title"],
                    "description": row["description"] or "", "patch": row["patch"], "label": row["label"]})
    return dumps(out)


def _gradingbench(path):
    import csv
    import io
    rows = {row["Grading ID"]: row for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))}
    out = []
    for grading_id in GRADINGBENCH_TEST:
        row = rows[grading_id]
        if int(row["Points"]) not in (0, 1, 6, 7):
            raise ValueError("the Hyperagents subset only keeps 0, 1, 6 or 7 points")
        out.append({"id": grading_id, "problem_id": row["Problem ID"], "problem": row["Problem"], "solution": row["Solution"],
                    "guidelines": row["Grading guidelines"], "response": row["Response"], "points": int(row["Points"]), "reward": row["Reward"]})
    return dumps(out)


def _paper_review(path):
    import csv
    import io
    csv.field_size_limit(1 << 30)
    rows = {row["question_id"]: row for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))}
    out = [{"id": key, "paper_text": rows[key]["paper_text"], "outcome": rows[key]["outcome"]} for key in PAPER_REVIEW_TEST]
    if sorted(row["outcome"] for row in out) != ["accept"] * 50 + ["reject"] * 50:
        raise ValueError("the paper-review test split must hold 50 accepted and 50 rejected papers")
    return dumps(out)


def _arxivmath(paths):
    import pyarrow.parquet as pq
    out = []
    for name in sorted(ARXIVMATH):
        for row in pq.read_table(paths[name]).to_pylist():
            out.append({"id": f"{name}-{int(row['problem_idx'])}", "problem": row["problem"], "answer": row["answer"], "source": row.get("source")})
    if len(out) != 40:
        raise ValueError("ArXivMath December 2025 and January 2026 hold 17 + 23 problems")
    return dumps(out)


def _travel_validation(path):
    from .llm_bench_travel import parse_validation_csv
    return dumps(parse_validation_csv(path.read_text(encoding="utf-8")))


def _travel_flights(flights, validation):
    """Flight rows of every route and date listed in the validation queries' reference information (raw CSV lines)."""
    import csv
    import io
    import re
    from .llm_bench_travel import parse_validation_csv
    routes = set()
    for row in parse_validation_csv(validation.read_text(encoding="utf-8")):
        for unit in row["reference_information"]:
            match = re.match(r"Flight from (.+) to (.+) on (\d{4}-\d{2}-\d{2})$", unit["Description"])
            if match:
                routes.add((match.group(1).strip(), match.group(2).strip(), match.group(3)))
    lines = flights.read_text(encoding="utf-8").splitlines(keepends=True)
    header = next(csv.reader([lines[0]]))
    origin, dest, date = header.index("OriginCityName"), header.index("DestCityName"), header.index("FlightDate")
    kept = [lines[0]] + [line for line in lines[1:] if (lambda r: (r[origin], r[dest], r[date]) in routes)(next(csv.reader(io.StringIO(line))))]
    return "".join(kept).encode("utf-8")


def _pupa(path):
    """GEPA's PUPA test split: rows 222 to 442 of the pupa_new configuration (papillon_data.py)."""
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    out = [{"id": index, "user_query": row["user_query"], "target_response": row["target_response"],
            "pii_units": sorted({unit for unit in row["pii_units"].split("||") if unit})} for index, row in enumerate(rows) if 222 <= index < 443]
    if len(out) != 221:
        raise ValueError("GEPA's PUPA test split has 221 items")
    return dumps(out)


def _proofbench(path):
    import csv
    import io
    out = [{"id": row["Problem ID"], "problem": row["Problem"], "solution": row["Solution"], "guidelines": row["Grading guidelines"],
            "category": row["Category"], "level": row["Level"], "source": row["Source"]} for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))]
    if len(out) != 60:
        raise ValueError("IMO-ProofBench has 60 problems")
    return dumps(out)


DERIVED = {
    "addsub.json": {"sources": ["addsub-raw"], "derive": lambda paths: _arithmetic(paths["addsub-raw"])},
    "singleeq.json": {"sources": ["singleeq-raw"], "derive": lambda paths: _arithmetic(paths["singleeq-raw"])},
    "nlp4lp.json": {"sources": ["llm4or-nlp4lp"], "derive": lambda paths: _nlp4lp(paths["llm4or-nlp4lp"])},
    "resocratic.json": {"sources": ["llm4or-resocratic"], "derive": lambda paths: _resocratic(paths["llm4or-resocratic"])},
    "ifbench_test.json": {"sources": ["ifbench-test"], "derive": lambda paths: _ifbench(paths["ifbench-test"])},
    "ifbench_checkers/__init__.py": {"sources": [], "derive": lambda paths: b'"""IFBench instruction checkers (pinned copy prepared by scripts/prepare_llm_bench.py)."""\n'},
    **{f"ifbench_checkers/{name}": {"sources": [f"ifbench-{name}"], "derive": _copy(f"ifbench-{name}")} for name in IFBENCH_CHECKERS},
    **{f"nltk_data/{name}": {"sources": [f"nltk-{name.split('/')[1][:-4]}"], "derive": _copy(f"nltk-{name.split('/')[1][:-4]}")} for name in NLTK_PACKAGES},
    "meeting_planning_test.json": {"sources": ["natural-plan-meeting"], "derive": lambda paths: _meeting_test(paths["natural-plan-meeting"])},
    "leetcode_contest.json": {"sources": ["leetcode-contest"], "derive": lambda paths: _leetcode_contest(paths["leetcode-contest"])},
    "livecodebench_sica.json.gz": {"sources": sorted(llm_bench_lcb.source_specs()), "derive": llm_bench_lcb.derive},
    "crave_test.json": {"sources": ["crave-test"], "derive": lambda paths: _crave(paths["crave-test"])},
    "proofbench.json": {"sources": ["imobench-proofbench"], "derive": lambda paths: _proofbench(paths["imobench-proofbench"])},
    "gradingbench_test.json": {"sources": ["imobench-gradingbench"], "derive": lambda paths: _gradingbench(paths["imobench-gradingbench"])},
    "paper_review_test.json.gz": {"sources": ["hyperagents-paper-review"], "derive": lambda paths: _paper_review(paths["hyperagents-paper-review"])},
    "arxivmath.json": {"sources": sorted(ARXIVMATH), "derive": _arxivmath},
    "travelplanner_validation.json": {"sources": ["travelplanner-validation"], "derive": lambda paths: _travel_validation(paths["travelplanner-validation"])},
    "travel_flights.csv": {"sources": ["travelplanner-flights", "travelplanner-validation"],
                           "derive": lambda paths: _travel_flights(paths["travelplanner-flights"], paths["travelplanner-validation"])},
    **{name: {"sources": [f"travelplanner-{name.split('.')[0]}"], "derive": _copy(f"travelplanner-{name.split('.')[0]}")} for name in TRAVEL_TABLES},
    "pupa_test.json": {"sources": ["pupa-new"], "derive": lambda paths: _pupa(paths["pupa-new"])},
    "jailbreak.json": {"sources": ["jailbreak-test"], "derive": lambda paths: _jailbreak(paths["jailbreak-test"])},
    "finer.json": {"sources": ["mce-finer-test", "mce-finer-context"], "derive": lambda paths: _finer(paths["mce-finer-test"], paths["mce-finer-context"])},
    "uspto.json": {"sources": ["mce-uspto-test"], "derive": lambda paths: _uspto(paths["mce-uspto-test"])},
}
