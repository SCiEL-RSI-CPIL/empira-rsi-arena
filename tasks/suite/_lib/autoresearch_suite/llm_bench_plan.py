"""Natural-language planning benchmarks of Mind Evolution: ``solve(question, llm) -> plan text``.

Meeting Planning (Natural Plan) is scored with a port of the benchmark's own
evaluation script, ``evaluate_meeting_planning.py`` of google-deepmind/natural-plan
(Apache-2.0, revision ca76db336072ff8931db43bc1ca8d381038cf073): the plan text
after ``SOLUTION:`` is split on periods and simulated step by step; a problem is
solved when the plan schedules as many valid meetings as the golden plan.
TravelPlanner lives in ``llm_bench_travel`` and is dispatched from here.
"""
from __future__ import annotations

import collections
import datetime

from . import llm_bench_core as core
from .runtime import CandidateError


# Adapted from google-deepmind/natural-plan evaluate_meeting_planning.py, Copyright 2024 Google LLC, Apache-2.0.
def convert_to_time_obj(time_str):
    return datetime.datetime.strptime(time_str, "%I:%M%p")


def process_constraints(data):
    constraints = collections.defaultdict(dict)
    for name, location, times, meeting_time in data:
        constraints[name]["location"] = location
        constraints[name]["start_time"] = convert_to_time_obj(times.split("to")[0].strip())
        constraints[name]["end_time"] = convert_to_time_obj(times.split("to")[1].strip())
        constraints[name]["meeting_time"] = meeting_time
    return constraints


def validator_from_text(plan, processed_constraints, start_location, initial_time, dist_matrix):
    """Number of valid meetings before the first invalid step (official rule; errors end the simulation)."""
    met_with = {}
    score = 0
    cur_location = start_location
    cur_time = convert_to_time_obj(initial_time)
    for step in plan:
        try:
            if step.startswith("You start"):
                continue
            elif step.startswith("You travel"):
                destination = step.split("travel to ")[1].split(" in")[0].strip()
                cur_time = cur_time + datetime.timedelta(minutes=dist_matrix[cur_location][destination])
                cur_location = destination
            elif step.startswith("You wait"):
                end_time = convert_to_time_obj(step.split("wait until ")[1].split(".")[0].strip())
                if end_time <= cur_time:
                    raise ValueError("Cannot go backwards in time")
                cur_time = end_time
            elif step.startswith("You meet"):
                person = step.split("meet ")[1].split(" for")[0].strip()
                if person in met_with:
                    raise ValueError("Person already met")
                met_with[person] = 1
                new_time = cur_time + datetime.timedelta(minutes=processed_constraints[person]["meeting_time"])
                if (cur_location == processed_constraints[person]["location"] and cur_time >= processed_constraints[person]["start_time"]
                        and new_time <= processed_constraints[person]["end_time"]):
                    score += 1
                    cur_time = new_time
                else:
                    raise ValueError("Invalid meeting time or location")
            else:
                raise ValueError("Unknown plan format")
        except (ValueError, KeyError, IndexError):
            # The official script catches ValueError only; unknown places or people (KeyError) and
            # truncated steps (IndexError) would crash it, so they end the simulation here as well.
            break
    return score


def parse_text_plan(plan):
    prefix = "SOLUTION:"
    if prefix in plan:
        plan = plan[plan.find(prefix) + len(prefix):].strip()
    return [step.strip() for step in plan.split(".") if step.strip()]
# End of the adapted Natural Plan code.


def meeting_score(row, response):
    start_location, initial_time = row["constraints"][0]
    constraints = process_constraints(row["constraints"][1:])
    predicted = validator_from_text(parse_text_plan(response), constraints, start_location, initial_time, row["dist_matrix"])
    golden = validator_from_text(row["golden_plan"], constraints, start_location, initial_time, row["dist_matrix"])
    return {"score": float(predicted == golden), "meetings": predicted, "golden_meetings": golden}


def _meeting(P):
    rows = core.read_json("meeting_planning_test.json")
    items = [{"id": f"meeting-{row['id']}", "question": row["prompt_5shot"], "row": row} for row in rows]
    return items, lambda item, response: meeting_score(item["row"], response)


def evaluate(solve, P):
    core.require_signature(solve, 2, "solve(question, llm)")
    if P.get("benchmark") == "travelplanner":
        from . import llm_bench_travel
        items, scorer = llm_bench_travel.items(P)
    elif P.get("benchmark") == "meeting_planning":
        items, scorer = _meeting(P)
    else:
        raise CandidateError(f"unknown planning benchmark: {P.get('benchmark')}")
    sample = core.fixed_sample(items, core.sample_size(P), P.get("seed", 0))
    client = core.task_llm(P)
    out = core.run_items(sample, lambda item: core.checked_answer(solve(item["question"], client), "plan"), scorer, client)
    out["details"]["benchmark"] = P["benchmark"]
    return out
