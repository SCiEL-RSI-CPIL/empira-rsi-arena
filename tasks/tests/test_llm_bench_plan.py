"""Offline tests of the planning evaluators (Natural Plan meeting planning, TravelPlanner)."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "suite" / "_lib"))
sys.path.insert(0, str(ROOT / "tests"))

from autoresearch_suite import llm_bench_core as core  # noqa: E402
from autoresearch_suite import llm_bench_plan as plan  # noqa: E402
from test_llm_bench import DataDir, FakeLLM  # noqa: E402

ROW = {
    "id": 7, "num_people": 2, "prompt_5shot": "PROMPT",
    "constraints": [["Nob Hill", "9:00AM"], ["William", "The Castro", "12:15PM to 10:00PM", 75], ["Ann", "Nob Hill", "9:30AM to 10:00AM", 30]],
    "dist_matrix": {"Nob Hill": {"The Castro": 17}, "The Castro": {"Nob Hill": 16}},
    "golden_plan": ["You start at Nob Hill at 9:00AM.", "You wait until 9:30AM.", "You meet Ann for 30 minutes from 9:30AM to 10:00AM.",
                    "You travel to The Castro in 17 minutes and arrive at 10:17AM.", "You wait until 12:15PM.",
                    "You meet William for 75 minutes from 12:15PM to 1:30PM."],
}


class MeetingPlanningTests(unittest.TestCase):
    def score(self, text):
        return plan.meeting_score(ROW, text)

    def test_golden_plan_solves(self):
        result = self.score("Some reasoning.\nSOLUTION: " + " ".join(ROW["golden_plan"]))
        self.assertEqual((result["score"], result["meetings"], result["golden_meetings"]), (1.0, 2, 2))

    def test_fewer_meetings_fail(self):
        text = "SOLUTION: You start at Nob Hill at 9:00AM. You travel to The Castro in 17 minutes and arrive at 9:17AM. You wait until 12:15PM. You meet William for 75 minutes from 12:15PM to 1:30PM."
        self.assertEqual(self.score(text)["score"], 0.0)

    def test_invalid_steps_stop_simulation(self):
        too_early = "SOLUTION: You start at Nob Hill at 9:00AM. You meet Ann for 30 minutes from 9:00AM to 9:30AM."
        self.assertEqual(self.score(too_early)["meetings"], 0)
        unknown_place = "SOLUTION: You start at Nob Hill at 9:00AM. You travel to Mars in 5 minutes and arrive at 9:05AM. You meet Ann for 30 minutes."
        self.assertEqual(self.score(unknown_place)["meetings"], 0)
        twice = ("SOLUTION: You start at Nob Hill at 9:00AM. You wait until 9:30AM. You meet Ann for 30 minutes from 9:30AM to 10:00AM. "
                 "You meet Ann for 30 minutes from 10:00AM to 10:30AM.")
        self.assertEqual(self.score(twice)["meetings"], 1)
        self.assertEqual(self.score("")["meetings"], 0)

    def test_evaluate_uses_prompt_only(self):
        seen = []
        with DataDir({"meeting_planning_test.json": [ROW]}), mock.patch.object(core, "task_llm", return_value=FakeLLM()):
            out = plan.evaluate(lambda q, llm: seen.append(q) or "SOLUTION: " + " ".join(ROW["golden_plan"]),
                                {"benchmark": "meeting_planning", "samples": 5, "seed": 0})
        self.assertEqual((out["score"], seen), (1.0, ["PROMPT"]))


def travel_db():
    import pandas as pd
    return {
        "city_state": {"Alpha": "StateA", "Beta": "StateB"},
        "flights": pd.DataFrame([["F0000001", 100, "08:00", "10:00", "2 hours", "2022-03-01", "Alpha", "Beta", 300],
                                 ["F0000002", 120, "18:00", "20:00", "2 hours", "2022-03-03", "Beta", "Alpha", 300]],
                                columns=['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime', 'FlightDate', 'OriginCityName', 'DestCityName', 'Distance']),
        "restaurants": pd.DataFrame([[f"R{i}", 10 * i, "Chinese, Cafe", 4.0, "Beta"] for i in range(1, 7)], columns=['Name', 'Average Cost', 'Cuisines', 'Aggregate Rating', 'City']),
        "attractions": pd.DataFrame([[f"A{i}", 0.0, 0.0, "addr", "phone", "web", "Beta"] for i in range(1, 4)],
                                    columns=['Name', 'Latitude', 'Longitude', 'Address', 'Phone', 'Website', 'City']),
        "accommodations": pd.DataFrame([["H1", 100, "Entire home/apt", "No parties", 1, 2, 4.5, "Beta"]],
                                       columns=['NAME', 'price', 'room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']),
        "distance": pd.DataFrame(columns=["origin", "destination", "duration", "distance"]),
    }


QUERY = {"org": "Alpha", "dest": "Beta", "days": 3, "visiting_city_number": 1, "people_number": 1, "budget": 1000,
         "local_constraint": {"house rule": None, "cuisine": None, "room type": None, "transportation": None}}
PLAN = [
    {"days": 1, "current_city": "from Alpha to Beta", "transportation": "Flight Number: F0000001, from Alpha to Beta, Departure Time: 08:00, Arrival Time: 10:00",
     "breakfast": "-", "attraction": "A1, Beta;", "lunch": "R1, Beta", "dinner": "R2, Beta", "accommodation": "H1, Beta"},
    {"days": 2, "current_city": "Beta", "transportation": "-", "breakfast": "R3, Beta", "attraction": "A2, Beta;", "lunch": "R4, Beta",
     "dinner": "R5, Beta", "accommodation": "H1, Beta"},
    {"days": 3, "current_city": "from Beta to Alpha", "transportation": "Flight Number: F0000002, from Beta to Alpha, Departure Time: 18:00, Arrival Time: 20:00",
     "breakfast": "R6, Beta", "attraction": "A3, Beta;", "lunch": "-", "dinner": "-", "accommodation": "-"},
]


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is not installed")
class TravelPlannerTests(unittest.TestCase):
    def check(self, plan=PLAN, **changes):
        from autoresearch_suite import llm_bench_travel_checks as checks
        query = json.loads(json.dumps(QUERY))
        query.update(changes)
        return checks.final_pass(travel_db(), query, plan)

    def test_valid_plan_passes(self):
        passed, commonsense, hard = self.check()
        self.assertTrue(passed, (commonsense, hard))
        from autoresearch_suite import llm_bench_travel_checks as checks
        self.assertEqual(checks.get_total_cost(travel_db(), QUERY, PLAN), 100 + 120 + 10 * 21 + 100 * 2)

    def test_constraint_failures(self):
        self.assertFalse(self.check(budget=500)[0])
        self.assertFalse(self.check(local_constraint={"house rule": "parties", "cuisine": None, "room type": None, "transportation": None})[0])
        self.assertFalse(self.check(local_constraint={"house rule": None, "cuisine": ["Mexican"], "room type": None, "transportation": None})[0])
        repeated = json.loads(json.dumps(PLAN))
        repeated[1]["lunch"] = "R1, Beta"
        self.assertFalse(self.check(repeated)[1]["is_valid_restaurants"][0])
        unknown = json.loads(json.dumps(PLAN))
        unknown[1]["dinner"] = "Nowhere Diner, Beta"
        passed, commonsense, hard = self.check(unknown)
        self.assertEqual((passed, commonsense["is_valid_information_in_sandbox"][0], hard), (False, False, None))
        self.assertFalse(self.check(PLAN[:2])[0])

    def test_reply_parsing(self):
        from autoresearch_suite import llm_bench_travel as travel
        reply = "Here is the plan:\n```json\n" + json.dumps(PLAN) + "\n```"
        self.assertEqual(travel.parse_plan(reply), PLAN)
        self.assertIsNone(travel.parse_plan("no plan"))
        self.assertIsNone(travel.parse_plan("[1, 2, 3]"))
        with mock.patch("autoresearch_suite.llm_bench_travel.database", travel_db):
            row = dict(QUERY, query="q", reference_information=[])
            self.assertEqual(travel.score_plan(row, reply)["score"], 1.0)
            self.assertEqual(travel.score_plan(row, "sorry")["delivered"], False)


if __name__ == "__main__":
    unittest.main()
