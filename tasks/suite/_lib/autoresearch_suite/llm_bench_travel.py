"""TravelPlanner in sole-planning mode (Mind Evolution's TravelPlanner task).

The candidate receives the user query and the query's reference information
(TravelPlanner's sole-planning input) and returns the plan as a JSON list of
day records with TravelPlanner's keys. The plan is checked with the
benchmark's commonsense and hard constraints (``llm_bench_travel_checks``, a
port of the official evaluation) against the benchmark database; an item is
solved when it passes every constraint (the Final Pass Rate). The flight table
is restricted to the routes and dates of the validation queries' reference
information, which contains every flight a sole-planning plan can cite.
"""
from __future__ import annotations

import csv
import io

from . import llm_bench_core as core
from . import llm_bench_travel_checks as checks

_DB = None
FORMAT = ("Write the plan as a JSON list with one object per day and exactly these keys: \"days\" (1, 2, ...), \"current_city\", \"transportation\", "
          "\"breakfast\", \"attraction\", \"lunch\", \"dinner\", \"accommodation\". current_city is the city of that day, or \"from A to B\" on a day of "
          "travel. transportation is \"Flight Number: <number>, from A to B, Departure Time: <hh:mm>, Arrival Time: <hh:mm>\", \"Self-driving, from A to B, "
          "...\", \"Taxi, from A to B, ...\" or \"-\". Each meal and the accommodation is \"<name>, <city>\" or \"-\". attraction lists \"<name>, <city>;\" "
          "entries, each ending with \";\", or is \"-\". Use \"-\" where nothing is needed (for example after returning home), take every flight, "
          "restaurant, attraction and accommodation from the reference information, and keep the whole trip within the budget and the query's constraints.")


def _table(name, columns=None, dropna=True):
    import pandas as pd
    frame = pd.read_csv(io.StringIO(core.data_path(name).read_text(encoding="utf-8")))
    if dropna:
        frame = frame.dropna()
    return frame[columns] if columns else frame


def database():
    """The benchmark tables, read as the upstream tool classes read them (selected columns, rows with gaps dropped)."""
    global _DB
    if _DB is None:
        states = {}
        for line in core.data_path("travel_city_states.txt").read_text(encoding="utf-8").split("\n"):
            if "\t" in line:
                city, state = line.split("\t")
                states[city] = state
        _DB = {"flights": _table("travel_flights.csv", ['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime', 'FlightDate', 'OriginCityName', 'DestCityName', 'Distance']),
               "accommodations": _table("travel_accommodations.csv", ['NAME', 'price', 'room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']),
               "restaurants": _table("travel_restaurants.csv", ['Name', 'Average Cost', 'Cuisines', 'Aggregate Rating', 'City']),
               "attractions": _table("travel_attractions.csv", ['Name', 'Latitude', 'Longitude', 'Address', 'Phone', 'Website', 'City']),
               "distance": _table("travel_distance.csv", dropna=False), "city_state": states}
    return _DB


def question_text(row):
    blocks = "\n\n".join(f"=== {unit['Description']} ===\n{unit['Content']}" for unit in row["reference_information"])
    return f"{row['query']}\n\nReference information (the only data the plan may use):\n\n{blocks}\n\n{FORMAT}"


def parse_plan(reply):
    """The JSON list of day records that ends last in the reply, or None."""
    plan = core.last_json(reply, kind=list)
    if not plan or not all(isinstance(day, dict) for day in plan):
        return None
    return [{key: value if isinstance(value, (str, int)) or value is None else str(value) for key, value in day.items()} for day in plan]


def score_plan(row, reply):
    plan = parse_plan(reply)
    if plan is None:
        return {"score": 0.0, "delivered": False}
    passed, commonsense, hard = checks.final_pass(database(), row, plan)
    failed = sorted(name for name, result in list(commonsense.items()) + list((hard or {}).items()) if result[0] is False)
    return {"score": float(passed), "delivered": True, "failed": failed[:6]}


def items(P):
    rows = core.read_json("travelplanner_validation.json")
    out = [{"id": f"travel-{row['id']}", "question": question_text(row), "row": row} for row in rows]
    return out, lambda item, reply: score_plan(item["row"], reply)


def parse_validation_csv(text):
    """Rows of TravelPlanner's validation.csv with typed fields (used by the prepare step)."""
    import ast
    out = []
    for index, row in enumerate(csv.DictReader(io.StringIO(text))):
        out.append({"id": index + 1, "org": row["org"], "dest": row["dest"], "days": int(row["days"]), "visiting_city_number": int(row["visiting_city_number"]),
                    "date": ast.literal_eval(row["date"]), "people_number": int(row["people_number"]), "local_constraint": ast.literal_eval(row["local_constraint"]),
                    "budget": int(row["budget"]), "query": row["query"], "level": row["level"], "reference_information": ast.literal_eval(row["reference_information"])})
    return out
