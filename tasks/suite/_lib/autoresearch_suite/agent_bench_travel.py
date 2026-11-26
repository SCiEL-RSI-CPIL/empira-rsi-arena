"""TravelPlanner in two-stage (tool-use) mode, the setting AgentSquare evaluates.

The agent must gather the information itself with TravelPlanner's search tools
and then submit a plan; no reference information is given. Tools follow the
benchmark's ``tools/*/apis.py`` and the argument checks of
``agents/tool_agents.py`` (OSU-NLP-Group/TravelPlanner, MIT) on the benchmark
database: tables are read as the tool classes read them and results are printed
with ``DataFrame.to_string(index=False)``. The plan is checked with the suite's
port of TravelPlanner's commonsense and hard constraints
(``llm_bench_travel_checks``, shared with the Mind Evolution sole-planning task).
Episode score: the commonsense-constraint micro pass rate of the plan (passed
commonsense constraints / 8; 0 when no plan is submitted); the hard-constraint
micro pass rate and the final pass are reported in the details.
"""
from __future__ import annotations

import hashlib
import json
import re

from arena_task_paths import data_root
from . import llm_bench_core
from . import llm_bench_travel as travel
from . import llm_bench_travel_checks as checks

TOOLS = ["FlightSearch[Departure City, Destination City, Date]", "GoogleDistanceMatrix[Origin, Destination, Mode]",
         "AccommodationSearch[City]", "RestaurantSearch[City]", "AttractionSearch[City]", "CitySearch[State]",
         "NotebookWrite[Short Description]", "Finish[Plan JSON]"]
INSTRUCTION = """You are a travel-planning agent. Answer the user's query below by first collecting information with the tools and then submitting a complete plan. Reply with exactly one action per turn, written as Tool[arguments]:
(1) FlightSearch[Departure City, Destination City, Date]: flights between two cities on a date (YYYY-MM-DD).
(2) GoogleDistanceMatrix[Origin, Destination, Mode]: distance, duration and cost of travelling by Mode, which is self-driving or taxi.
(3) AccommodationSearch[City]: accommodations in a city.
(4) RestaurantSearch[City]: restaurants in a city.
(5) AttractionSearch[City]: attractions in a city.
(6) CitySearch[State]: the cities of a state.
(7) NotebookWrite[Short Description]: record the result of the previous search under a short description.
(8) Finish[Plan JSON]: submit the final plan and end the episode.
Only information returned by the tools may appear in the plan. """ + travel.FORMAT


def data_dir():
    return data_root() / "agent_bench"


def database():
    """Benchmark tables as the tool classes read them (selected columns, rows with gaps dropped).

    Flights come from this package's subset (the validation queries' cities and dates); the other tables are the
    llm_bench package's prepared copies of the same benchmark database.
    """
    import pandas as pd
    path = data_dir() / "travel" / "flights.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run python scripts/prepare_agent_bench.py travel")
    states = {}
    for line in llm_bench_core.data_path("travel_city_states.txt").read_text(encoding="utf-8").split("\n"):
        if "\t" in line:
            city, state = line.split("\t")
            states[city] = state
    return {"flights": pd.read_csv(path).dropna()[['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime', 'FlightDate',
                                                   'OriginCityName', 'DestCityName', 'Distance']],
            "accommodations": travel._table("travel_accommodations.csv", ['NAME', 'price', 'room type', 'house_rules', 'minimum nights',
                                                                          'maximum occupancy', 'review rate number', 'city']),
            "restaurants": travel._table("travel_restaurants.csv", ['Name', 'Average Cost', 'Cuisines', 'Aggregate Rating', 'City']),
            "attractions": travel._table("travel_attractions.csv", ['Name', 'Latitude', 'Longitude', 'Address', 'Phone', 'Website', 'City']),
            "distance": travel._table("travel_distance.csv", dropna=False), "city_state": states}


def parse_action(text):
    """TravelPlanner's parse_action (``^(\\w+)\\[(.+)\\]$``) on the stripped reply; the plan of Finish may span lines."""
    match = re.match(r"^(\w+)\[(.+)\]$", text.strip(), flags=re.S)
    return (match.group(1), match.group(2)) if match else (None, None)


def show(data):
    return data.to_string(index=False) if hasattr(data, "to_string") else str(data)


class Adapter:
    def __init__(self, params):
        self.params = params
        self.db = database()
        self.rows = {row["id"]: row for row in llm_bench_core.read_json("travelplanner_validation.json")}
        self.cities = set(self.db["city_state"])
        self.states = {}
        for city, state in self.db["city_state"].items():
            self.states.setdefault(state, []).append(city)

    def episodes(self):
        first, last, seed = int(self.params.get("first", 151)), int(self.params.get("last", 180)), self.params.get("seed", 0)
        ids = sorted(range(first, last + 1), key=lambda i: hashlib.sha256(f"{seed}|travel-{i}".encode()).hexdigest())
        return [{"id": f"travel-{i}", "query": i} for i in ids[:int(self.params.get("samples", 10))]]

    def reset(self, spec):
        self.row = self.rows[spec["query"]]
        self.plan, self.last, self.notebook = None, None, []
        return {"instruction": INSTRUCTION + "\n\nQuery: " + self.row["query"], "observation": "", "valid_actions": list(TOOLS), "done": False}

    def step(self, reply):
        kind, arg = parse_action(reply)
        if kind == "Finish":
            self.plan = travel.parse_plan(arg)
            observation = "Plan received." if self.plan is not None else "Plan received but it is not a JSON list of day records."
            return {"observation": observation, "reward": 0.0, "done": True, "valid_actions": []}
        return {"observation": self.tool(kind, arg), "reward": 0.0, "done": False, "valid_actions": list(TOOLS)}

    def tool(self, kind, arg):
        db = self.db
        try:
            if kind == "FlightSearch":
                origin, destination, date = arg.split(", ")[:3]
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                    return f"'{date}' is not in the format YYYY-MM-DD"
                for city in (origin, destination):
                    if city not in self.cities:
                        return f"{city} is not a valid city."
                table = db["flights"]
                found = table[(table["OriginCityName"] == origin) & (table["DestCityName"] == destination) & (table["FlightDate"] == date)]
                self.last = found if len(found) else f"There is no flight from {origin} to {destination} on {date}."
            elif kind in ("AccommodationSearch", "RestaurantSearch", "AttractionSearch"):
                name, column, empty = {"AccommodationSearch": ("accommodations", "city", "There is no attraction in this city."),
                                       "RestaurantSearch": ("restaurants", "City", "There is no restaurant in this city."),
                                       "AttractionSearch": ("attractions", "City", "There is no attraction in this city.")}[kind]
                if arg not in self.cities:
                    return f"{arg} is not a valid city."
                found = db[name][db[name][column] == arg].reset_index(drop=True)
                self.last = found if len(found) else empty
            elif kind == "GoogleDistanceMatrix":
                origin, destination, mode = arg.split(", ")[:3]
                self.last = self.distance(checks.extract_before_parenthesis(origin), checks.extract_before_parenthesis(destination), mode)
            elif kind == "CitySearch":
                if arg == "Planner":
                    return "Planner is not a state."
                self.last = self.states.get(arg, "Invalid State")
            elif kind == "NotebookWrite":
                self.notebook.append({"Short Description": arg, "Content": show(self.last)})
                return f"The information has been recorded in Notebook, and its index is {len(self.notebook) - 1}."
            else:
                return ("Invalid Action. Valid Actions are FlightSearch[Departure City, Destination City, Date] / AccommodationSearch[City] / "
                        "RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / "
                        "GoogleDistanceMatrix[Origin, Destination, Mode] and Finish[Plan JSON].")
        except ValueError:
            return f"Illegal {kind}. Please try again."
        return show(self.last).strip()

    def distance(self, origin, destination, mode):
        table = self.db["distance"]
        found = table[(table["origin"] == origin) & (table["destination"] == destination)]
        if len(found) == 0:
            return f"{mode}, from {origin} to {destination}, no valid information."
        duration, distance = found["duration"].values[0], found["distance"].values[0]
        if not isinstance(duration, str) or not isinstance(distance, str) or "day" in duration:
            return "No valid information."
        km = float(distance.replace("km", "").replace(",", ""))
        cost = int(km * 0.05) if "driving" in mode else (int(km) if mode == "taxi" else None)
        return f"{mode}, from {origin} to {destination}, duration: {duration}, distance: {distance}, cost: {cost}"

    def score(self):
        if self.plan is None:
            return {"score": 0.0, "details": {"success": False, "delivered": False, "level": self.row["level"], "days": self.row["days"]}}
        passed, commonsense, hard = checks.final_pass(self.db, self.row, self.plan)
        common = sum(1 for result in commonsense.values() if result[0] is True)
        details = {"success": bool(passed), "delivered": True, "commonsense_passed": common, "level": self.row["level"], "days": self.row["days"],
                   "hard_passed": sum(1 for r in (hard or {}).values() if r[0] is True),
                   "hard_applicable": sum(1 for r in (hard or {}).values() if r[0] is not None),
                   "failed": sorted(name for name, r in list(commonsense.items()) + list((hard or {}).items()) if r[0] is False)[:8]}
        return {"score": common / len(checks.COMMONSENSE), "details": json.loads(json.dumps(details))}
