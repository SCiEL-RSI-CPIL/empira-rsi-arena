"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import json
import re

_STATE = {}


def _query_facts(instruction, llm):
    query = instruction.split("Query:", 1)[-1].strip()
    reply = llm("Extract the trip facts of this travel request as JSON with keys origin (city), destination (city or state), "
                "dates (list of YYYY-MM-DD, one per day), cities (number of cities to visit).\n\n" + query, max_tokens=300, json_mode=True)
    facts = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    facts["dates"] = [str(d) for d in facts["dates"]]
    return facts


def _route_actions(facts, cities):
    days, dates = len(facts["dates"]), facts["dates"]
    stops = [1 + 2 * i for i in range(len(cities))] + [days]
    legs = [facts["origin"]] + cities + [facts["origin"]]
    actions = []
    for city in cities:
        actions += [f"AccommodationSearch[{city}]", f"RestaurantSearch[{city}]", f"AttractionSearch[{city}]"]
    for i in range(len(legs) - 1):
        date = dates[min(stops[i], days) - 1]
        actions += [f"FlightSearch[{legs[i]}, {legs[i + 1]}, {date}]", f"GoogleDistanceMatrix[{legs[i]}, {legs[i + 1]}, self-driving]"]
    return actions


def act(instruction, observation, valid_actions, history, llm):
    """Scripted information gathering (one search per needed item), then a single planning call with everything that was found."""
    state = _STATE.setdefault("episode", {"queue": None, "facts": None})
    if state["facts"] is None:
        state["facts"] = _query_facts(instruction, llm)
        count = int(state["facts"].get("cities") or 1)
        if count > 1:
            return "CitySearch[" + state["facts"]["destination"] + "]"
        state["queue"] = _route_actions(state["facts"], [state["facts"]["destination"]])
    if state["queue"] is None:
        cities = [c.strip(" '\"") for c in observation.strip("[]").split(",") if c.strip()]
        state["queue"] = _route_actions(state["facts"], cities[:int(state["facts"]["cities"])])
    if state["queue"]:
        return state["queue"].pop(0)
    notes = "\n\n".join(f"{action}\n{seen[:3000]}" for action, seen, _ in history)
    query = instruction.split("Query:", 1)[-1].strip()
    plan_format = instruction[instruction.find("Write the plan as a JSON list"):instruction.find("\n\nQuery:")]
    system = "You are a travel planner. The searches are finished and no tools are available. Reply with the JSON plan only."
    request = ("Query: " + query + "\n\nCollected information:\n" + notes + "\n\n" + plan_format +
               "\nIf some information is missing, still write the most complete plan the collected information allows.")
    plan = llm([{"role": "system", "content": system}, {"role": "user", "content": request}], max_tokens=8192)
    match = re.search(r"\[.*\]", plan, flags=re.S)
    return "Finish[" + (match.group(0) if match else "[]") + "]"
# EVOLVE-BLOCK-END
