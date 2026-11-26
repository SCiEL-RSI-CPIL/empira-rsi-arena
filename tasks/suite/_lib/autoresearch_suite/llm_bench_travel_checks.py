"""TravelPlanner constraint checks, ported from the benchmark's evaluation code.

Adapted from OSU-NLP-Group/TravelPlanner (MIT) at commit
e52c87f4ac348a3410c46dc3553c519db5ec5e23: evaluation/commonsense_constraint.py,
evaluation/hard_constraint.py and the helpers of utils/func.py and
tools/googleDistanceMatrix/apis.py. The logic is unchanged; the database
tables arrive through ``db`` (a mapping of pandas DataFrames plus the
city-to-state map) instead of module-level globals.
"""
from __future__ import annotations

import math
import re


def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s


def get_valid_name_city(info):
    match = re.search(r'(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$', info)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()
    return "-", "-"


def extract_from_to(text):
    matches = re.search(r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)", text)
    return matches.groups() if matches else (None, None)


def count_consecutive_values(lst):
    if not lst:
        return []
    result, current, count = [], lst[0], 1
    for value in lst[1:]:
        if value == current:
            count += 1
        else:
            result.append((current, count))
            current, count = value, 1
    result.append((current, count))
    return result


def distance_cost(db, origin, destination, mode):
    """GoogleDistanceMatrix.run_for_evaluation: cost or None."""
    origin, destination = extract_before_parenthesis(origin), extract_before_parenthesis(destination)
    data = db["distance"]
    response = data[(data['origin'] == origin) & (data['destination'] == destination)]
    if len(response) > 0:
        duration, distance = response['duration'].values[0], response['distance'].values[0]
        if duration is None or distance is None or (isinstance(duration, float) and math.isnan(duration)) or (isinstance(distance, float) and math.isnan(distance)):
            return None
        if 'day' not in duration:
            if 'driving' in mode:
                return int(eval(distance.replace("km", "").replace(",", "")) * 0.05)  # noqa: S307 - numeric database field, as upstream
            if mode == "taxi":
                return int(eval(distance.replace("km", "").replace(",", "")))  # noqa: S307
    return None


def _match(table, name_column, city_column, name, city):
    return table[(table[name_column].astype(str).str.contains(re.escape(name))) & (table[city_column] == city)]


def _days(question, tested):
    return range(min(question['days'], len(tested)))


def is_valid_city_sequence(city_list):
    if len(city_list) < 3:
        return False
    visited, i = set(), 0
    while i < len(city_list):
        city = city_list[i]
        if city in visited and (i != 0 and i != len(city_list) - 1):
            return False
        count = 0
        while i < len(city_list) and city_list[i] == city:
            count += 1
            i += 1
        if count == 1 and 0 < i - 1 < len(city_list) - 1:
            return False
        visited.add(city)
    return True


def is_reasonable_visiting_city(db, question, tested):
    city_list = []
    for i in _days(question, tested):
        value = tested[i]['current_city']
        if 'from' in value:
            city1, city2 = extract_from_to(value)
            city1, city2 = extract_before_parenthesis(city1), extract_before_parenthesis(city2)
            if i == 0 and city1 != question['org']:
                return False, f"The first day's city should be {question['org']}."
            city_list += [city1, city2]
        else:
            city_list.append(extract_before_parenthesis(value))
    if city_list[0] != city_list[-1]:
        return False, "The trip should be a closed circle."
    if not is_valid_city_sequence(city_list):
        return False, "The city sequence is invalid."
    for idx, city in enumerate(city_list):
        if city not in db["city_state"]:
            return False, f"{city} is not a valid city."
        if idx not in [0, len(city_list) - 1] and question['days'] > 3 and db["city_state"][city] != question['dest']:
            return False, f"{city} is not in {question['dest']}."
    return True, None


def is_valid_restaurants(db, question, tested):
    seen = []
    for i in _days(question, tested):
        for meal in ('breakfast', 'lunch', 'dinner'):
            value = tested[i].get(meal)
            if value and value != '-':
                if value in seen:
                    return False, f"The restaurant in day {i + 1} {meal} is repeated."
                seen.append(value)
    return True, None


def is_valid_attractions(db, question, tested):
    seen = []
    for i in _days(question, tested):
        value = tested[i].get('attraction')
        if value and value != '-':
            for attraction in value.split(';')[:-1]:
                if attraction in seen:
                    return False, f"The attraction '{attraction}' in day {i + 1} is repeated."
                seen.append(attraction)
    return True, None


def transportation_match(text):
    if 'taxi' in text.lower():
        return 'Taxi'
    if 'self-driving' in text.lower():
        return 'Self-driving'
    if 'flight' in text.lower():
        return 'Flight'
    return None


def is_valid_transportation(db, question, tested):
    if tested[0]['transportation'] and tested[0]['transportation'] != '-':
        modes = [transportation_match(tested[0]['transportation'])]
    else:
        return False, "The transportation in day 1 should not be empty."
    for i in _days(question, tested):
        value = tested[i].get('transportation')
        if value and value != '-':
            modes.append(transportation_match(value))
    if ('Self-driving' in modes and 'Flight' in modes) or ('Taxi' in modes and 'Self-driving' in modes):
        return False, "The transportation is conflicting."
    return True, None


def is_valid_information_in_current_city(db, question, tested):
    for i in _days(question, tested):
        unit = tested[i]
        current = unit['current_city']
        if 'from' in current:
            city1, city2 = extract_from_to(current)
            cities = [extract_before_parenthesis(city1), extract_before_parenthesis(city2)]
        else:
            cities = [extract_before_parenthesis(current)]
        value = unit.get('transportation')
        if value and value != '-':
            for city in cities:
                if city not in value:
                    return False, f"The transportation in day {i + 1} is invalid city choice."
        for meal in ('breakfast', 'lunch', 'dinner'):
            value = unit.get(meal)
            if value and value != '-' and not any(city in value for city in cities):
                return False, f"The {meal} in day {i + 1} is invalid city choice."
        value = unit.get('attraction')
        if value and value != '-':
            for attraction in value.split(';')[:-1]:
                if not any(city in attraction for city in cities):
                    return False, f"The attraction in day {i + 1} is invalid city choice."
        value = unit.get('accommodation')
        if value and value != '-' and cities[-1] not in value:
            return False, f"The accommodation in day {i + 1} is invalid city choice."
    return True, None


def is_valid_information_in_sandbox(db, question, tested):
    flights, restaurants, attractions, accommodations = db["flights"], db["restaurants"], db["attractions"], db["accommodations"]
    for i in _days(question, tested):
        unit = tested[i]
        value = unit['transportation']
        if value and value != '-':
            org_city, dest_city = extract_from_to(value)
            if org_city is None or dest_city is None:
                org_city, dest_city = extract_from_to(unit['current_city'])
            if 'flight number' in value.lower():
                try:
                    org_city, dest_city = extract_before_parenthesis(org_city), extract_before_parenthesis(dest_city)
                except TypeError:
                    raise ValueError(f"The transportation {value} in day {i + 1} can not be parsed.") from None
                number = value.split('Flight Number: ')[1].split(',')[0]
                if len(flights[(flights['Flight Number'] == number) & (flights['OriginCityName'] == org_city) & (flights['DestCityName'] == dest_city)]) < 1:
                    return False, f"The flight number in day {i + 1} is invalid in the sandbox."
            elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                try:
                    org_city, dest_city = extract_before_parenthesis(org_city), extract_before_parenthesis(dest_city)
                except TypeError:
                    org_city, dest_city = '-', '-'
                mode = 'self-driving' if 'self-driving' in value.lower() else 'taxi'
                if distance_cost(db, org_city, dest_city, mode) is None:
                    return False, f"The {mode} in day {i + 1} is invalid in the sandbox."
        for meal in ('breakfast', 'lunch', 'dinner'):
            value = unit.get(meal)
            if value and value != '-':
                name, city = get_valid_name_city(value)
                if len(_match(restaurants, 'Name', 'City', name, city)) < 1:
                    return False, f"The {meal} in day {i + 1} is invalid in the sandbox."
        value = unit.get('attraction')
        if value and value != '-':
            for attraction in value.split(';')[:-1]:
                name, city = get_valid_name_city(attraction)
                if len(_match(attractions, 'Name', 'City', name, city)) < 1:
                    return False, f"The attraction {attraction} in day {i + 1} is invalid in the sandbox."
        value = unit.get('accommodation')
        if value and value != '-':
            name, city = get_valid_name_city(value)
            if len(_match(accommodations, 'NAME', 'city', name, city)) < 1:
                return False, f"The accommodation in day {i + 1} is invalid in the sandbox."
    return True, None


def is_valid_accommodation(db, question, tested):
    data = []
    for i in _days(question, tested):
        if 'accommodation' not in tested[i]:
            return False, "No Accommodation Info."
        data.append(tested[i]['accommodation'])
    for stay in count_consecutive_values(data):
        if stay and stay[0] not in ['-', '']:
            name, city = get_valid_name_city(stay[0])
            found = _match(db["accommodations"], 'NAME', 'city', name, city)
            if len(found) == 1 and stay[1] < found.iloc[0]['minimum nights']:
                return False, f"The accommodation {stay[0]} do not obey the minumum nights rule."
    return True, None


def is_valid_visiting_city_number(question, tested):
    cities = set()
    for i in _days(question, tested):
        value = tested[i]['current_city']
        if 'from' in value:
            city1, city2 = extract_from_to(value)
            city1, city2 = extract_before_parenthesis(city1), extract_before_parenthesis(city2)
            if i == 0 and city1 != question['org']:
                return False, f"The first day's city should be {question['org']}."
            cities.update([city1, city2])
        else:
            cities.add(extract_before_parenthesis(value))
    cities.discard(question['org'])
    if len(cities) != question['visiting_city_number']:
        return False, f"The number of visiting cities should be {question['visiting_city_number']}."
    return True, None


def is_valid_days(question, tested):
    count = sum(1 for i in _days(question, tested)
                if tested[i] != {} and tested[i]['current_city'] != "You don't need to fill in the information for this or later days.")
    return (True, None) if count == question['days'] else (False, f"The number of days should be {question['days']}.")


def is_not_absent(db, question, tested):
    needed, present = 6 * question['days'], 0
    if not is_valid_days(question, tested)[0]:
        return False, "Invalid Days"
    if not is_valid_visiting_city_number(question, tested)[0]:
        return False, "Invalid City Number"
    for i in _days(question, tested):
        unit = tested[i]
        for key in ('transportation', 'breakfast', 'lunch', 'dinner', 'attraction', 'accommodation'):
            if key not in unit:
                return False, f"No {key.capitalize()} Info."
        if ('from ' in unit['current_city'] or 'to ' in unit['current_city']) and unit['transportation'] in ['', '-']:
            return False, f"No transportation in day {i + 1} is not allowed."
        if ('from ' not in unit['current_city'] and ' to ' not in unit['current_city']) and unit['attraction'] in ['', '-']:
            return False, f"No attaction in day {i + 1} is not allowed."
        if i != question['days'] - 1 and unit['accommodation'] in ['', '-']:
            return False, f"No accommodation in day {i + 1} is not allowed."
        if (unit['breakfast'] in ['', '-'] or unit['lunch'] in ['', '-'] or unit['dinner'] in ['', '-']) and 'from ' not in unit['current_city']:
            return False, f"No meal in day {i + 1} is not allowed."
        present += sum(1 for key in unit if unit[key] and unit[key] != '-')
    if present * 1.0 / needed < 0.5:
        return False, "The absent information is more than 50%."
    return True, None


COMMONSENSE = ("is_reasonable_visiting_city", "is_valid_restaurants", "is_valid_attractions", "is_valid_accommodation", "is_valid_transportation",
               "is_valid_information_in_current_city", "is_valid_information_in_sandbox", "is_not_absent")


# --- hard constraints (evaluation/hard_constraint.py) ---------------------------------------------

def get_total_cost(db, question, tested):
    total = 0
    people = question['people_number']
    for i in _days(question, tested):
        unit = tested[i]
        value = unit['transportation']
        if value and value != '-':
            org_city, dest_city = extract_from_to(value)
            if org_city is None or dest_city is None:
                org_city, dest_city = extract_from_to(unit['current_city'])
            if org_city is not None and dest_city is not None:
                if 'flight number' in value.lower():
                    found = db["flights"][db["flights"]['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]]
                    if len(found) > 0:
                        total += found['Price'].values[0] * people
                elif 'self-driving' in value.lower():
                    total += distance_cost(db, org_city, dest_city, 'self-driving') * math.ceil(people * 1.0 / 5)
                elif 'taxi' in value.lower():
                    total += distance_cost(db, org_city, dest_city, 'taxi') * math.ceil(people * 1.0 / 4)
        for meal in ('breakfast', 'lunch', 'dinner'):
            if unit[meal] and unit[meal] != '-':
                name, city = get_valid_name_city(unit[meal])
                found = _match(db["restaurants"], 'Name', 'City', name, city)
                if len(found) > 0:
                    total += found['Average Cost'].values[0] * people
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            found = _match(db["accommodations"], 'NAME', 'city', name, city)
            if len(found) > 0:
                total += found['price'].values[0] * math.ceil(people * 1.0 / found['maximum occupancy'].values[0])
    return total


RULES = {"smoking": "No smoking", "parties": "No parties", "children under 10": "No children under 10", "visitors": "No visitors", "pets": "No pets"}
ROOMS = {"shared room": "Shared room", "private room": "Private room", "entire room": "Entire home/apt"}


def _stays(db, question, tested):
    for i in _days(question, tested):
        value = tested[i]['accommodation']
        if value and value != '-':
            name, city = get_valid_name_city(value)
            found = _match(db["accommodations"], 'NAME', 'city', name, city)
            if len(found) > 0:
                yield found


def is_valid_room_rule(db, question, tested):
    rule = question['local_constraint']['house rule']
    if rule is None:
        return None, None
    for found in _stays(db, question, tested):
        if rule in RULES and RULES[rule] in str(found['house_rules'].values[0]):
            return False, f"The house rule should be {rule}."
    return True, None


def is_valid_cuisine(db, question, tested):
    wanted = question['local_constraint']['cuisine']
    if not wanted:
        return None, None
    found_cuisines = set()
    for i in _days(question, tested):
        for meal in ('breakfast', 'lunch', 'dinner'):  # upstream skips the rest of a day after a meal in the origin city
            value = tested[i][meal]
            if value and value != '-':
                name, city = get_valid_name_city(value)
                if city == question['org']:
                    break
                found = _match(db["restaurants"], 'Name', 'City', name, city)
                if len(found) > 0:
                    found_cuisines.update(cuisine for cuisine in wanted if cuisine in found.iloc[0]['Cuisines'])
    for cuisine in wanted:
        if cuisine not in found_cuisines:
            return False, f"The cuisine {cuisine} is not satisfied."
    return True, None


def is_valid_hard_transportation(db, question, tested):
    rule = question['local_constraint']['transportation']
    if rule is None:
        return None, None
    for i in _days(question, tested):
        value = tested[i]['transportation']
        if value and value != '-':
            if (rule == 'no flight' and 'Flight' in value) or (rule == 'no self-driving' and 'Self-driving' in value):
                return False, f"The transportation should not be {rule}."
    return True, None


def is_valid_room_type(db, question, tested):
    rule = question['local_constraint']['room type']
    if rule is None:
        return None, None
    for found in _stays(db, question, tested):
        room = found['room type'].values[0]
        if (rule == 'not shared room' and room == 'Shared room') or (rule in ROOMS and room != ROOMS[rule]):
            return False, f"The room type should be {rule}."
    return True, None


def _safe(check, *args):
    try:
        return check(*args)
    except Exception as error:  # noqa: BLE001 - upstream would crash on malformed plans; count the check as failed
        return False, f"check raised {type(error).__name__}"


def final_pass(db, question, tested):
    """eval.py's final pass: every commonsense and every hard constraint holds (None means not applicable)."""
    funcs = globals()
    commonsense = {name: _safe(funcs[name], db, question, tested) for name in COMMONSENSE}
    if not (commonsense['is_not_absent'][0] and commonsense['is_valid_information_in_sandbox'][0]):
        return False, commonsense, None
    hard = {"valid_cuisine": _safe(is_valid_cuisine, db, question, tested), "valid_room_rule": _safe(is_valid_room_rule, db, question, tested),
            "valid_transportation": _safe(is_valid_hard_transportation, db, question, tested), "valid_room_type": _safe(is_valid_room_type, db, question, tested),
            "valid_cost": _safe(lambda d, q, t: (bool(get_total_cost(d, q, t) <= q['budget']), None), db, question, tested)}
    passed = all(result[0] is not False for result in commonsense.values()) and all(result[0] is not False for result in hard.values())
    return passed, commonsense, hard
