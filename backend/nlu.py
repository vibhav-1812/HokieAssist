"""
Transit and campus NLU for HokieAssist: place aliases, regex/heuristic intents, optional OpenAI fallback.

The CAMPUS_PLACES table and BT route coverage are maintained by the team. Some regex scaffolding was
drafted with Cursor and then extended and tested against real VT queries; see docs/AI_USAGE_LOG.md.
"""
import os
import re
import json
from typing import Dict, Optional
# PM5 refactor: named constants for intent values
INTENT_TRANSIT_ROUTE = "transit_route"
INTENT_NEXT_BUS = "next_bus"
INTENT_UNKNOWN = "unknown"
try:
    from openai import OpenAI
    import requests
except Exception:
    OpenAI = None  # type: ignore
    requests = None  # type: ignore

CAMPUS_PLACES: Dict[str, str] = {
    # Major Campus Buildings
    "goodwin hall": "635 Prices Fork Rd, Blacksburg, VA 24061",
    "lavery hall": "460 Old Turner St, Blacksburg, VA 24060",
    "squires": "Squires Student Center, Blacksburg, VA 24061",
    "torgersen hall": "Torgersen Hall, Blacksburg, VA 24061",
    "torgersen": "Torgersen Hall, Blacksburg, VA 24061",
    "mcbryde hall": "McBryde Hall, Blacksburg, VA 24061",
    "mcbryde": "McBryde Hall, Blacksburg, VA 24061",
    "norris hall": "Norris Hall, Blacksburg, VA 24061",
    "norris": "Norris Hall, Blacksburg, VA 24061",
    "randolph hall": "Randolph Hall, Blacksburg, VA 24061",
    "randolph": "Randolph Hall, Blacksburg, VA 24061",
    "newman library": "Newman Library, Blacksburg, VA 24061",
    "newman": "Newman Library, Blacksburg, VA 24061",
    "dietrick hall": "Dietrick Hall, Blacksburg, VA 24061",
    "dietrick": "Dietrick Hall, Blacksburg, VA 24061",
    "west eggleston": "West Eggleston, Blacksburg, VA 24061",
    "west egg": "West Eggleston, Blacksburg, VA 24061",
    "east eggleston": "East Eggleston, Blacksburg, VA 24061",
    "east egg": "East Eggleston, Blacksburg, VA 24061",
    
    # Dining Halls
    "d2": "D2 Dining Hall, Blacksburg, VA 24061",
    "d2 dining": "D2 Dining Hall, Blacksburg, VA 24061",
    "owens food court": "Owens Food Court, Blacksburg, VA 24061",
    "owens": "Owens Food Court, Blacksburg, VA 24061",
    "hokie grill": "Hokie Grill, Blacksburg, VA 24061",
    "turner place": "Turner Place, Blacksburg, VA 24061",
    "turner": "Turner Place, Blacksburg, VA 24061",
    
    # Residential Halls
    "barringer hall": "Barringer Hall, Blacksburg, VA 24061",
    "barringer": "Barringer Hall, Blacksburg, VA 24061",
    "hoge hall": "Hoge Hall, Blacksburg, VA 24061",
    "hoge": "Hoge Hall, Blacksburg, VA 24061",
    "johnson hall": "Johnson Hall, Blacksburg, VA 24061",
    "johnson": "Johnson Hall, Blacksburg, VA 24061",
    "lee hall": "Lee Hall, Blacksburg, VA 24061",
    "lee": "Lee Hall, Blacksburg, VA 24061",
    "miles hall": "Miles Hall, Blacksburg, VA 24061",
    "miles": "Miles Hall, Blacksburg, VA 24061",
    "pridemore hall": "Pridemore Hall, Blacksburg, VA 24061",
    "pridemore": "Pridemore Hall, Blacksburg, VA 24061",
    "slusher hall": "Slusher Hall, Blacksburg, VA 24061",
    "slusher": "Slusher Hall, Blacksburg, VA 24061",
    "vawter hall": "Vawter Hall, Blacksburg, VA 24061",
    "vawter": "Vawter Hall, Blacksburg, VA 24061",
    
    # Off-Campus Areas
    "main street": "Main Street, Blacksburg, VA 24060",
    "main st": "Main Street, Blacksburg, VA 24060",
    "progress street": "Progress Street, Blacksburg, VA 24060",
    "progress st": "Progress Street, Blacksburg, VA 24060",
    "university city boulevard": "University City Blvd, Blacksburg, VA 24060",
    "university city": "University City Blvd, Blacksburg, VA 24060",
    "ucb": "University City Blvd, Blacksburg, VA 24060",
    "toms creek": "Toms Creek, Blacksburg, VA 24060",
    "hethwood": "Hethwood, Blacksburg, VA 24060",
    "harding avenue": "Harding Avenue, Blacksburg, VA 24060",
    "harding": "Harding Avenue, Blacksburg, VA 24060",
    "patrick henry drive": "Patrick Henry Drive, Blacksburg, VA 24060",
    "patrick henry": "Patrick Henry Drive, Blacksburg, VA 24060",
    "corporate research center": "Corporate Research Center, Blacksburg, VA 24060",
    "crc": "Corporate Research Center, Blacksburg, VA 24060",
    "downtown": "Downtown Blacksburg, VA 24060",
    "downtown blacksburg": "Downtown Blacksburg, VA 24060",
    
    # Sports Facilities
    "lane stadium": "Lane Stadium, Blacksburg, VA 24061",
    "cassell coliseum": "Cassell Coliseum, Blacksburg, VA 24061",
    "cassell": "Cassell Coliseum, Blacksburg, VA 24061",
    "english field": "English Field, Blacksburg, VA 24061",
    "aquatic center": "Aquatic Center, Blacksburg, VA 24061",
    "mccomas hall": "McComas Hall, Blacksburg, VA 24061",
    "mccomas": "McComas Hall, Blacksburg, VA 24061",
    
    # Parking Areas
    "perry street": "Perry Street, Blacksburg, VA 24061",
    "perry st": "Perry Street, Blacksburg, VA 24061",
    "squires parking": "Squires Parking Lot, Blacksburg, VA 24061",
    "goodwin parking": "Goodwin Hall Parking, Blacksburg, VA 24061",
    
    # Common Campus References
    "campus": "Virginia Tech, Blacksburg, VA 24061",
    "vt": "Virginia Tech, Blacksburg, VA 24061",
    "virginia tech": "Virginia Tech, Blacksburg, VA 24061",
    "tech": "Virginia Tech, Blacksburg, VA 24061",
}

def normalize_place(name: str) -> str:
    if not name:
        return name
    key = name.strip().lower()
    return CAMPUS_PLACES.get(key, name)

def simple_parse(query: str) -> Dict[str, Optional[str]]:
    q = query.lower()
    intent = "generic"
    dest = None
    orig = None
    bus_route = None
    
    # Enhanced patterns for better coverage
    
    # 1. Bus schedule queries with specific routes
    bus_schedule_patterns = [
        r"when.*(next|soonest).*(cas|bt|transit|bus|crc|hdg|hxp|nmp|sme|tcp|ttt|ucb).*(?:to|going|coming)",  # "when is next CAS bus"
        r"(next|when).*(cas|bt|transit|bus|crc|hdg|hxp|nmp|sme|tcp|ttt|ucb).*(come|coming|arrive|schedule)",  # "when does CAS bus come"
        r"i.*am.*at\s+(?P<orig>[\w\s,]+).*when.*(cas|bt|transit|bus|crc|hdg|hxp|nmp|sme|tcp|ttt|ucb)",  # "I am at X when is CAS bus"
        r"when.*(cas|bt|transit|bus|crc|hdg|hxp|nmp|sme|tcp|ttt|ucb).*from\s+(?P<orig>[\w\s,]+)",  # "when is bus from X"
        r"(cas|bt|transit|bus|crc|hdg|hxp|nmp|sme|tcp|ttt|ucb).*(schedule|next|when|time)",  # "CAS schedule" or "CAS next"
    ]
    
    for pattern in bus_schedule_patterns:
        m = re.search(pattern, q)
        if m:
            intent = INTENT_NEXT_BUS
            if "orig" in m.groupdict() and m.group("orig"):
                orig = m.group("orig").strip()
            # Extract bus route if mentioned
            if "cas" in q:
                bus_route = "CAS"
            elif "bt" in q:
                bus_route = "BT"
            elif "crc" in q:
                bus_route = "CRC"
            elif "hdg" in q:
                bus_route = "HDG"
            elif "hxp" in q:
                bus_route = "HXP"
            elif "nmp" in q:
                bus_route = "NMP"
            elif "sme" in q:
                bus_route = "SME"
            elif "tcp" in q:
                bus_route = "TCP"
            elif "ttt" in q:
                bus_route = "TTT"
            elif "ucb" in q:
                bus_route = "UCB"
            break
    
    # 2. Enhanced route/direction patterns
    route_patterns = [
        r"(quickest|fastest|best).*(route|way).*to\s+(?P<dest>[\w\s,]+)",  # "fastest route to X"
        r"(how|way).*(get|go).*to\s+(?P<dest>[\w\s,]+).*from\s+(?P<orig>[\w\s,]+)",  # "how to get to X from Y"
        r"from\s+(?P<orig>[\w\s,]+?)\s+to\s+(?P<dest>[\w\s,]+)",  # "from X to Y"
        r"(how|way).*(get|go).*from\s+(?P<orig>[\w\s,]+).*to\s+(?P<dest>[\w\s,]+)",  # "how to get from X to Y"
        r"(directions|route).*from\s+(?P<orig>[\w\s,]+).*to\s+(?P<dest>[\w\s,]+)",  # "directions from X to Y"
        r"(how|way).*(get|go|travel).*to\s+(?P<dest>[\w\s,]+)",  # "how to get to X" (general)
        r"(directions|route).*to\s+(?P<dest>[\w\s,]+)",  # "directions to X"
        r"(travel|go).*from\s+(?P<orig>[\w\s,]+).*to\s+(?P<dest>[\w\s,]+)",  # "travel from X to Y"
    ]
    
    for pattern in route_patterns:
        m = re.search(pattern, q)
        if m:
            intent = INTENT_TRANSIT_ROUTE
            if "dest" in m.groupdict() and m.group("dest"):
                dest = m.group("dest").strip()
            if "orig" in m.groupdict() and m.group("orig"):
                orig = m.group("orig").strip()
            break
    
    # 3. Current location patterns
    location_patterns = [
        r"i.*am.*(at|in|currently at)\s+(?P<orig>[\w\s,]+)",  # "I am at X"
        r"(currently|right now).*(at|in)\s+(?P<orig>[\w\s,]+)",  # "currently at X"
        r"from.*here.*to\s+(?P<dest>[\w\s,]+)",  # "from here to X"
        r"at\s+(?P<orig>[\w\s,]+).*right now",  # "at X right now"
    ]
    
    for pattern in location_patterns:
        m = re.search(pattern, q)
        if m:
            if intent == "generic":
                intent = INTENT_TRANSIT_ROUTE
            if "orig" in m.groupdict() and m.group("orig") and not orig:
                orig = m.group("orig").strip()
            if "dest" in m.groupdict() and m.group("dest") and not dest:
                dest = m.group("dest").strip()
            break
    
    # 4. Next bus patterns (enhanced)
    next_bus_patterns = [
        r"(next|soonest).*(bus|route).*to\s+(?P<dest>[\w\s,]+)",  # "next bus to X"
        r"when.*(bus|route).*to\s+(?P<dest>[\w\s,]+)",  # "when is bus to X"
        r"(next|when).*(bus|route).*(?P<dest>[\w\s,]+)",  # "next bus X"
    ]
    
    if intent == "generic":
        for pattern in next_bus_patterns:
            m = re.search(pattern, q)
            if m:
                intent = INTENT_NEXT_BUS
                if "dest" in m.groupdict() and m.group("dest"):
                    dest = m.group("dest").strip()
                break
    
    # 5. Enhanced building name detection
    building_keywords = {
        # Major Buildings
        "goodwin": "goodwin hall",
        "lavery": "lavery hall",
        "squires": "squires",
        "torgersen": "torgersen hall",
        "mcbryde": "mcbryde hall",
        "norris": "norris hall",
        "randolph": "randolph hall",
        "newman": "newman library",
        "dietrick": "dietrick hall",
        "west egg": "west eggleston",
        "east egg": "east eggleston",
        
        # Dining Halls
        "d2": "d2",
        "owens": "owens food court",
        "hokie grill": "hokie grill",
        "turner": "turner place",
        
        # Residential Halls
        "barringer": "barringer hall",
        "hoge": "hoge hall",
        "johnson": "johnson hall",
        "lee": "lee hall",
        "miles": "miles hall",
        "pridemore": "pridemore hall",
        "slusher": "slusher hall",
        "vawter": "vawter hall",
        
        # Off-Campus Areas
        "main": "main street",
        "progress": "progress street",
        "university city": "university city",
        "ucb": "university city",
        "toms creek": "toms creek",
        "hethwood": "hethwood",
        "harding": "harding avenue",
        "patrick henry": "patrick henry drive",
        "crc": "corporate research center",
        "downtown": "downtown",
        
        # Sports Facilities
        "cassell": "cassell coliseum",
        "lane": "lane stadium",
        "english field": "english field",
        "aquatic": "aquatic center",
        "mccomas": "mccomas hall",
        
        # Parking
        "perry": "perry street",
        "parking": "campus",
        
        # General Campus
        "campus": "campus",
        "vt": "virginia tech",
        "tech": "virginia tech",
    }
    
    for keyword, full_name in building_keywords.items():
        if keyword in q:
            if not dest and (f"to {keyword}" in q or f"{keyword}" in q.split()[-3:]):
                dest = full_name
            if not orig and (f"from {keyword}" in q or f"at {keyword}" in q):
                orig = full_name
    
    # 6. Address pattern detection
    address_pattern = r"\d+\s+[\w\s]+\s+(street|st|road|rd|drive|dr|avenue|ave|way|lane|ln)"
    addresses = re.findall(address_pattern, q, re.IGNORECASE)
    if addresses:
        # Find the full address in the query
        for match in re.finditer(address_pattern, q, re.IGNORECASE):
            full_address = match.group(0)
            if "from" in q[:match.start()]:
                orig = full_address
            elif "to" in q[:match.start()] or intent == INTENT_TRANSIT_ROUTE:
                dest = full_address
    
    # Check if user specifically asked for bus
    bus_only = any(word in q for word in ["bus from", "bus to", "take bus", "by bus", "using bus"])
    
    return {
        "intent": intent,
        "origin": normalize_place(orig) if orig else None,
        "destination": normalize_place(dest) if dest else None,
        "bus_route": bus_route,
        "bus_only": bus_only
    }

def nvidia_nim_parse(query: str) -> Dict[str, Optional[str]]:
    """Use NVIDIA NIMs for intent parsing"""
    nim_endpoint = os.getenv("NVIDIA_NIM_ENDPOINT")
    nim_api_key = os.getenv("NVIDIA_NIM_API_KEY")
    
    if not nim_endpoint or not nim_api_key or requests is None:
        return simple_parse(query)
    
    system_prompt = (
        "Extract JSON: {intent:[next_bus,transit_route,generic], origin, destination}. "
        "Prefer campus building names as given."
    )
    
    try:
        headers = {
            "Authorization": f"Bearer {nim_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "meta/llama-3.1-8b-instruct",  # or your preferred NVIDIA NIM model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(f"{nim_endpoint}/v1/chat/completions", 
                               headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        data = json.loads(content)
        
        return {
            "intent": data.get("intent") or "generic",
            "origin": normalize_place(data.get("origin")) if data.get("origin") else None,
            "destination": normalize_place(data.get("destination")) if data.get("destination") else None,
            "bus_route": data.get("bus_route")
        }
    except Exception as e:
        print(f"NVIDIA NIM error: {e}")
        return simple_parse(query)

def openai_parse(query: str) -> Dict[str, Optional[str]]:
    """Use OpenAI for intent parsing"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return simple_parse(query)
    
    client = OpenAI(api_key=api_key)
    system = (
        "Extract JSON: {intent:[next_bus,transit_route,generic], origin, destination}. "
        "Prefer campus building names as given."
    )
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        
        data = json.loads(resp.choices[0].message.content)
        return {
            "intent": data.get("intent") or "generic",
            "origin": normalize_place(data.get("origin")) if data.get("origin") else None,
            "destination": normalize_place(data.get("destination")) if data.get("destination") else None,
            "bus_route": data.get("bus_route")
        }
    except Exception:
        return simple_parse(query)

def llm_parse(query: str) -> Dict[str, Optional[str]]:
    """Main parsing function - uses optimized simple parser for fast, reliable campus queries"""
    # Use simple parser only - fast, reliable, and works great for campus queries
    return simple_parse(query)

def parse_transit_query(query: str) -> Dict[str, Optional[str]]:
    return llm_parse(query)
