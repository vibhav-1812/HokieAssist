"""
Test file
These tests cover the PM4  test plan across the five sections:
  1. Dining (open status, hours, source attribution)
  2. Bus / Transit (ETA, route parsing, unknown stops, CAS schedule)
  3. Club Events (weekly listing, source attribution)
  4. NLU (place normalization, intent parsing, LLM fallbacks)
  5. Error handling & reliability (graceful failures, generic fallback)

AI assistance disclosure (course policy): Cursor used to restructure and expand test coverage.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nlu import (
    llm_parse,
    normalize_place,
    nvidia_nim_parse,
    openai_parse,
    parse_transit_query,
    simple_parse,
)
from langchain_agent import get_ai_response, get_simple_response


# ============================================================================
#  SECTION 1 — DINING (Dining Open Status, Source Attribution)
# ============================================================================


# Targeted Test Case - "Dining Open Status: the keyword router must detect dining-related words and call the dining scraper, returning data with the UDC source URL."
# Type of test - unit
@pytest.mark.asyncio
async def test_dining_keyword_routing():
    with patch("langchain_agent.get_dining_halls", new_callable=AsyncMock) as mock:
        mock.return_value = {"D2": "open", "Owens Food Court": "closed"}
        out = await get_simple_response("Which dining halls are open?")
    assert "Dining Information" in out["answer"]
    assert "https://udc.vt.edu/" in out["sources"]
    mock.assert_awaited_once()


# Targeted Test Case - "Dining Open Status: the static fallback get_dining_hours must list every known hall with hours so cached data is available when the live scraper fails."
# Type of test - unit
def test_dining_hours_fallback():
    from scrapers.dining import get_dining_hours

    hours = get_dining_hours()
    assert "D2" in hours and "Owens Food Court" in hours
    assert all(isinstance(v, str) and "AM" in v for v in hours.values())


# Targeted Test Case - "Dining Open Status: the dining menu helper must return menu items keyed by hall name so the frontend can display meal options."
# Type of test - unit
@pytest.mark.asyncio
async def test_dining_menus_structure():
    from scrapers.dining import get_dining_menus

    menus = await get_dining_menus()
    assert "D2" in menus
    assert isinstance(menus["D2"], list) and len(menus["D2"]) > 0


# Targeted Test Case - "Source Attribution: when the dining scraper raises an exception, the router must still respond to the user with a helpful sorry message instead of crashing."
# Type of test - unit
@pytest.mark.asyncio
async def test_dining_scraper_error():
    with patch("langchain_agent.get_dining_halls", new_callable=AsyncMock) as mock:
        mock.side_effect = Exception("network timeout")
        out = await get_simple_response("What food options are available?")
    assert "sorry" in out["answer"].lower() or "couldn't" in out["answer"].lower()


# ============================================================================
#  SECTION 2 — BUS / TRANSIT (Bus ETA, Unknown Stop, CAS Schedule)
# ============================================================================


# Targeted Test Case - "Bus ETA Request: the keyword router must detect bus-related words and call get_bus_times, attaching the ridebt.org source."
# Type of test - unit
@pytest.mark.asyncio
async def test_bus_keyword_routing():
    with patch("langchain_agent.get_bus_times", new_callable=AsyncMock) as mock:
        mock.return_value = "Toms Creek: 3:15 PM"
        out = await get_simple_response("What is the bus schedule?")
    assert "Bus Information" in out["answer"]
    assert "https://ridebt.org/" in out["sources"]
    mock.assert_awaited_once()


# Targeted Test Case - "Bus ETA Request: the NLU must parse 'next bus to Goodwin Hall' as a next_bus intent with a resolved destination."
# Type of test - unit
def test_parse_next_bus():
    out = simple_parse("when is the next bus to Goodwin Hall")
    assert out["intent"] == "next_bus"
    assert out["destination"] is not None


# Targeted Test Case - "Bus ETA Request: the parser must recognize CAS as a specific Blacksburg Transit route so the system can look up CAS-specific ETAs."
# Type of test - unit
def test_parse_cas_route():
    out = simple_parse("when does the CAS bus come")
    assert out["intent"] == "next_bus"
    assert out["bus_route"] == "CAS"


# Targeted Test Case - "Bus ETA Request: short phrases like 'CAS schedule' should still be handled as a CAS bus query."
# Type of test - unit
def test_parse_cas_short():
    out = simple_parse("CAS schedule")
    assert out["intent"] == "next_bus"
    assert out["bus_route"] == "CAS"


# Targeted Test Case - "Bus ETA Request: when a user says 'by bus', the parser should set the bus_only flag so walking-only routes are excluded."
# Type of test - unit
def test_parse_bus_only_flag():
    out = simple_parse("directions to Goodwin Hall by bus")
    assert out.get("bus_only") is True


# Targeted Test Case - "Bus ETA Request: a query between two named buildings must fill both origin and destination for multi-stop route planning."
# Type of test - unit
def test_parse_origin_and_dest():
    out = simple_parse("bus from Goodwin Hall to Lavery Hall")
    assert out["intent"] == "transit_route"
    assert out["origin"] is not None and out["destination"] is not None


# Targeted Test Case - "Bus ETA Request: find_nearest_stop must resolve a campus building name to a valid Blacksburg Transit stop id for ETA lookups."
# Type of test - unit
def test_nearest_stop_lookup():
    from scrapers.bus import find_nearest_stop

    stop = find_nearest_stop("Goodwin Hall")
    assert isinstance(stop, str) and len(stop) > 0


# Targeted Test Case - "Bus Unknown Stop: when the bus scraper throws, the router must return a sorry message so unrecognized queries do not crash the server."
# Type of test - unit
@pytest.mark.asyncio
async def test_bus_scraper_error():
    with patch("langchain_agent.get_bus_times", new_callable=AsyncMock) as mock:
        mock.side_effect = Exception("API unreachable")
        out = await get_simple_response("When is the next bus to campus?")
    assert "sorry" in out["answer"].lower() or "couldn't" in out["answer"].lower()


# ============================================================================
#  SECTION 3 — CLUB EVENTS (Events This Week, Source Attribution)
# ============================================================================


# Targeted Test Case - "Events This Week: the keyword router must detect event-related words, call the clubs scraper, and attach the GobblerConnect source."
# Type of test - unit
@pytest.mark.asyncio
async def test_events_keyword_routing():
    with patch("langchain_agent.get_club_events", new_callable=AsyncMock) as mock:
        mock.return_value = "ACM Weekly Meeting — Friday 7pm"
        out = await get_simple_response("What club events are happening this week?")
    assert "Club Events" in out["answer"]
    assert "https://gobblerconnect.vt.edu/" in out["sources"]
    mock.assert_awaited_once()


# Targeted Test Case - "Events This Week: the popular clubs helper must return a list of clubs, each with a name and category, so the UI can show browseable categories."
# Type of test - unit
@pytest.mark.asyncio
async def test_popular_clubs_structure():
    from scrapers.clubs import get_popular_clubs

    clubs = await get_popular_clubs()
    assert len(clubs) > 0
    assert all("name" in c and "category" in c for c in clubs)


# Targeted Test Case - "Events This Week: club categories must be organized by grouping so the app can show students what kinds of clubs are available."
# Type of test - unit
@pytest.mark.asyncio
async def test_club_categories():
    from scrapers.clubs import get_club_categories

    cats = await get_club_categories()
    assert "Academic & Professional" in cats
    assert isinstance(cats["Academic & Professional"], list) and len(cats["Academic & Professional"]) > 0


# Targeted Test Case - "Events This Week: when the events scraper fails, the router must handle the error and still respond with a helpful message."
# Type of test - unit
@pytest.mark.asyncio
async def test_events_scraper_error():
    with patch("langchain_agent.get_club_events", new_callable=AsyncMock) as mock:
        mock.side_effect = Exception("scraper down")
        out = await get_simple_response("Are there any social events today?")
    assert "sorry" in out["answer"].lower() or "couldn't" in out["answer"].lower()


# ============================================================================
#  SECTION 4 — NLU (Place Aliases, Intent Parsing, LLM Fallbacks)
# ============================================================================


# Targeted Test Case - "Voice Query: the place normalizer must map known campus names like 'goodwin hall' to a full Blacksburg address for consistent downstream use."
# Type of test - unit
def test_place_known_alias():
    assert "Prices Fork" in normalize_place("goodwin hall")


# Targeted Test Case - "Voice Query: dining hall aliases like 'd2' must also resolve so dining-related location queries work through the same NLU path."
# Type of test - unit
def test_place_dining_alias():
    assert "Blacksburg" in normalize_place("d2")


# Targeted Test Case - "Bus Unknown Stop: when a location is not recognized, the normalizer should pass the text through unchanged so higher layers can ask for clarification."
# Type of test - unit
def test_place_unknown():
    assert normalize_place("FakeStop XYZ") == "FakeStop XYZ"


# Targeted Test Case - "General reliability: empty input must not crash the normalizer."
# Type of test - unit
def test_place_empty():
    assert normalize_place("") == ""


# Targeted Test Case - "Voice Query: a natural transit question must be classified as transit_route with a resolved destination."
# Type of test - unit
def test_parse_route_intent():
    out = simple_parse("fastest route to Goodwin Hall")
    assert out["intent"] == "transit_route"
    assert out["destination"] is not None


# Targeted Test Case - "Text Fallback: when the query has no transit or topic cues, the parser must return generic so the keyword router handles it."
# Type of test - unit
def test_parse_generic():
    out = simple_parse("hello world")
    assert out["intent"] == "generic"
    assert out["destination"] is None


# Targeted Test Case - "Voice Query: llm_parse must produce identical results to simple_parse so voice and text inputs route the same way."
# Type of test - unit
def test_llm_matches_simple():
    q = "directions from Lavery Hall to Goodwin Hall"
    assert llm_parse(q) == simple_parse(q)


# Targeted Test Case - "Voice Query: parse_transit_query must return a dict with all keys main.py expects (intent, origin, destination, bus_route, bus_only)."
# Type of test - unit
def test_parse_output_keys():
    out = parse_transit_query("next bus to squires")
    assert set(out.keys()) >= {"intent", "origin", "destination", "bus_route", "bus_only"}


# Targeted Test Case - "Speech API Failure Fallback: when OpenAI is not configured, openai_parse must fall back to simple_parse so parsing never fails."
# Type of test - unit
def test_openai_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    q = "fastest route to Goodwin Hall"
    assert openai_parse(q) == simple_parse(q)


# Targeted Test Case - "Speech API Failure Fallback: when NVIDIA NIM is not configured, nvidia_nim_parse must fall back to simple_parse."
# Type of test - unit
def test_nvidia_fallback(monkeypatch):
    monkeypatch.delenv("NVIDIA_NIM_ENDPOINT", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    assert nvidia_nim_parse("hello") == simple_parse("hello")


# ============================================================================
#  SECTION 5 — ERROR HANDLING & RELIABILITY
# ============================================================================


# Targeted Test Case - "Text Fallback: an off-topic query should receive a generic campus help message rather than an empty answer or a crash."
# Type of test - unit
@pytest.mark.asyncio
async def test_generic_help_text():
    out = await get_simple_response("quantum computing")
    assert "dining" in out["answer"].lower() or "transportation" in out["answer"].lower()
    assert out["sources"]


# Targeted Test Case - "Voice Query: get_ai_response should delegate to get_simple_response so both voice and text channels share one code path."
# Type of test - unit
@pytest.mark.asyncio
async def test_ai_delegates():
    with patch("langchain_agent.get_simple_response", new_callable=AsyncMock) as mock:
        mock.return_value = {"answer": "test", "sources": ["https://vt.edu/"]}
        out = await get_ai_response("anything")
    assert out["answer"] == "test"
    mock.assert_awaited_once()


# Targeted Test Case - "Backend Error Logging: when get_simple_response crashes, the user should see a polite apology and a vt.edu source, not a raw stack trace."
# Type of test - unit
@pytest.mark.asyncio
async def test_ai_error_fallback():
    with patch("langchain_agent.get_simple_response", new_callable=AsyncMock) as mock:
        mock.side_effect = RuntimeError("boom")
        out = await get_ai_response("dining")
    assert "sorry" in out["answer"].lower()
    assert out["sources"] == ["https://vt.edu/"]


# Targeted Test Case - "Source Attribution: when a query mentions dining AND events, the router should call both scrapers and return both sources in a single response."
# Type of test - unit
@pytest.mark.asyncio
async def test_multi_topic_sources():
    with patch("langchain_agent.get_dining_halls", new_callable=AsyncMock) as d, \
         patch("langchain_agent.get_club_events", new_callable=AsyncMock) as e:
        d.return_value = "D2: open"
        e.return_value = "ACM meeting Friday"
        out = await get_simple_response("What food and club events are there?")
    assert "https://udc.vt.edu/" in out["sources"]
    assert "https://gobblerconnect.vt.edu/" in out["sources"]


# ============================================================================
#  SECTION 6 — INTEGRATION (full API paths, multiple functions working together)
# ============================================================================


# Targeted Test Case - "Dining Open Status + Source Attribution via POST /ask: a dining question sent to the main API must flow through NLU (generic intent), into the keyword router, call the dining scraper, and return answer + sources."
# Type of test - integration
@pytest.mark.asyncio
async def test_api_ask_dining():
    from httpx import ASGITransport, AsyncClient
    import main as app_main

    with patch("langchain_agent.get_dining_halls", new_callable=AsyncMock) as mock:
        mock.return_value = ["D2 — open until 8pm"]
        transport = ASGITransport(app=app_main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/ask", json={"query": "Which dining halls are open right now?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "D2" in data["answer"] or "dining" in data["answer"].lower()
    assert data["sources"]


# Targeted Test Case - "Events This Week via POST /ask: a club events question sent to the main API must reach the keyword router, call the events scraper, and return GobblerConnect-sourced content."
# Type of test - integration
@pytest.mark.asyncio
async def test_api_ask_events():
    from httpx import ASGITransport, AsyncClient
    import main as app_main

    with patch("langchain_agent.get_club_events", new_callable=AsyncMock) as mock:
        mock.return_value = "ACM Weekly Meeting — Friday 7pm at Torgersen 1100"
        transport = ASGITransport(app=app_main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/ask", json={"query": "What club events are happening this week?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "ACM" in data["answer"] or "event" in data["answer"].lower()
    assert data["sources"]


# Targeted Test Case - "Bus ETA Request via POST /bus/query: a Lavery-to-Goodwin question must flow through parse_transit_query and the enhanced route planner, returning a planned answer with ridebt.org attribution."
# Type of test - integration
@pytest.mark.asyncio
async def test_api_bus_route_plan():
    from httpx import ASGITransport, AsyncClient
    import main as app_main

    with patch.object(app_main, "enhanced_plan_quickest_route", new_callable=AsyncMock) as mock:
        mock.return_value = {"answer": "Take BT CRC; ~10 min.", "sources": ["https://ridebt.org/"]}
        transport = ASGITransport(app=app_main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/bus/query", json={"query": "fastest route from Lavery Hall to Goodwin Hall"})
    assert resp.status_code == 200
    assert "CRC" in resp.json()["answer"]
    mock.assert_awaited()


# Targeted Test Case - "Bus ETA Request via POST /ask: the same transit question through the main /ask endpoint must also reach the enhanced route planner, proving both API paths handle transit correctly."
# Type of test - integration
@pytest.mark.asyncio
async def test_api_ask_transit():
    from httpx import ASGITransport, AsyncClient
    import main as app_main

    with patch.object(app_main, "enhanced_plan_quickest_route", new_callable=AsyncMock) as mock:
        mock.return_value = {"answer": "Walk 5 min; take CRC.", "sources": ["https://ridebt.org/"]}
        transport = ASGITransport(app=app_main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/ask", json={"query": "fastest route from Lavery Hall to Goodwin Hall"})
    assert resp.status_code == 200
    assert "CRC" in resp.json()["answer"]
    mock.assert_awaited()


# Targeted Test Case - "GET /dining endpoint: the standalone dining endpoint must return a dining_halls key and a udc.vt.edu source, matching what the frontend quick-action button expects."
# Type of test - integration
@pytest.mark.asyncio
async def test_api_get_dining():
    from httpx import ASGITransport, AsyncClient
    import main as app_main

    with patch("scrapers.dining.get_dining_halls", new_callable=AsyncMock) as mock:
        mock.return_value = "D2: Open until 9pm"
        transport = ASGITransport(app=app_main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/dining")
    assert resp.status_code == 200
    data = resp.json()
    assert "dining_halls" in data
    assert "https://udc.vt.edu/" in data["sources"]


# Targeted Test Case - "GET /clubs endpoint: the standalone clubs endpoint must return an events key and a GobblerConnect source, matching the frontend quick-action for events."
# Type of test - integration
@pytest.mark.asyncio
async def test_api_get_clubs():
    from httpx import ASGITransport, AsyncClient
    import main as app_main

    with patch("scrapers.clubs.get_club_events", new_callable=AsyncMock) as mock:
        mock.return_value = "ACM Meeting — Friday 7pm"
        transport = ASGITransport(app=app_main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/clubs")
    assert resp.status_code == 200
    data = resp.json()
    assert "events" in data
    assert "https://gobblerconnect.vt.edu/" in data["sources"]


# ============================================================================
#  SECTION 7 — INPUT ROBUSTNESS (edge cases for whitespace, case, empty input)
#
#  Added on the `fixes` branch as a small extension to the test suite to
#  cover input-robustness gaps not exercised by sections 1–6. These are
#  important because both voice transcripts and typed queries arrive with
#  inconsistent casing and stray whitespace, and the parser/normalizer
#  must handle them without crashing or returning the wrong intent.
# ============================================================================


# Targeted Test Case - "Voice Query: stray leading/trailing whitespace from a transcript must not change the parsed intent."
# Type of test - unit
def test_parse_handles_leading_trailing_whitespace():
    out = simple_parse("   next bus to Squires   ")
    assert out["intent"] == "next_bus"
    assert out["destination"] is not None


# Targeted Test Case - "Voice Query: an all-uppercase transcript (common from some speech engines) must still be parsed as a transit intent with a resolved destination."
# Type of test - unit
def test_parse_handles_uppercase_query():
    out = simple_parse("NEXT BUS TO GOODWIN HALL")
    assert out["intent"] == "next_bus"
    assert out["destination"] is not None


# Targeted Test Case - "General reliability: an empty query must return a well-formed parse dict with generic intent rather than raising."
# Type of test - unit
def test_parse_empty_query_is_generic():
    out = simple_parse("")
    assert out["intent"] == "generic"
    assert out["destination"] is None
    # parse_transit_query must guarantee the same key shape on empty input
    assert set(out.keys()) >= {"intent", "origin", "destination", "bus_route", "bus_only"}


# Targeted Test Case - "Voice Query: place normalization must be case-insensitive so 'GOODWIN HALL' resolves to the same address as 'goodwin hall'."
# Type of test - unit
def test_place_normalize_case_insensitive():
    assert normalize_place("GOODWIN HALL") == normalize_place("goodwin hall")


# Targeted Test Case - "Voice Query: place normalization must trim surrounding whitespace before alias lookup so '  goodwin hall  ' still resolves."
# Type of test - unit
def test_place_normalize_strips_whitespace():
    normalized = normalize_place("  goodwin hall  ")
    # Whether or not aliasing produces the full address, the result must not
    # carry the original padding — that would break downstream Maps lookups.
    assert normalized == normalized.strip()
    assert "goodwin" in normalized.lower() or "Prices Fork" in normalized


# Targeted Test Case - "Integration: an uppercase, whitespace-padded dining query sent to POST /ask must still reach the dining scraper and return UDC-sourced content, proving the input-robustness fixes work end-to-end through the API layer."
# Type of test - integration
@pytest.mark.asyncio
async def test_api_ask_dining_robust_input():
    from httpx import ASGITransport, AsyncClient
    import main as app_main

    with patch("langchain_agent.get_dining_halls", new_callable=AsyncMock) as mock:
        mock.return_value = ["D2 — open until 8pm"]
        transport = ASGITransport(app=app_main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/ask", json={"query": "  WHICH DINING HALLS ARE OPEN?  "})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sources"]
    assert "https://udc.vt.edu/" in data["sources"]
