"""Module for cleaning and normalizing location data"""

from typing import Optional, Dict
import re

# US State mappings
US_STATES = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
}

# Common location patterns
LOCATION_PATTERNS = [
    # City, State
    r"([A-Za-z\s\.]+),\s*(?:([A-Za-z]{2})|([A-Za-z\s]+))",
    # State only
    r"\b(?:([A-Za-z]{2})|([A-Za-z\s]+))\b",
]


def normalize_state(state: str) -> Optional[str]:
    """Convert state name to standard two-letter code"""
    state = state.lower().strip()

    # If it's already a valid state code
    if state.upper() in US_STATES.values():
        return state.upper()

    # Try to match full state name
    return US_STATES.get(state)


def extract_location_components(text: str) -> Dict[str, Optional[str]]:
    """Extract city and state from location text"""
    if not text:
        return {"city": None, "state": None}

    # Clean the text
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace

    # Try city, state pattern first
    for pattern in LOCATION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                city = groups[0]
                state = groups[1] or groups[2]  # Try both 2-letter and full state name
                if state:
                    normalized_state = normalize_state(state)
                    if normalized_state:
                        return {
                            "city": city.strip().title() if city else None,
                            "state": normalized_state,
                        }

    # If no clear city/state pattern, try to find just a state
    words = text.split()
    for word in words:
        normalized_state = normalize_state(word)
        if normalized_state:
            return {"city": None, "state": normalized_state}

    return {"city": None, "state": None}


def clean_location(location: Optional[str]) -> Optional[str]:
    """Clean and normalize location string"""
    if not location:
        return None

    # Remove common noise patterns
    location = re.sub(
        r"accomplishments?|summary|skills?|highlights?",
        "",
        location,
        flags=re.IGNORECASE,
    )
    location = re.sub(r"\n+", " ", location)  # Replace newlines with spaces
    location = re.sub(r"\s+", " ", location)  # Normalize whitespace
    location = location.strip()

    if not location:
        return None

    # Extract components
    components = extract_location_components(location)

    # Format the clean location
    if components["city"] and components["state"]:
        return f"{components['city']}, {components['state']}"
    elif components["state"]:
        return components["state"]

    return None
