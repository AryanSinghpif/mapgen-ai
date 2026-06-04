"""
aliases.py — Curated India district name alias dictionary
==========================================================
Maps historical / alternate district names → canonical names used in modern
shapefiles (typically 2011 Census vintage or later).

HOW TO EXTEND
-------------
Add entries as  "old or alternate name": "shapefile canonical name"
Keys and values are matched AFTER normalization (lowercase, no accents,
no punctuation), so you don't need to worry about case or diacritics here —
but writing them cleanly makes the list easier to maintain.

Organized by category for readability. All entries are case-insensitive at
match time, so "Bangalore" and "bangalore" both work.
"""

ALIASES: dict[str, str] = {

    # ── Major city / administrative renames ──────────────────────────────
    "bangalore":                "bengaluru",
    "bangalore urban":          "bengaluru urban",
    "bangalore rural":          "bengaluru rural",
    "mysore":                   "mysuru",
    "mangalore":                "dakshina kannada",   # common shorthand
    "hubli":                    "dharwad",            # Hubli-Dharwad district
    "hubli dharwad":            "dharwad",
    "belgaum":                  "belagavi",
    "bijapur":                  "vijayapura",
    "gulbarga":                 "kalaburagi",
    "shimoga":                  "shivamogga",
    "tumkur":                   "tumakuru",
    "bellary":                  "ballari",
    "bidar":                    "bidar",              # no change, but common mispelling
    "hospet":                   "vijayanagara",       # 2021 new district

    "bombay":                   "mumbai",
    "poona":                    "pune",
    "nasik":                    "nashik",
    "sholapur":                 "solapur",
    "kolhapur":                 "kolhapur",
    "aurangabad":               "chhatrapati sambhajinagar",  # renamed 2023
    "osmanabad":                "dharashiv",                  # renamed 2023
    "ahmednagar":               "ahilyanagar",                # renamed 2023

    "madras":                   "chennai",
    "coimbatore":               "coimbatore",         # unchanged but alias kept for completeness
    "tirunelveli":              "tirunelveli",
    "tiruchirapalli":           "tiruchirappalli",
    "vellore":                  "vellore",
    "chidambaram":              "cuddalore",          # Chidambaram is in Cuddalore dist.

    "calcutta":                 "kolkata",
    "burdwan":                  "purba bardhaman",
    "west burdwan":             "paschim bardhaman",
    "east burdwan":             "purba bardhaman",
    "midnapur":                 "paschim medinipur",  # after split
    "east midnapur":            "purba medinipur",
    "west midnapur":            "paschim medinipur",
    "hooghly":                  "hugli",
    "howrah":                   "haora",
    "24 parganas":              "north 24 parganas",  # ambiguous; safer to surface to user

    "allahabad":                "prayagraj",          # renamed 2018
    "faizabad":                 "ayodhya",            # renamed 2018
    "muzaffarnagar":            "muzaffarnagar",
    "gorakhpur":                "gorakhpur",
    "varanasi":                 "varanasi",
    "meerut":                   "meerut",
    "bareilly":                 "bareilly",
    "lucknow":                  "lucknow",
    "kanpur":                   "kanpur nagar",
    "kanpur nagar":             "kanpur nagar",
    "kanpur dehat":             "kanpur dehat",
    "agra":                     "agra",
    "mathura":                  "mathura",

    "gurgaon":                  "gurugram",           # renamed 2016
    "mewat":                    "nuh",                # renamed 2016
    "mohindergarh":             "mahendragarh",

    "pondicherry":              "puducherry",

    "daman":                    "daman",
    "diu":                      "diu",

    # ── Post-2011 district splits (boundary-vintage note) ────────────────
    # If using a 2011-boundary shapefile, these NEW districts won't exist.
    # The matcher will flag them as unmatched — this is CORRECT BEHAVIOR.
    # Entries below map the new-district name to the pre-split parent,
    # so the data can still be plotted on an older shapefile (with caveat).

    "palghar":                  "thane",              # split from Thane 2014 (MH)
    "raigad":                   "raigad",
    "pathankot":                "gurdaspur",          # split from Gurdaspur 2011 (PB)

    "palwal":                   "faridabad",          # split from Faridabad 2008 (HR)

    "hapur":                    "ghaziabad",          # split 2011 (UP)
    "shamli":                   "muzaffarnagar",      # split 2011 (UP)
    "sambhal":                  "moradabad",          # split 2011 (UP)
    "amroha":                   "jyotiba phule nagar",
    "jyotiba phule nagar":      "amroha",             # renamed 2012

    "boudh":                    "phulbani",           # Odisha
    "subarnapur":               "sonepur",            # Odisha

    # ── Common transliteration variants ──────────────────────────────────
    "thiruvananthapuram":       "thiruvananthapuram",
    "trivandrum":               "thiruvananthapuram",
    "calicut":                  "kozhikode",
    "trichur":                  "thrissur",
    "cochin":                   "ernakulam",
    "ernakulam":                "ernakulam",
    "quilon":                   "kollam",
    "alleppey":                 "alappuzha",
    "palghat":                  "palakkad",
    "cannanore":                "kannur",
    "kasaragod":                "kasaragod",
    "wynand":                   "wayanad",

    "hyderabad":                "hyderabad",
    "secunderabad":             "hyderabad",          # part of Hyderabad district
    "warangal":                 "warangal urban",     # after 2016 split
    "warangal rural":           "warangal rural",
    "ranga reddy":              "ranga reddy",
    "karimnagar":               "karimnagar",
    "nizamabad":                "nizamabad",
    "adilabad":                 "adilabad",
    "khammam":                  "khammam",
    "mahbubnagar":              "mahbubnagar",

    "vishakhapatnam":           "visakhapatnam",
    "vizag":                    "visakhapatnam",
    "vizianagaram":             "vizianagaram",
    "krishna":                  "krishna",
    "guntur":                   "guntur",
    "nellore":                  "sri potti sriramulu nellore",
    "cuddapah":                 "kadapa",
    "kurnool":                  "kurnool",
    "anantapur":                "anantapur",
    "chittoor":                 "chittoor",

    "guwahati":                 "kamrup metropolitan",
    "nowgong":                  "nagaon",
    "tezpur":                   "sonitpur",
    "dibrugarh":                "dibrugarh",

    "jamshedpur":               "east singhbhum",
    "ranchi":                   "ranchi",
    "dhanbad":                  "dhanbad",
    "bokaro":                   "bokaro",
    "hazaribagh":               "hazaribag",

    "patna":                    "patna",
    "gaya":                     "gaya",
    "muzaffarpur":              "muzaffarpur",
    "bhagalpur":                "bhagalpur",
    "darbhanga":                "darbhanga",

    "bhopal":                   "bhopal",
    "indore":                   "indore",
    "jabalpur":                 "jabalpur",
    "gwalior":                  "gwalior",
    "rewa":                     "rewa",

    "jaipur":                   "jaipur",
    "jodhpur":                  "jodhpur",
    "udaipur":                  "udaipur",
    "ajmer":                    "ajmer",
    "kota":                     "kota",
    "bikaner":                  "bikaner",
    "alwar":                    "alwar",

    "ahmedabad":                "ahmedabad",
    "surat":                    "surat",
    "vadodara":                 "vadodara",
    "rajkot":                   "rajkot",
    "baroda":                   "vadodara",
    "broach":                   "bharuch",
    "bharuch":                  "bharuch",

    "chandigarh":               "chandigarh",
    "amritsar":                 "amritsar",
    "ludhiana":                 "ludhiana",
    "jalandhar":                "jalandhar",
    "patiala":                  "patiala",

    "dehradun":                 "dehradun",
    "haridwar":                 "haridwar",
    "nainital":                 "nainital",

    "shimla":                   "shimla",
    "kangra":                   "kangra",
    "mandi":                    "mandi",

    "srinagar":                 "srinagar",
    "jammu":                    "jammu",
    "anantnag":                 "anantnag",
    "baramulla":                "baramulla",

    "delhi":                    "new delhi",
    "new delhi":                "new delhi",
    "south delhi":              "south delhi",
    "north delhi":              "north delhi",
    "east delhi":               "east delhi",
    "west delhi":               "west delhi",

    # ── Odisha (formerly Orissa) ─────────────────────────────────────────
    "orissa":                   "odisha",             # state-level (for context)
    "cuttack":                  "cuttack",
    "bhubaneswar":              "khordha",
    "khurda":                   "khordha",
    "puri":                     "puri",
    "sambalpur":                "sambalpur",
    "koraput":                  "koraput",
    "mayurbhanj":               "mayurbhanj",
    "balangir":                 "balangir",
    "bolangir":                 "balangir",
    "sundergarh":               "sundargarh",
    "sundargarh":               "sundargarh",
}


# ── Reverse alias lookup (shapefile → common alternatives) ───────────────
# Not used for matching, but useful for generating tooltips.
REVERSE_ALIASES: dict[str, list[str]] = {}
for alt, canonical in ALIASES.items():
    REVERSE_ALIASES.setdefault(canonical, []).append(alt)
