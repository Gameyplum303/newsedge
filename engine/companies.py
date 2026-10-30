"""Built-in company data: names, products, government business, relationships and exposures.

These are the defaults. Edits made in the dashboard's Companies tab, and profiles for companies you
add, are stored in data/profiles.json and override them (see profiles.py).
"""

# Ticker -> names used to recognise the company inside general business headlines.
# Names are matched case-sensitively on word boundaries.
COMPANIES = {
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "NVDA": ["Nvidia", "NVIDIA"],
    "AMZN": ["Amazon"],
    "GOOGL": ["Alphabet", "Google"],
    "META": ["Meta Platforms", "Meta", "Facebook", "Instagram"],
    "TSLA": ["Tesla"],
    "AVGO": ["Broadcom"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "INTC": ["Intel"],
    "NFLX": ["Netflix"],
    "ORCL": ["Oracle"],
    "CRM": ["Salesforce"],
    "ADBE": ["Adobe"],
    "PLTR": ["Palantir"],
    "UBER": ["Uber"],
    "COIN": ["Coinbase"],
    "JPM": ["JPMorgan", "JP Morgan"],
    "BAC": ["Bank of America"],
    "GS": ["Goldman Sachs"],
    "WFC": ["Wells Fargo"],
    "V": ["Visa Inc"],
    "MA": ["Mastercard"],
    "WMT": ["Walmart"],
    "COST": ["Costco"],
    "TGT": ["Target Corp"],
    "HD": ["Home Depot"],
    "NKE": ["Nike"],
    "SBUX": ["Starbucks"],
    "MCD": ["McDonald's"],
    "DIS": ["Disney"],
    "BA": ["Boeing"],
    "LMT": ["Lockheed Martin", "Lockheed"],
    "RTX": ["RTX Corp", "Raytheon"],
    "NOC": ["Northrop Grumman", "Northrop"],
    "GD": ["General Dynamics"],
    "LHX": ["L3Harris"],
    "XOM": ["Exxon", "ExxonMobil"],
    "CVX": ["Chevron"],
    "PFE": ["Pfizer"],
    "LLY": ["Eli Lilly"],
    "JNJ": ["Johnson & Johnson"],
    "UNH": ["UnitedHealth"],
    "MRK": ["Merck"],
    "KO": ["Coca-Cola"],
    "PEP": ["PepsiCo"],
    "F": ["Ford Motor"],
    "GM": ["General Motors"],
}

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "JPM", "NFLX"]

# Products, brands and units that identify a company in news that doesn't name it ("AWS outage").
# Kept to names that are unambiguous in headlines (no "Windows", "Chase", "Gemini").
COMPANY_ALIASES = {
    "AAPL": ["iPhone", "iPad", "MacBook", "App Store", "Apple Watch", "Vision Pro"],
    "MSFT": ["Azure", "Xbox", "Windows 11", "LinkedIn", "GitHub", "Activision", "Copilot"],
    "NVDA": ["GeForce", "CUDA"],
    "AMZN": ["AWS", "Amazon Web Services", "Whole Foods", "Prime Video"],
    "GOOGL": ["YouTube", "Android", "Gmail", "Waymo", "Google Cloud"],
    "META": ["WhatsApp", "Oculus"],
    "TSLA": ["Cybertruck", "Model Y", "Model 3", "Autopilot", "Gigafactory", "Full Self-Driving"],
    "AMD": ["Ryzen", "Radeon", "EPYC"],
    "JPM": ["Chase Bank", "JPMorgan Chase"],
    "BA": ["737 MAX", "787 Dreamliner", "Starliner"],
    "DIS": ["ESPN", "Disneyland", "Disney World", "Pixar"],
    "UBER": ["Uber Eats"],
    "XOM": ["ExxonMobil"],
    "WMT": ["Sam's Club"],
    "GM": ["Chevrolet", "Cadillac"],
    "F": ["F-150"],
    "LMT": ["F-35", "Sikorsky"],
    "RTX": ["Pratt & Whitney", "Collins Aerospace", "Patriot missile"],
    "NOC": ["B-21"],
    "GD": ["Gulfstream", "Electric Boat"],
}


# Government business: how much each area matters to the company (0..1, roughly its revenue share,
# rounded up for strategic importance). Drives how war, budget and policy news is read for it.
GOVERNMENT_EXPOSURE = {
    "LMT": {"defense": 1.0}, "NOC": {"defense": 1.0}, "LHX": {"defense": 1.0},
    "GD": {"defense": 0.8}, "RTX": {"defense": 0.6},
    "BA": {"defense": 0.4},
    "PLTR": {"defense": 0.6, "civilian": 0.4},
    "ORCL": {"civilian": 0.2, "defense": 0.1},
    "MSFT": {"civilian": 0.15, "defense": 0.1},
    "AMZN": {"civilian": 0.1, "defense": 0.05},
    "GOOGL": {"civilian": 0.05, "defense": 0.05},
    "INTC": {"civilian": 0.2, "defense": 0.1},      # CHIPS Act funding, US government stake
    "UNH": {"healthcare": 0.5},
    "PFE": {"healthcare": 0.25}, "MRK": {"healthcare": 0.25}, "LLY": {"healthcare": 0.25}, "JNJ": {"healthcare": 0.2},
}
# Search terms used to find each area's government news (see profiles.related_terms).
GOVERNMENT_QUERY_TERMS = {
    "defense": ["Pentagon", "defense budget", "defense spending", "NATO", "arms package"],
    "civilian": ["federal contract", "government cloud", "CHIPS Act"],
    "healthcare": ["Medicare", "Medicaid", "drug pricing"],
}
GOVERNMENT_AREA_LABELS = {"defense": "defense", "civilian": "civilian IT & services", "healthcare": "healthcare programs"}

# Budget and policy news per government area (case-insensitive). "all" applies to every area.
GOVERNMENT_TOPICS = {
    "defense": r"\b(defen[cs]e (budget|spending|bill|contracts?)|Pentagon|Department of (Defense|War)|DoD|military (aid|spending|budget)|NATO|arms (sales?|deal|package)|weapons|munitions|missile defen[cs]e|Golden Dome)\b",
    "civilian": r"\b(federal (IT|cloud|agencies|contracts?|workforce)|GSA|FedRAMP|government cloud|DOGE|CHIPS Act)\b",
    "healthcare": r"\b(Medicare|Medicaid|HHS|CMS|drug pricing|price negotiations?|BARDA|Affordable Care Act|ACA subsidies)\b",
    "all": r"\b(government shutdown|federal budget|spending bill|continuing resolution|debt ceiling|appropriations)\b",
}

# Company-to-company relationships. News about the related company flows to this one, weighted
# by `weight` (0..1). Suppliers, customers and partners move together; competitors move opposite.
RELATION_SIGN = {"supplier": 1, "customer": 1, "partner": 1, "competitor": -1}
RELATIONSHIPS = {
    "NVDA": [{"names": ["TSMC", "Taiwan Semiconductor"], "relation": "supplier", "weight": 0.8},
             {"names": ["OpenAI"], "relation": "customer", "weight": 0.4},
             {"names": ["Super Micro", "Supermicro"], "relation": "partner", "weight": 0.3},
             {"names": ["AMD"], "relation": "competitor", "weight": 0.3}],
    "AMD": [{"names": ["TSMC", "Taiwan Semiconductor"], "relation": "supplier", "weight": 0.8},
            {"names": ["OpenAI"], "relation": "customer", "weight": 0.4},
            {"names": ["Nvidia"], "relation": "competitor", "weight": 0.3},
            {"names": ["Intel"], "relation": "competitor", "weight": 0.3}],
    "AAPL": [{"names": ["TSMC", "Taiwan Semiconductor"], "relation": "supplier", "weight": 0.6},
             {"names": ["Foxconn", "Hon Hai"], "relation": "supplier", "weight": 0.6},
             {"names": ["Qualcomm"], "relation": "supplier", "weight": 0.3},
             {"names": ["Samsung"], "relation": "competitor", "weight": 0.2}],
    "MSFT": [{"names": ["OpenAI", "ChatGPT"], "relation": "partner", "weight": 0.7},
             {"names": ["Nvidia"], "relation": "supplier", "weight": 0.2}],
    "AMZN": [{"names": ["Anthropic"], "relation": "partner", "weight": 0.4},
             {"names": ["Nvidia"], "relation": "supplier", "weight": 0.2}],
    "GOOGL": [{"names": ["Anthropic"], "relation": "partner", "weight": 0.3},
              {"names": ["OpenAI", "ChatGPT"], "relation": "competitor", "weight": 0.3}],
    "META": [{"names": ["EssilorLuxottica", "Ray-Ban"], "relation": "partner", "weight": 0.2},
             {"names": ["TikTok", "ByteDance"], "relation": "competitor", "weight": 0.3}],
    "TSLA": [{"names": ["Panasonic"], "relation": "supplier", "weight": 0.4},
             {"names": ["CATL"], "relation": "supplier", "weight": 0.3},
             {"names": ["SpaceX", "xAI"], "relation": "partner", "weight": 0.3},
             {"names": ["BYD"], "relation": "competitor", "weight": 0.3}],
    "NFLX": [{"names": ["Warner Bros", "Paramount"], "relation": "competitor", "weight": 0.2}],
    "BA": [{"names": ["Spirit AeroSystems"], "relation": "supplier", "weight": 0.6},
           {"names": ["GE Aerospace"], "relation": "supplier", "weight": 0.3},
           {"names": ["Airbus"], "relation": "competitor", "weight": 0.4}],
    "PLTR": [{"names": ["Anduril"], "relation": "competitor", "weight": 0.2}],
    "INTC": [{"names": ["TSMC", "Taiwan Semiconductor"], "relation": "competitor", "weight": 0.3}],
    "LMT": [{"names": ["Anduril"], "relation": "competitor", "weight": 0.2}],
}

# Indirect exposure: a world event that doesn't name a company but plausibly moves it.
#   pattern     regex the headline/summary must match (case-insensitive)
#   categories  event types it applies to (see sentiment.WORLD_EVENT_CATEGORIES)
#   tickers     ticker -> +1 if bad news hurts the stock, -1 if the stock tends to benefit
EXPOSURES = [
    {"name": "Taiwan chip supply", "pattern": r"\b(Taiwan|TSMC|Hsinchu)\b",
     "categories": {"disaster", "accident", "geopolitical", "labor", "cyber_outage"},
     "tickers": {"NVDA": 1, "AMD": 1, "AAPL": 1, "AVGO": 1}},
    {"name": "China manufacturing", "pattern": r"\b(Foxconn|Zhengzhou|Shenzhen|Chinese factor(y|ies)|China's factories)\b",
     "categories": {"disaster", "accident", "geopolitical", "labor", "health"},
     "tickers": {"AAPL": 1, "TSLA": 1, "NKE": 1}},
    {"name": "Middle East oil supply", "pattern": r"\b(Strait of Hormuz|Red Sea|Persian Gulf|Iran|Saudi|OPEC|oil tankers?|oil fields?)\b",
     "categories": {"geopolitical", "accident", "disaster"},
     "tickers": {"XOM": -1, "CVX": -1}},
    {"name": "US Gulf Coast energy", "pattern": r"\b(Gulf of Mexico|Gulf of America|Gulf Coast|Louisiana|refiner(y|ies))\b",
     "categories": {"disaster", "accident"},
     "tickers": {"XOM": 1, "CVX": 1}},
    {"name": "Commercial aviation safety", "pattern": r"\b(plane crash|air crash|airliner|passenger jet|FAA|NTSB|midair|aviation)\b",
     "categories": {"accident"},
     "tickers": {"BA": 1}},
    {"name": "Cloud & internet infrastructure", "pattern": r"\b(cloud outage|internet outage|data cent(er|re)s?|undersea cable|submarine cable)\b",
     "categories": {"cyber_outage", "disaster", "accident"},
     "tickers": {"AMZN": 1, "MSFT": 1, "GOOGL": 1}},
    {"name": "Disease outbreak", "pattern": r".",
     "categories": {"health"},
     "tickers": {"PFE": -1, "MRK": -1, "JNJ": -1, "LLY": -1}},
]
