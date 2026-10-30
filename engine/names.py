"""Company names: from official listings to the names headlines actually use."""
import re

# Words dropped to get the name headlines use ("Palantir Technologies" -> "Palantir").
GENERIC_NAME_WORDS = {"technologies", "technology", "systems", "solutions", "holdings", "group", "industries",
                      "international", "enterprises", "brands", "communications", "networks", "software",
                      "platforms", "worldwide", "global", "corporation", "incorporated", "company", "inc", "corp",
                      "manufacturing", "services", "labs", "laboratories", "therapeutics", "pharmaceuticals",
                      "biosciences", "bancorp", "financial", "resources", "energy", "partners", "trust", "co"}
# Short names too common to identify a company on their own.
TOO_COMMON = {"united", "american", "first", "general", "national", "global", "international", "new", "big",
              "target", "block", "square", "gap", "match", "sun", "core", "advanced", "applied", "digital",
              "data", "energy", "health", "bank", "capital", "precision"}


def clean_company_name(name: str) -> str:
    """'Shopify Inc. Class A' -> 'Shopify': drops legal suffixes. "" if too short to match safely."""
    name = re.sub(r",?\s+(Inc\.?|Incorporated|Corporation|Corp\.?|Co\.?|Ltd\.?|plc|Holdings?|"
                  r"Group|Company|Class [A-C]|N\.V\.|S\.A\.)\b.*$", "", name or "", flags=re.I)
    name = re.sub(r"[\s&,]+$", "", name).strip()          # "JPMorgan Chase & Co." -> "JPMorgan Chase"
    return name if len(name) > 2 else ""


def headline_names(company: str) -> list[str]:
    """Names to match in headlines: the full company name plus its short form when that's distinctive."""
    full = company.strip()
    words = full.split()
    while len(words) > 1 and words[-1].lower().strip(".,&") in GENERIC_NAME_WORDS:
        words = words[:-1]
    short = " ".join(words).strip(" ,&")
    names = [full]
    if short and short != full and short.lower() not in TOO_COMMON and len(short) > 2:
        names.append(short)
    return names
