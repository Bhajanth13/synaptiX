"""Intent router: cheap keyword rules first. Swap for a small trained classifier later."""

import re

# Keywords are matched on lowercased text. Queries are pivoted to English when a translator is
# installed; a few native keywords keep routing working without one.
RULES = [
    ("jurisdiction", ["who do i", "whom", "responsible", "complain to", "sarpanch", "mla", "mp ",
                      "panchayat", "power", "whose job", "who handles", "who fixes", "who is in charge",
                      "किससे", "कौन ठीक", "किसकी ज़िम्मेदारी", "ಯಾರಿಗೆ", "ಯಾರ ಜವಾಬ್ದಾರಿ"]),
    ("events", ["when is", "upcoming", "next gram sabha", "camp", "deadline", "election date", "meeting"]),
    ("legal", ["law", "legal", "rights", "notice", "contract", "wage", "land", "loan", "summons",
               "fir", "act", "court", "eligible", "scheme", "कानून", "मज़दूरी", "मजदूरी", "ಕಾನೂನು", "ವೇತನ"]),
    ("jargon", ["what is", "what does", "meaning", "mean", "explain", "define",
                "मतलब", "क्या है", "क्या होता", "ಅರ್ಥ", "ಎಂದರೇನು"]),
]


def route(text: str) -> str:
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    scores = {skill: sum(k in t for k in kws) for skill, kws in RULES}
    best = max(scores, key=scores.get)
    return best if scores[best] else "jargon"
