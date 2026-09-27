"""
PRAG — Biases: what each account curates towards

A bias is what the agent looks for on one account:

  interests      what the LLM scores every reel against ("what this
                 person values" in its prompt)
  watch_words    caption / hashtag words that make the quick rules watch
  skip_words     caption / hashtag words that skip a reel instantly
  creator_hints  parts of account names that suggest a relevant page
  seeds          accounts to follow before the first session (optional)
  persona        score reels by how likely the person described is to
                 engage with them (their tastes and views), not by how
                 informative they are: for studying what the algorithm feeds
                 someone like that

Each account's bias is biases/<account>.json — the panel's Bias tab edits
it — and an account without one uses the News (India) preset, which is
config.py's settings. Changing the interests changes the scorer
fingerprint, so the report only compares reels scored under one bias.
"""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field

from config import INTERESTS, NEWS_CREATOR_HINTS, NEWS_KEYWORDS, SKIP_KEYWORDS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIAS_DIR = os.path.join(PROJECT_ROOT, "biases")
SEEDS_DIR = os.path.join(PROJECT_ROOT, "seeds")
HANDLE = re.compile(r"[A-Za-z0-9._]{1,30}")
MAX_NAME, MAX_INTERESTS, MAX_WORDS, MAX_WORD, MAX_SEEDS = 40, 4000, 300, 40, 60
DEFAULT = "news-india"
LIST_LABELS = {"watch_words": "Watch words", "skip_words": "Skip words", "creator_hints": "Account-name hints"}


def interests_fingerprint(interests: str) -> str:
    return hashlib.sha1(interests.strip().encode()).hexdigest()[:8]


@dataclass
class Bias:
    name: str
    interests: str
    watch_words: list[str] = field(default_factory=list)
    skip_words: list[str] = field(default_factory=list)
    creator_hints: list[str] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    persona: bool = False

    def __post_init__(self) -> None:
        self.interests = self.interests.strip()  # the prompt strips it too, so fingerprints don't change

    @property
    def fingerprint(self) -> str:
        """Short id of the interests (and persona scoring); a model trained for one bias doesn't score another."""
        return interests_fingerprint(self.interests + ("\n[persona]" if self.persona else ""))

    def to_dict(self) -> dict:
        return asdict(self)


def words(value) -> list[str]:
    """Words from comma- or line-separated text (or a list): Latin ones lowercased, '#' dropped, no repeats."""
    items = value if isinstance(value, (list, tuple, set)) else re.split(r"[,\n]", str(value or ""))
    out: list[str] = []
    for item in items:
        word = " ".join(str(item).split()).lstrip("#@")
        word = word.lower() if word.isascii() else word
        if word and word not in out:
            out.append(word)
    return out


def validated(data: dict) -> Bias:
    """A Bias from what someone entered, or ValueError saying what to fix."""
    name = " ".join(str(data.get("name") or "").split())[:MAX_NAME] or "Custom"
    interests = str(data.get("interests") or "").strip()
    if len(interests) < 20:
        raise ValueError("Describe what to look for in at least a sentence.")
    if len(interests) > MAX_INTERESTS:
        raise ValueError(f"Keep \"what to look for\" under {MAX_INTERESTS} characters.")
    lists = {}
    for key, label in LIST_LABELS.items():
        items = words(data.get(key))
        if len(items) > MAX_WORDS:
            raise ValueError(f"{label}: at most {MAX_WORDS}.")
        long = next((w for w in items if len(w) > MAX_WORD), None)
        if long:
            raise ValueError(f"{label}: \"{long[:20]}…\" is too long for one word.")
        lists[key] = items
    seeds = words(data.get("seeds"))
    bad = next((s for s in seeds if not HANDLE.fullmatch(s)), None)
    if bad:
        raise ValueError(f"\"{bad}\" isn't an Instagram handle.")
    if len(seeds) > MAX_SEEDS:
        raise ValueError(f"At most {MAX_SEEDS} accounts to follow.")
    return Bias(name=name, interests=interests, seeds=seeds, persona=bool(data.get("persona")), **lists)


# ── Presets ───────────────────────────────────────────────────────────

def _seed_file(name: str) -> list[str]:
    """Handles from seeds/<name>.txt: one per line, '#' starts a comment."""
    try:
        with open(os.path.join(SEEDS_DIR, name)) as fh:
            return words([line.split("#", 1)[0] for line in fh])
    except OSError:
        return []


_ENTERTAINMENT = sorted(SKIP_KEYWORDS)
_NOT_GAMES = [w for w in _ENTERTAINMENT if w not in {"gaming", "pubg", "bgmi", "freefire", "valorant", "gamer"}]

PRESETS: dict[str, Bias] = {
    "news-india": Bias(
        name="News (India)",
        interests=INTERESTS,
        watch_words=sorted(NEWS_KEYWORDS),
        skip_words=_ENTERTAINMENT,
        creator_hints=list(NEWS_CREATOR_HINTS),
        seeds=_seed_file("news_india.txt"),
    ),
    "tech-ai": Bias(
        name="Tech & AI",
        interests="""
I want to keep up with technology and AI without scrolling for hours.

HIGH value (0.75-1.0): specific, new information, e.g.
- AI models, products and research results, and what they can actually do
- Launches and updates of apps, gadgets and developer tools, with real details
- Big-tech and startup news: funding, acquisitions, layoffs, policy
- Cybersecurity: breaches, scams and vulnerabilities, and what to do about them
- Government rules and decisions about technology and AI

MEDIUM value (0.40-0.74): useful but not new or not specific: explainers,
tutorials, career advice for tech jobs, honest reviews, informed opinion.

LOW value (0.00-0.39): no real information: hype and "top 10 AI tools"
lists with nothing concrete, get-rich-quick, entertainment, memes, comedy,
music, lifestyle, motivational quotes, devotional content, ads and
self-promotion.
""",
        watch_words=["ai", "chatgpt", "openai", "gemini", "claude", "llm", "gpt", "deepseek", "copilot", "startup",
                     "funding", "acquisition", "layoffs", "launch", "launched", "release", "update", "iphone",
                     "android", "apple", "google", "microsoft", "nvidia", "samsung", "chip", "semiconductor",
                     "cybersecurity", "hack", "hacked", "breach", "scam", "privacy", "robot", "robotics", "coding",
                     "developer", "software", "tech", "technology", "gadget", "5g", "quantum"],
        skip_words=_NOT_GAMES,  # game news can be tech news
        creator_hints=["tech", "gadget", "geek", "coder", "coding", "digital", "cyber", "robot"],
    ),
    "money-markets": Bias(
        name="Money & markets",
        interests="""
I want to understand money, markets and the economy without scrolling for hours.

HIGH value (0.75-1.0): specific, new information, e.g.
- Market moves and the news behind them; company results, IPOs, mergers
- RBI, SEBI and government decisions: interest rates, taxes, GST, new rules
- Changes to banking, loans, UPI, insurance, pensions and savings schemes
- Prices that affect people: fuel, gold, food, inflation
- Financial scams and frauds to watch out for

MEDIUM value (0.40-0.74): useful but not new: explainers on tax, investing
and personal finance, analysis and informed opinion on the economy.

LOW value (0.00-0.39): no real information: stock tips and "guaranteed
returns" without evidence, trading-course and app promotions, luxury
lifestyle, motivational quotes, entertainment, memes, devotional content,
ads and self-promotion.
""",
        watch_words=["sensex", "nifty", "stock", "stocks", "shares", "market", "ipo", "rbi", "sebi", "repo", "gst",
                     "tax", "itr", "budget", "inflation", "gdp", "economy", "rupee", "gold", "silver", "petrol",
                     "diesel", "loan", "emi", "bank", "banking", "upi", "mutualfund", "sip", "insurance", "pension",
                     "epfo", "results", "earnings", "scam", "fraud", "बाजार", "शेयर", "टैक्स", "महंगाई", "बैंक",
                     "लोन", "सोना"],
        skip_words=_ENTERTAINMENT,
        creator_hints=["finance", "money", "market", "stock", "invest", "wealth", "capital", "economic",
                       "business", "mint"],
    ),
    "science-health": Bias(
        name="Science & health",
        interests="""
I want to keep up with science, space and health without scrolling for hours.

HIGH value (0.75-1.0): specific, new information, e.g.
- Research findings and discoveries, and who made them
- Space missions and launches (ISRO, NASA and others)
- Health advisories and outbreaks from health authorities; new treatments
  and approvals
- Climate, extreme weather and the environment

MEDIUM value (0.40-0.74): useful but not new: explainers and general
knowledge, how things work, evidence-based health and nutrition advice.

LOW value (0.00-0.39): no real information: miracle cures and health claims
without evidence, supplement and product promotion, fitness influencer
content, pseudoscience, entertainment, memes, motivational quotes,
devotional content and astrology.
""",
        watch_words=["science", "scientists", "research", "discovery", "isro", "nasa", "space", "mission",
                     "satellite", "rocket", "moon", "mars", "chandrayaan", "gaganyaan", "vaccine", "virus",
                     "outbreak", "icmr", "disease", "cancer", "health", "medicine", "doctor", "climate", "earthquake",
                     "cyclone", "heatwave", "flood", "wildlife", "fossil", "dna", "विज्ञान", "अंतरिक्ष", "इसरो",
                     "बीमारी", "स्वास्थ्य"],
        skip_words=_ENTERTAINMENT,
        creator_hints=["science", "space", "health", "doctor", "research", "nature", "isro", "nasa", "physics",
                       "biology"],
    ),
    "persona": Bias(
        name="Persona (fill in)",
        persona=True,
        interests="""
Who this person is: [age, where they live, languages, work, religion, politics, what they care about].

What they watch to the end, like and follow: [the news and opinions they agree with, the pages they trust,
sports, humour, faith, local issues, hobbies: describe the whole person, not just one side of them].

What they scroll past: [content they find boring or disagree with].
""",
    ),
}
DEFAULT_FINGERPRINT = PRESETS[DEFAULT].fingerprint


def preset_for(bias: Bias) -> str | None:
    """The preset whose interests this bias uses, if any."""
    return next((key for key, preset in PRESETS.items() if preset.fingerprint == bias.fingerprint), None)


# ── Each account's bias ───────────────────────────────────────────────

def path_for(account: str) -> str:
    if not HANDLE.fullmatch(account or ""):
        raise ValueError(f"Not an account name: {account!r}")
    return os.path.join(BIAS_DIR, f"{account}.json")


def load(account: str | None) -> Bias:
    """The account's bias: biases/<account>.json, or the default preset."""
    if account and HANDLE.fullmatch(account):
        path = path_for(account)
        try:
            with open(path) as fh:
                return validated(json.load(fh))
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            print(f"⚠️  {path} can't be used ({exc}); using the {PRESETS[DEFAULT].name} bias.")
    return PRESETS[DEFAULT]


def has_own(account: str) -> bool:
    return os.path.exists(path_for(account))


def save(account: str, bias: Bias) -> str:
    path = path_for(account)
    os.makedirs(BIAS_DIR, exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(bias.to_dict(), fh, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return path
