"""
PRAG — Personas: who each account scrolls as

A persona is a person described by a set of biases, and it's what the agent
behaves like on one account:

  interests      who the person is, what they watch to the end, like and
                 follow, and what they scroll past (the LLM scores every
                 reel against this)
  watch_words    caption / hashtag words that make the quick rules watch
  skip_words     caption / hashtag words that skip a reel instantly
  creator_hints  parts of account names that suggest a relevant page
  seeds          accounts to follow before the first session (optional)
  persona        score reels by how likely that person is to watch them to
                 the end and engage (true for every persona the panel saves);
                 false only for the default below, which scores how
                 informative a reel is

Each account's persona is biases/<account>.json (the panel's Persona tab
edits it). An account without one scrolls with the News (India) default,
which is config.py's settings. The presets the tab starts from are
editable: biases/presets.json once changed, BUILT_IN until then. Changing
the persona changes the scorer fingerprint, so the report only compares
reels scored under one persona. (In code a persona is still a Bias.)
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


# ── The default, and the persona presets ─────────────────────────────

def _seed_file(name: str) -> list[str]:
    """Handles from seeds/<name>.txt: one per line, '#' starts a comment."""
    try:
        with open(os.path.join(SEEDS_DIR, name)) as fh:
            return words([line.split("#", 1)[0] for line in fh])
    except OSError:
        return []


def _without(*drop: str) -> list[str]:
    """The default skip words minus some: a persona mustn't instantly skip what it enjoys."""
    return sorted(w for w in SKIP_KEYWORDS if w not in drop)


def _person(text: str) -> str:
    """A persona description: paragraphs kept, each on one line (it's edited in a text box)."""
    return "\n\n".join(" ".join(paragraph.split()) for paragraph in text.strip().split("\n\n"))


_GAMES = ("gaming", "pubg", "bgmi", "freefire", "valorant", "gamer")
_HUMOUR = ("comedy", "funny", "meme", "memes", "standup")
_DEVOTIONAL = ("bhakti", "bhajan", "mahadev", "harharmahadev", "bholenath", "hanumanji", "bajrangbali", "jaishreeram",
               "radheradhe", "radhekrishna")
_ASTROLOGY = ("astrology", "horoscope", "rashifal", "jyotish")

# An account without a persona scrolls with this: config.py's News (India) interests, scored for how informative a
# reel is. Its fingerprint is the one the fallback model was trained for, and older scores were judged under it.
DEFAULT = "news-india"
DEFAULT_BIAS = Bias(
    name="News (India)",
    interests=INTERESTS,
    watch_words=sorted(NEWS_KEYWORDS),
    skip_words=sorted(SKIP_KEYWORDS),
    creator_hints=list(NEWS_CREATOR_HINTS),
    seeds=_seed_file("news_india.txt"),
)
DEFAULT_FINGERPRINT = DEFAULT_BIAS.fingerprint

# What the Persona tab starts from. Each one is a whole person, not a single interest, so the feed curates towards
# what someone like them would really be shown. The panel edits a copy (biases/presets.json); these stay as they are.
BUILT_IN: dict[str, Bias] = {
    "news-follower": Bias(
        name="News follower (India)",
        persona=True,
        interests=_person("""
Who this person is: a 30-year-old office worker in Delhi who follows the news every day, in Hindi and English.
Cares about politics and where the country is going without being loyal to any party, and about prices, jobs,
exams and safety.

What they watch to the end, like and follow: breaking news and current affairs from national news channels,
newspapers and reporters; government decisions and new schemes; the economy, prices and the markets; court
rulings; weather and safety alerts; space missions and science; explainers about how things work; cricket results.

What they scroll past: memes, pranks, dance trends, celebrity gossip, astrology, motivational quotes, and anyone
selling something.
"""),
        watch_words=sorted(NEWS_KEYWORDS),
        skip_words=sorted(SKIP_KEYWORDS),
        creator_hints=list(NEWS_CREATOR_HINTS),
        seeds=_seed_file("news_india.txt"),
    ),
    "right-leaning": Bias(
        name="Right-leaning voter (India)",
        persona=True,
        interests=_person("""
Who this person is: a 38-year-old shop owner in Lucknow who speaks Hindi, is a practising Hindu, and supports the
BJP and Prime Minister Narendra Modi. Proud of the armed forces and of India's development; family and tradition
matter a lot to them.

What they watch to the end, like and follow: speeches and interviews of the Prime Minister and BJP leaders; news
channels and commentators who share their views; the army and national security; new highways, railways and
development projects; devotional content and temple festivals; cricket; Hindi comedy and family skits.

What they scroll past: speeches and commentary by opposition leaders, content that criticises the government or
mocks religion, and English-language lifestyle and fashion content.
"""),
        watch_words=["modi", "narendra modi", "bjp", "yogi", "amit shah", "army", "indian army", "defence", "military",
                     "hindu", "sanatan", "ram mandir", "ayodhya", "bharat", "vande bharat", "development",
                     "infrastructure", "cricket", "मोदी", "भाजपा", "योगी", "सेना", "सनातन", "हिंदू", "राम मंदिर"],
        skip_words=["makeup", "skincare", "cosmetics", "lipstick", "glowup", "fashion", "ootd", "outfit"],
        creator_hints=["bjp", "modi", "bharat", "sanatan", "hindu", "yogi"],
    ),
    "left-leaning": Bias(
        name="Left-leaning voter (India)",
        persona=True,
        interests=_person("""
Who this person is: a 27-year-old postgraduate student in Kolkata who speaks Bengali, Hindi and English, and
supports the Congress and other opposition parties. Cares about jobs, prices, farmers, civil liberties and
minority rights; not religious.

What they watch to the end, like and follow: speeches and interviews of Rahul Gandhi and other opposition leaders;
independent journalists and commentators who question the government; political satire and stand-up comedy;
reports on unemployment, inflation, farmers and protests; climate and the environment; indie music; football and
cricket.

What they scroll past: pro-government and nationalist commentary, religious and devotional content, astrology,
and get-rich-quick advice.
"""),
        watch_words=["rahul gandhi", "congress", "opposition", "democracy", "constitution", "unemployment",
                     "inflation", "farmers", "protest", "satire", "standup", "climate", "football", "राहुल गांधी",
                     "कांग्रेस", "बेरोजगारी", "महंगाई", "किसान", "संविधान"],
        skip_words=sorted(_DEVOTIONAL + _ASTROLOGY),
        creator_hints=["congress", "satire", "comedy", "standup"],
    ),
    "tech": Bias(
        name="Tech enthusiast",
        persona=True,
        interests=_person("""
Who this person is: a 24-year-old software engineer in Bengaluru who mostly watches in English and follows AI,
gadgets and startups closely.

What they watch to the end, like and follow: new AI models and tools and what they can actually do; phone and
gadget launches, and reviews with real specs; coding tips; startup funding and big-tech news; cybersecurity scams
and breaches; gaming news; clever tech memes; Formula 1.

What they scroll past: politics, devotional content, astrology, dance trends and celebrity gossip.
"""),
        watch_words=["ai", "chatgpt", "openai", "gemini", "claude", "llm", "gpt", "deepseek", "copilot", "startup",
                     "funding", "acquisition", "layoffs", "launch", "launched", "release", "update", "iphone",
                     "android", "apple", "google", "microsoft", "nvidia", "samsung", "chip", "semiconductor",
                     "cybersecurity", "hack", "hacked", "breach", "scam", "privacy", "robot", "robotics", "coding",
                     "developer", "software", "tech", "technology", "gadget", "5g", "quantum", "f1"],
        skip_words=_without(*_GAMES, *_HUMOUR),
        creator_hints=["tech", "gadget", "geek", "coder", "coding", "digital", "cyber", "robot"],
    ),
    "investor": Bias(
        name="Investor",
        persona=True,
        interests=_person("""
Who this person is: a 42-year-old chartered accountant in Mumbai who invests in stocks and mutual funds, and
watches in English and Hindi.

What they watch to the end, like and follow: market moves and company results; RBI and SEBI decisions; tax, GST
and budget changes; personal-finance explainers; business news and interviews with founders; property and gold
prices.

What they scroll past: memes, dance trends, devotional content, celebrity gossip, and trading courses that promise
guaranteed returns.
"""),
        watch_words=["sensex", "nifty", "stock", "stocks", "shares", "market", "ipo", "rbi", "sebi", "repo", "gst",
                     "tax", "itr", "budget", "inflation", "gdp", "economy", "rupee", "gold", "silver", "loan", "emi",
                     "bank", "banking", "upi", "mutualfund", "sip", "insurance", "pension", "results", "earnings",
                     "बाजार", "शेयर", "टैक्स", "महंगाई", "बैंक", "सोना"],
        skip_words=sorted(SKIP_KEYWORDS),
        creator_hints=["finance", "money", "market", "stock", "invest", "wealth", "capital", "economic", "business"],
    ),
    "science-health": Bias(
        name="Science & health reader",
        persona=True,
        interests=_person("""
Who this person is: a 33-year-old doctor in Pune who watches in English and Marathi and trusts evidence over hype.

What they watch to the end, like and follow: research findings and discoveries; ISRO and NASA missions; health
advisories and new treatments; evidence-based fitness and nutrition; climate and wildlife; explainers about how the
body and the universe work.

What they scroll past: miracle cures, supplement promotions, pseudoscience and astrology, politics and memes.
"""),
        watch_words=["science", "scientists", "research", "discovery", "isro", "nasa", "space", "mission",
                     "satellite", "rocket", "moon", "mars", "vaccine", "virus", "outbreak", "icmr", "disease",
                     "cancer", "health", "medicine", "doctor", "nutrition", "climate", "wildlife", "fossil", "dna",
                     "विज्ञान", "अंतरिक्ष", "इसरो", "बीमारी", "स्वास्थ्य"],
        skip_words=_without("fitness", "workout", "gym"),
        creator_hints=["science", "space", "health", "doctor", "research", "nature", "isro", "nasa", "physics",
                       "biology"],
    ),
    "blank": Bias(
        name="Blank persona (fill in)",
        persona=True,
        interests=_person("""
Who this person is: [age, where they live, languages, work, religion, politics, what they care about].

What they watch to the end, like and follow: [the news and opinions they agree with, the pages they trust,
sports, humour, faith, local issues, hobbies: describe the whole person, not just one side of them].

What they scroll past: [content they find boring or disagree with].
"""),
    ),
}

PRESET_KEY = re.compile(r"[a-z0-9-]{1,48}")


def presets_path() -> str:
    return os.path.join(BIAS_DIR, "presets.json")


def presets() -> dict[str, Bias]:
    """The presets the Persona tab offers: the edited set in biases/presets.json, or the built-in ones."""
    try:
        with open(presets_path()) as fh:
            stored = json.load(fh)
    except FileNotFoundError:
        return dict(BUILT_IN)
    except (OSError, ValueError) as exc:
        print(f"⚠️  {presets_path()} can't be used ({exc}); using the built-in presets.")
        return dict(BUILT_IN)
    out: dict[str, Bias] = {}
    for key, data in (stored.items() if isinstance(stored, dict) else []):
        try:
            if PRESET_KEY.fullmatch(key) and isinstance(data, dict):
                out[key] = validated({**data, "persona": True})
        except ValueError:
            continue  # one broken preset doesn't hide the others
    return out


def _write_presets(items: dict[str, Bias]) -> None:
    os.makedirs(BIAS_DIR, exist_ok=True)
    path = presets_path()
    with open(path + ".tmp", "w") as fh:
        json.dump({key: bias.to_dict() for key, bias in items.items()}, fh, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def save_preset(bias: Bias, key: str | None = None) -> str:
    """Keep a persona as a preset: a new one named after it (no key), or in place of preset `key`."""
    items = presets()
    if key is None:
        base = re.sub(r"[^a-z0-9]+", "-", bias.name.lower()).strip("-")[:40] or "persona"
        key, n = base, 2
        while key in items:
            key, n = f"{base}-{n}", n + 1
    elif key not in items:
        raise ValueError("That preset isn't there any more: save it as a new one.")
    items[key] = Bias(**{**bias.to_dict(), "persona": True})
    _write_presets(items)
    return key


def delete_preset(key: str) -> None:
    items = presets()
    if key not in items:
        raise ValueError("That preset isn't there any more.")
    del items[key]
    _write_presets(items)


def restore_presets() -> None:
    """Back to the built-in presets (the edited ones are dropped)."""
    try:
        os.remove(presets_path())
    except FileNotFoundError:
        pass


def preset_for(bias: Bias) -> str | None:
    """The preset this bias is, if any."""
    return next((key for key, preset in presets().items() if preset.fingerprint == bias.fingerprint), None)


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
            print(f"⚠️  {path} can't be used ({exc}); using the {DEFAULT_BIAS.name} bias.")
    return DEFAULT_BIAS


def has_own(account: str) -> bool:
    return os.path.exists(path_for(account))


def save(account: str, bias: Bias) -> str:
    path = path_for(account)
    os.makedirs(BIAS_DIR, exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(bias.to_dict(), fh, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return path
