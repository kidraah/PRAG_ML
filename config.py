"""
PRAG — Configuration

Everything the agent is tuned by lives here. INTERESTS is the one to edit
first: it is the only description of "valuable" the scorer ever sees, so
the account's feed is steered toward whatever it describes.
"""

# ── Local LLM (LM Studio, OpenAI-compatible) ──────────────────────────

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_STUDIO_MODELS_URL = "http://127.0.0.1:1234/v1/models"
LM_STUDIO_CLI = "~/.lmstudio/bin/lms"   # used to start the server / load the model if it's down
LLM_MODEL_NAME = "qwen/qwen3.5-9b"
LLM_CONTEXT_LENGTH = 4096
LLM_THINKING = False                   # False: reasoning models (Qwen3.5, ...) answer directly instead of thinking first
# Who judges reels in a session: "llm" (LM Studio scores and writes, the system as it is) or "jev" (Jev, through
# OpenRouter, judges each reel in ~0.4 s and the local LLM writes the text in the background; a session doesn't
# start when Jev can't be reached). "jev" is still being finished and tested: keep "llm" until then.
SCORER = "llm"
SCORING_WORKERS = 2                    # reels scored at once; LM Studio must allow this many parallel predictions


# ── What counts as valuable ───────────────────────────────────────────

INTERESTS = """
I want to catch valuable news and information that shows up on Instagram Reels,
often before (or instead of) anywhere else, without scrolling for hours.

HIGH value (0.75-1.0): specific, factual, new or time-sensitive information, e.g.
- Breaking news and current events (India and world), including politics,
  policy and government decisions, reported as news whatever the stance
- New government schemes, rule changes, deadlines, exams, jobs, scholarships
- Economy, markets, prices, taxes, personal-finance rules
- Technology, AI, science and startup developments
- Safety alerts, scam warnings, local incidents

MEDIUM value (0.40-0.74): informative but not new or not specific: explainers,
general knowledge, history, commentary or opinion on current events, how-tos.

LOW value (0.00-0.39): no real information: entertainment, memes, comedy,
music, dance, lifestyle, fashion, food, fitness, motivational quotes,
devotional content and rituals, astrology, ads and self-promotion.
"""

# Instant skip (no LLM call) when a caption word or hashtag matches one of
# these. Latin-script, lowercase, whole words. A fraction of gated reels is
# still scored (EXPLORATION_RATE) so the report can show whether the gate
# is throwing away anything valuable.
SKIP_KEYWORDS = {
    # entertainment & lifestyle
    "cooking", "recipe", "foodie", "baking", "yummy", "tasty",
    "gaming", "pubg", "bgmi", "freefire", "valorant", "gamer",
    "makeup", "beauty", "skincare", "cosmetics", "lipstick", "glowup",
    "fashion", "ootd", "outfit", "dance", "dancer", "choreography",
    "bollywood", "song", "songs", "lyrics", "comedy", "funny", "meme", "memes",
    "standup", "fitness", "workout", "gym", "bodybuilding",
    # devotional & astrology (remove these if you want that content scored)
    "bhakti", "bhajan", "mahadev", "harharmahadev", "bholenath", "hanumanji",
    "bajrangbali", "jaishreeram", "radheradhe", "radhekrishna",
    "astrology", "horoscope", "rashifal", "jyotish",
}


# Quick rules (main.py --rules): a reel is watched when its account is one
# this account follows, or its caption / hashtags contain one of these.
# Latin words match whole words (case-insensitive); Devanagari match anywhere.
# Avoid ambiguous words: "court" is also a tennis court, and "सरकार" / "sarkar" is
# also a devotional title for a deity.
NEWS_KEYWORDS = {
    # English / Hinglish
    "news", "breaking", "alert", "update", "updates", "government", "govt", "yojana",
    "scheme", "policy", "rbi", "sebi", "gst", "tax", "budget", "election", "supremecourt", "highcourt", "police",
    "arrest", "imd", "exam", "result", "recruitment", "bharti", "vacancy", "jobs", "upsc", "ssc",
    "neet", "jee", "cbse", "scholarship", "price", "petrol", "diesel", "inflation", "economy",
    "sensex", "nifty", "isro", "scam", "fraud", "cybercrime", "parliament", "minister",
    "notification", "deadline", "warning", "accident", "flood", "earthquake", "cyclone", "heatwave",
    # Hindi
    "खबर", "समाचार", "ब्रेकिंग", "अलर्ट", "चेतावनी", "केंद्र सरकार", "राज्य सरकार", "सरकारी", "योजना",
    "नियम", "बजट", "टैक्स",
    "चुनाव", "अदालत", "पुलिस", "गिरफ्तार", "मौसम", "परीक्षा", "रिजल्ट", "भर्ती", "नौकरी",
    "छात्रवृत्ति", "महंगाई", "पेट्रोल", "धोखाधड़ी", "साइबर", "ठगी", "हादसा", "बाढ़", "भूकंप",
    "मंत्री", "संसद", "फैसला",
}
# Parts of an account name that suggest a news page ("nationupdate3", "aajtaknews").
NEWS_CREATOR_HINTS = ("news", "khabar", "samachar", "update", "times", "today", "express", "report")


# ── Perception ────────────────────────────────────────────────────────

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
WHISPER_LANGUAGES = ("hi", "en")       # accepted auto-detected languages
WHISPER_FALLBACK_LANGUAGE = "hi"       # re-transcribe with this when detection lands elsewhere (Hindi is often detected as Urdu)
MAX_TRANSCRIBE_SECONDS = 90
TRANSCRIBE_MUSIC_TRACKS = False        # licensed songs only yield lyrics

USE_COVER_OCR = True                   # read headline text off the cover image
OCR_LANGUAGES = ["hi", "en"]
OCR_MIN_CONFIDENCE = 0.3


# ── Engagement policy (value_score is 0-1) ────────────────────────────

WATCH_THRESHOLD = 0.40                 # below: skip
LIKE_THRESHOLD = 0.75                  # at or above: watch fully, maybe like
SAVE_THRESHOLD = 0.85                  # at or above: maybe also save

# People like only some of what they enjoy, so a like / save is a chance
# that rises with the score (at its threshold, at a perfect 1.0), and
# both are capped per session.
LIKE_PROBABILITY = (0.35, 0.85)
SAVE_PROBABILITY = (0.30, 0.70)
MAX_LIKES_PER_SESSION = 20
MAX_SAVES_PER_SESSION = 5

REWATCH_THRESHOLD = 0.85               # the very best reels...
REWATCH_PROBABILITY = 0.5              # ...are played through again this often (the player loops them)
REWATCH_FACTOR = (1.3, 2.0)            # dwell as a multiple of the reel's duration
REWATCH_MAX_SECONDS = 120

SKIP_DWELL = (1.2, 3.0)                # seconds on screen before skipping
FLICK_DWELL = (0.6, 1.4)               # skip already decided before the reel appeared (preloaded feed data)
AD_DWELL = (1.0, 2.5)
OBSERVE_DWELL = (2.5, 5.0)             # --observe: every reel stays this long; nothing is scored or engaged with
WATCH_FRACTION = (0.35, 0.70)          # of the reel's duration
WATCH_DWELL_BOUNDS = (5.0, 25.0)
FULL_WATCH_FRACTION = (0.90, 1.00)
FULL_WATCH_BOUNDS = (8.0, 75.0)
MAX_DECISION_WAIT = 3.5                # a reel still unscored after this long is skipped (and scored later for the report)...
LATE_SCORE_WAIT = 25.0                 # ...unless the quick rules call it news-like: it keeps playing up to this long for its score

EXPLORATION_RATE = 0.10                # chance to watch a borderline reel / score a gated one
EXPLORE_MIN_SCORE = 0.25               # borderline = between this and WATCH_THRESHOLD

FOLLOW_MIN_HIGH_REELS = 3              # follow a creator after this many reels >= LIKE_THRESHOLD
MAX_FOLLOWS_PER_SESSION = 3
CREATOR_SKIP_MIN_SEEN = 4              # skip a creator's reels unscored once they've had this many...
CREATOR_SKIP_MAX_SCORE = 0.30          # ...and none scored above this


# ── Phase 2: stories, verification, digest ────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # small, multilingual, not an LLM
STORY_DISTANCE = 0.45                  # reels closer than this (cosine distance of topic + summary) are one story
DIGEST_HOURS = 24                      # the digest covers what the agent saw in this many hours
VERIFY_DAYS = 3                        # look for news coverage at least this far back...
VERIFY_MAX_DAYS = 30                   # ...or back to the reel's posting date, up to this
VERIFY_SIMILARITY = 0.60               # a headline covers the story if its meaning is this close to the summary


# ── Sessions & verification ───────────────────────────────────────────

SESSION_MINUTES = 40
DAILY_MAX_MINUTES = 120                # daily.py stops scrolling an account after this many minutes a day
MIN_BATTERY = 30                       # daily.py skips the session on battery below this % (the digest still updates)
ENGAGEMENT_MODES = {                   # daily.py --engagement
    "live": "likes, saves and follows",
    "dry": "watch time only, no likes, saves or follows",
    "mixed": "live for the day's first session, dry runs after",
}
DAILY_ENGAGEMENT = "mixed"             # daily.py's default engagement mode
CURATION_WINDOW = 100                  # judge the feed on the latest N organic reels
CURATION_TARGET_SHARE = 0.50           # share scoring >= WATCH_THRESHOLD to call it curated
