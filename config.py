"""
FeedLab — Configuration

Everything the agent is tuned by lives here. INTERESTS is the one to edit
first: it is the only description of "valuable" the scorer ever sees, so
the account's feed is steered toward whatever it describes.
"""

# ── Local LLM (LM Studio, OpenAI-compatible) ──────────────────────────

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_STUDIO_MODELS_URL = "http://127.0.0.1:1234/v1/models"
LM_STUDIO_CLI = "~/.lmstudio/bin/lms"   # used to start the server / load the model if it's down
LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"
LLM_CONTEXT_LENGTH = 4096


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
LIKE_THRESHOLD = 0.75                  # at or above: watch fully and like
SAVE_THRESHOLD = 0.85                  # at or above: also save

SKIP_DWELL = (1.2, 3.0)                # seconds on screen before skipping
AD_DWELL = (1.0, 2.5)
WATCH_FRACTION = (0.35, 0.70)          # of the reel's duration
WATCH_DWELL_BOUNDS = (5.0, 25.0)
FULL_WATCH_FRACTION = (0.90, 1.00)
FULL_WATCH_BOUNDS = (8.0, 75.0)
MAX_DECISION_WAIT = 6.0                # a reel still unscored after this long is skipped

EXPLORATION_RATE = 0.10                # chance to watch a borderline reel / score a gated one
EXPLORE_MIN_SCORE = 0.25               # borderline = between this and WATCH_THRESHOLD

FOLLOW_MIN_HIGH_REELS = 3              # follow a creator after this many reels >= LIKE_THRESHOLD
MAX_FOLLOWS_PER_SESSION = 3
CREATOR_SKIP_MIN_SEEN = 4              # skip a creator's reels unscored once they've had this many...
CREATOR_SKIP_MAX_SCORE = 0.30          # ...and none scored above this


# ── Sessions & verification ───────────────────────────────────────────

SESSION_MINUTES = 40
CURATION_WINDOW = 100                  # judge the feed on the latest N organic reels
CURATION_TARGET_SHARE = 0.50           # share scoring >= WATCH_THRESHOLD to call it curated
