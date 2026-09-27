"""
PRAG — Labelled Reels as a Dataset

Every scored impression is a labelled example: the reel's text (caption,
cover text, speech transcript) and its value score. This module turns the
impressions log into one example per reel, labelled by one scorer (by
default the review labels, so the labeller is consistent).
"""

import sqlite3
from dataclasses import dataclass

from config import WATCH_THRESHOLD

EXAMPLES_SQL = """
SELECT i.code, i.creator, i.caption, i.cover_text, i.transcript, i.value_score, i.rule_score, s.account
FROM impressions i JOIN sessions s ON s.id = i.session_id
WHERE s.scorer = ? AND i.score_basis = 'score' AND i.value_score IS NOT NULL AND i.is_ad = 0
ORDER BY i.id
"""


@dataclass
class Example:
    code: str
    creator: str
    account: str
    text: str
    value_score: float
    rule_score: float | None = None

    @property
    def informative(self) -> bool:
        return self.value_score >= WATCH_THRESHOLD


def reel_text(caption: str | None, cover_text: str | None, transcript: str | None, limit: int = 1500) -> str:
    """What the model reads: the same three texts the LLM scorer gets."""
    parts = [p.strip() for p in (caption, cover_text, transcript) if p and p.strip()]
    return " | ".join(parts)[:limit]


def load_examples(conn: sqlite3.Connection, scorer: str = "review") -> list[Example]:
    """One example per reel (its highest score if it was labelled twice)."""
    conn.row_factory = sqlite3.Row
    examples: dict[str, Example] = {}
    for row in conn.execute(EXAMPLES_SQL, (scorer,)):
        current = examples.get(row["code"])
        if current is None or row["value_score"] > current.value_score:
            examples[row["code"]] = Example(
                code=row["code"], creator=row["creator"] or "", account=row["account"],
                text=reel_text(row["caption"], row["cover_text"], row["transcript"]),
                value_score=row["value_score"],
                rule_score=row["rule_score"] if current is None else current.rule_score or row["rule_score"],
            )
    return list(examples.values())
