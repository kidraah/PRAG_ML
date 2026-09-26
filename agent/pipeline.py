"""
FeedLab — Lookahead Scoring Pipeline

Scores reels in the background, in feed order, as soon as their JSON
arrives — usually before they reach the screen — so the browser loop can
act the moment a reel appears. The reel on screen always jumps the queue;
reels that scrolled past unscored are back-filled when nothing upcoming
is waiting, so every impression still gets a score for the report.

Per reel: instant gates (ad, creator memory, keyword gate) → in parallel,
the audio-only track → Whisper and the cover image → OCR → the local LLM.
MLX (Whisper) and torch (EasyOCR) each get one dedicated thread.
"""

import asyncio
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import aiohttp

from agent.creators import CreatorMemory
from automation.feed_capture import FeedCatalog, ReelMeta
from config import EXPLORATION_RATE, TRANSCRIBE_MUSIC_TRACKS, USE_COVER_OCR
from perception import ocr, transcriber
from perception.heuristics import keyword_gate
from perception.local_evaluator import ScoringError, evaluate_value_async
from perception.media import download, new_session


@dataclass
class ScoreResult:
    code: str
    basis: str                          # score | ad | keyword_gate | creator_prior | llm_error
    value_score: float | None = None
    is_informational: bool | None = None
    topic: str = ""
    summary: str = ""
    reasoning: str = ""
    transcript: str = ""
    transcript_language: str | None = None
    cover_text: str = ""
    gate_keyword: str = ""
    gate_audit: bool = False            # a gate would have skipped it; scored anyway to audit the gate
    error: str | None = None
    seconds: float = 0.0

    def score_fields(self) -> dict:
        """Columns for the impressions table."""
        return {
            "score_basis": self.basis,
            "value_score": self.value_score,
            "is_informational": None if self.is_informational is None else int(self.is_informational),
            "topic": self.topic,
            "summary": self.summary,
            "reasoning": self.reasoning,
            "transcript": self.transcript,
            "transcript_language": self.transcript_language,
            "cover_text": self.cover_text,
            "gate_keyword": self.gate_keyword,
            "gate_audit": int(self.gate_audit),
            "error": self.error,
            "score_seconds": round(self.seconds, 2),
        }


class ScoringPipeline:
    def __init__(
        self,
        catalog: FeedCatalog,
        creators: CreatorMemory,
        *,
        rng: random.Random,
        user_agent: str,
        on_result: Callable[[ReelMeta, ScoreResult], None] | None = None,
        backfill_window: int = 8,
    ) -> None:
        self.catalog = catalog
        self.creators = creators
        self.rng = rng
        self.on_result = on_result
        self.backfill_window = backfill_window
        self._user_agent = user_agent

        self._results: dict[str, ScoreResult] = {}
        self._done: dict[str, asyncio.Event] = {}
        self._queued: set[str] = set()
        self._running: str | None = None
        self._current_code: str | None = None
        self._current_index = -1
        self._wake = asyncio.Event()

        self._mlx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._ocr = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
        self._tmp = tempfile.TemporaryDirectory(prefix="feedlab-")
        self._media: aiohttp.ClientSession | None = None
        self._llm: aiohttp.ClientSession | None = None

    # ── Lifecycle ──

    async def warm_up(self) -> None:
        """Load Whisper, OCR and the LLM so the first reels aren't waiting on model loads."""
        loop = asyncio.get_running_loop()
        self._open_sessions()
        loads = [loop.run_in_executor(self._mlx, transcriber.warm_up)]
        if USE_COVER_OCR:
            loads.append(loop.run_in_executor(self._ocr, ocr.warm_up))
        await asyncio.gather(*loads)
        await evaluate_value_async(self._llm, ReelMeta(code="warmup", caption="Warm-up request."))

    async def close(self) -> None:
        for session in (self._media, self._llm):
            if session is not None:
                await session.close()
        self._mlx.shutdown(wait=False, cancel_futures=True)
        self._ocr.shutdown(wait=False, cancel_futures=True)
        self._tmp.cleanup()

    def _open_sessions(self) -> None:
        if self._media is None:
            self._media = new_session(self._user_agent)
        if self._llm is None:
            self._llm = aiohttp.ClientSession()

    # ── Queue ──

    def enqueue(self, meta: ReelMeta) -> None:
        """Queue a reel for scoring (FeedCatalog calls this for each new feed reel)."""
        if meta.code in self._results or meta.code == self._running:
            return
        self._queued.add(meta.code)
        self._done.setdefault(meta.code, asyncio.Event())
        self._wake.set()

    def set_current(self, code: str | None, meta: ReelMeta | None) -> None:
        """Tell the pipeline which reel is on screen; it is scored next if still pending."""
        self._current_code = code
        if meta is not None and meta.feed_index is not None:
            self._current_index = meta.feed_index
        if meta is not None:
            self.enqueue(meta)

    def result(self, code: str | None) -> ScoreResult | None:
        return self._results.get(code) if code else None

    async def wait_for(self, code: str, timeout: float) -> ScoreResult | None:
        """The reel's result, waiting up to `timeout` seconds; None if it isn't ready."""
        if code in self._results:
            return self._results[code]
        event = self._done.setdefault(code, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout=max(0.0, timeout))
        except asyncio.TimeoutError:
            return None
        return self._results.get(code)

    def _pick_next(self) -> ReelMeta | None:
        if self._current_code in self._queued:
            return self.catalog.get(self._current_code)
        metas = [m for m in (self.catalog.get(c) for c in self._queued) if m is not None]
        upcoming = [m for m in metas if m.feed_index is not None and m.feed_index > self._current_index]
        if upcoming:
            return min(upcoming, key=lambda m: m.feed_index)
        recent = [
            m for m in metas
            if m.feed_index is not None and m.feed_index >= self._current_index - self.backfill_window
        ]
        if recent:
            return max(recent, key=lambda m: m.feed_index)
        self._queued.clear()  # everything left scrolled out of the back-fill window
        return None

    async def run(self) -> None:
        """Worker loop; runs until cancelled."""
        self._open_sessions()
        while True:
            meta = self._pick_next()
            if meta is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            self._queued.discard(meta.code)
            self._running = meta.code
            try:
                result = await self._score(meta)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — never let one reel kill the worker
                result = ScoreResult(meta.code, basis="llm_error", error=f"pipeline: {exc}")
            finally:
                self._running = None
            self._results[meta.code] = result
            self._done.setdefault(meta.code, asyncio.Event()).set()
            if result.basis == "score":
                self.creators.record(meta.creator, meta.code, result.value_score)
            if self.on_result is not None:
                self.on_result(meta, result)

    # ── Scoring ──

    async def _score(self, meta: ReelMeta) -> ScoreResult:
        started = time.monotonic()
        if meta.is_ad:
            return ScoreResult(meta.code, basis="ad")

        gate, keyword = None, ""
        if self.creators.should_skip(meta.creator):
            gate = "creator_prior"
        else:
            keyword = keyword_gate(meta.caption, meta.hashtags) or ""
            gate = "keyword_gate" if keyword else None
        if gate and self.rng.random() >= EXPLORATION_RATE:
            return ScoreResult(meta.code, basis=gate, gate_keyword=keyword, seconds=time.monotonic() - started)

        transcript, language, cover_text, problems = await self._perceive(meta)
        base = dict(
            transcript=transcript, transcript_language=language, cover_text=cover_text,
            gate_keyword=keyword, gate_audit=bool(gate),
        )
        try:
            judgement = await evaluate_value_async(self._llm, meta, transcript, language, cover_text)
        except ScoringError as exc:
            problems.append(str(exc))
            return ScoreResult(
                meta.code, basis="llm_error", error="; ".join(problems),
                seconds=time.monotonic() - started, **base,
            )
        return ScoreResult(
            meta.code,
            basis="score",
            value_score=judgement.value_score,
            is_informational=judgement.is_informational,
            topic=judgement.topic,
            summary=judgement.summary,
            reasoning=judgement.reasoning,
            error="; ".join(problems) or None,
            seconds=time.monotonic() - started,
            **base,
        )

    async def _perceive(self, meta: ReelMeta) -> tuple[str, str | None, str, list[str]]:
        """(transcript, language, cover_text, problems) — failures degrade to empty text."""
        loop = asyncio.get_running_loop()
        problems: list[str] = []

        async def speech() -> tuple[str, str | None]:
            if not (meta.audio_url and meta.has_audio):
                return "", None
            if meta.audio_type == "music" and not TRANSCRIBE_MUSIC_TRACKS:
                return "", None
            path = os.path.join(self._tmp.name, f"{meta.code}.m4a")
            try:
                await download(self._media, meta.audio_url, path)
                result = await loop.run_in_executor(self._mlx, transcriber.transcribe, path, meta.duration)
                return result.text, result.language
            except Exception as exc:  # noqa: BLE001
                problems.append(f"transcription: {exc}")
                return "", None
            finally:
                _remove(path)

        async def cover() -> str:
            if not (USE_COVER_OCR and meta.cover_url):
                return ""
            path = os.path.join(self._tmp.name, f"{meta.code}.jpg")
            try:
                await download(self._media, meta.cover_url, path)
                return await loop.run_in_executor(self._ocr, ocr.extract_text_from_frame, path)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"cover OCR: {exc}")
                return ""
            finally:
                _remove(path)

        (transcript, language), cover_text = await asyncio.gather(speech(), cover())
        return transcript, language, cover_text, problems


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
