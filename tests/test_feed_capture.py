"""Tests for automation/feed_capture.py — run with `python -m unittest discover -s tests`."""

import asyncio
import json
import unittest

from automation.feed_capture import (
    FeedCatalog,
    extract_hashtags,
    find_media,
    parse_body,
    parse_dash_manifest,
    parse_iso_duration,
    parse_media,
)
from tests.fixtures import (
    ads_page, embedded_script, feed_page, make_manifest, make_media, timeline_page,
)


class ParseBodyTests(unittest.TestCase):
    def test_strips_anti_hijacking_prefix(self):
        self.assertEqual(parse_body('for (;;);{"a": 1}'), [{"a": 1}])

    def test_newline_delimited_documents(self):
        self.assertEqual(parse_body('{"a": 1}\n\n{"b": 2}\nnot json'), [{"a": 1}, {"b": 2}])

    def test_garbage_and_empty(self):
        self.assertEqual(parse_body(""), [])
        self.assertEqual(parse_body("<html>"), [])


class FindMediaTests(unittest.TestCase):
    def test_sources(self):
        doc = {"x": [feed_page(make_media("F1")), ads_page(make_media("A1", product_type="ad")),
                     timeline_page(make_media("T1"))]}
        found = {m["code"]: source for m, source in find_media(doc)}
        self.assertEqual(found, {"F1": "feed", "A1": "ad", "T1": "other"})

    def test_embedded_first_batch(self):
        doc = json.loads(embedded_script(make_media("E1"), make_media("E2")))
        self.assertEqual([(m["code"], s) for m, s in find_media(doc)], [("E1", "feed"), ("E2", "feed")])


class ParseMediaTests(unittest.TestCase):
    def test_original_audio_reel(self):
        meta = parse_media(make_media(
            "C1", caption="सरकार की नई योजना #मोदी_सरकार #News, #news!", following=True), "feed")
        self.assertEqual(meta.creator, "news_creator")
        self.assertTrue(meta.creator_verified)
        self.assertTrue(meta.creator_following)
        self.assertEqual(meta.hashtags, ["मोदी_सरकार", "news"])
        self.assertEqual(meta.audio_type, "original")
        self.assertEqual(meta.audio_url, "https://cdn.example/audio.mp4")  # lowest bitrate
        self.assertAlmostEqual(meta.duration, 21.2)
        self.assertEqual(meta.cover_url, "https://cdn.example/cover-480.jpg")
        self.assertEqual((meta.like_count, meta.comment_count, meta.repost_count), (1234, 56, 7))
        self.assertEqual(meta.taken_at, 1783703326)
        self.assertFalse(meta.is_ad)
        self.assertEqual(meta.url, "https://www.instagram.com/reels/C1/")

    def test_music_track_and_hidden_counts(self):
        meta = parse_media(make_media("C2", audio="music", audio_title="Some Song", counts_hidden=True), "feed")
        self.assertEqual((meta.audio_type, meta.audio_title), ("music", "Some Song"))
        self.assertIsNone(meta.like_count)
        self.assertEqual(meta.caption, "")

    def test_ad_by_source_or_product_type(self):
        self.assertTrue(parse_media(make_media("A1"), "ad").is_ad)
        self.assertTrue(parse_media(make_media("A2", product_type="ad"), "other").is_ad)


class DashTests(unittest.TestCase):
    def test_iso_durations(self):
        self.assertAlmostEqual(parse_iso_duration("PT47.949207S"), 47.949207)
        self.assertAlmostEqual(parse_iso_duration("PT1M2.5S"), 62.5)
        self.assertEqual(parse_iso_duration("PT1H"), 3600)
        self.assertIsNone(parse_iso_duration("PT"))
        self.assertIsNone(parse_iso_duration("garbage"))

    def test_manifest_without_audio(self):
        self.assertEqual(parse_dash_manifest(make_manifest(9.5, audio_url=None)), (None, 9.5))

    def test_broken_manifest(self):
        self.assertEqual(parse_dash_manifest("<MPD"), (None, None))
        self.assertEqual(parse_dash_manifest(""), (None, None))

    def test_hashtags_dedupe_and_trailing_punctuation(self):
        self.assertEqual(extract_hashtags("#A #a. #बजट_2026! ##"), ["a", "बजट_2026"])


class CatalogTests(unittest.TestCase):
    def test_feed_order_ads_and_timeline(self):
        catalog = FeedCatalog()
        queued = []
        catalog.on_new_feed_reel = lambda m: queued.append(m.code)

        catalog.ingest_doc(json.loads(embedded_script(make_media("E1"), make_media("E2"))))
        catalog.ingest_doc(timeline_page(make_media("T1")))
        catalog.ingest_doc(ads_page(make_media("AD", product_type="ad")))
        catalog.ingest_doc(feed_page(make_media("P1"), make_media("E2", has_liked=True)))

        self.assertEqual(queued, ["E1", "E2", "P1"])
        self.assertEqual([catalog.get(c).feed_index for c in queued], [0, 1, 2])
        self.assertIsNone(catalog.get("T1").feed_index)
        self.assertTrue(catalog.get("AD").is_ad)
        self.assertIsNone(catalog.get("AD").feed_index)
        self.assertTrue(catalog.get("E2").has_liked)  # viewer state refreshed on re-delivery

    def test_timeline_reel_promoted_when_feed_delivers_it(self):
        catalog = FeedCatalog()
        queued = []
        catalog.on_new_feed_reel = lambda m: queued.append(m.code)
        catalog.ingest_doc(timeline_page(make_media("X")))
        catalog.ingest_doc(feed_page(make_media("X")))
        self.assertEqual(queued, ["X"])
        self.assertEqual(catalog.get("X").feed_index, 0)

    def test_bad_media_is_counted_not_fatal(self):
        catalog = FeedCatalog()
        bad = make_media("B1")
        bad["user"] = "not-a-dict"
        catalog.ingest_doc(feed_page(bad, make_media("G1")))
        self.assertEqual(catalog.parse_errors, 1)
        self.assertIsNotNone(catalog.get("G1"))

    def test_wait_for(self):
        async def scenario():
            catalog = FeedCatalog()
            asyncio.get_running_loop().call_later(0.15, catalog.ingest_doc, feed_page(make_media("LATE")))
            found = await catalog.wait_for("LATE", timeout=1.0)
            missing = await catalog.wait_for("NEVER", timeout=0.2)
            return found, missing

        found, missing = asyncio.run(scenario())
        self.assertEqual(found.code, "LATE")
        self.assertIsNone(missing)


if __name__ == "__main__":
    unittest.main()
