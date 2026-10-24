# Presenting PRAG in ten minutes

Start the panel first: `python launch.py`. The persona models load on the first question, which takes about
15 seconds, so ask one before you begin.

## 1. The problem (1 min)

Instagram's algorithm learns from watch time and feeds you more of what you linger on. PRAG asks it the other way
round: *what does the algorithm feed a particular kind of person, and what would a mind fed only that believe?*
Reels also carry news that never makes it to the papers, so what a feed serves matters.

## 2. Collection: the agent (2 min). Tabs: Run, Persona, Feed

- **Persona tab.** Each account is a persona: who they are and what they watch to the end. Show the structured
  profile under it: topic weights, stance and language preferences.
- **Run tab.** The agent scrolls Reels as that persona. It reads the feed's JSON a few reels ahead, transcribes
  speech with Whisper, reads the cover with OCR, and has a judge score each reel: the local LLM, or Jev. Jev now
  also sees the cover image, since TypeSafe enabled images on 14 October. The engagement policy turns the score
  into watch time, likes and follows. The browser is locked to the agent.
- **Feed tab.** The curation point (the first 100 judged reels where at least half fit the persona) and the topics
  the algorithm served.

## 3. The corpus (1 min). Tab: Corpus

About 24,000 unique items. Say where they come from, which the table shows: reels from the Instagram sessions;
about 15,300 reels written for the two feeds by an LLM following each persona's profile, 2,400 of them about this
week's real news, each linked to its article; news headlines; Wikipedia. Then the two persona cards: each feed leans the way its persona does (cricket, security and
infrastructure on one side; labour, civil liberties and indie music on the other), yet both get the other side's
politics and things they'd skip, the way a real feed does.

## 4. Training (3 min). Tab: Training, plus [TRAINING.md](TRAINING.md)

- **Preference models.** One classifier per persona. Four candidates on the same story-grouped splits; the chosen
  model beats the hand-written rules by about 0.3 PR-AUC. Point at the words each model leans on. Point at the
  *instagram* row: real reels it never trained on score 0.70 / 0.90 (only 59 each, so treat it as a first check).
- **Versions.** v1 → v2 after a new batch of session reels, scored on the same frozen evaluation set. It's
  retraining, and every version is kept.
- **Persona language models.** A LoRA adapter on a frozen Qwen3 4B, one per persona. Show the loss curve and the
  perplexity table: base model, then the adapter on its own persona, then the adapter on the other persona.
- **Why not train an LLM from scratch?** A feed is far too little text to learn a language. The base model supplies
  the language; the feed supplies everything the persona knows, through training and through retrieval restricted
  to its feed.

## 5. The personas, live (3 min). Tab: Ask

1. **Feed predictor.** Paste the expressway reel from the "Try" list: @mlproject01 would watch it, and it lands in
   @mlproject01's feed. Then paste the gig-workers reel and watch the result flip.
2. **One question, two worldviews.** "Is the government handling unemployment well?" Two answers, each citing its
   own feed, and the divergence of their evidence.
3. **Debate.** "The Agniveer scheme", two rounds, then the moderator's summary.
4. **Chat.** Ask @mlproject01 something its feed never covered: it says so instead of making it up.

## 6. Limits and next steps (1 min)

- The labels are Jev's judgement of the persona, not a real person's. The Label tab is a blind human check.
- Written reels are realistic text, not Instagram posts; the corpus records which is which, and the results report
  each source separately.
- Next: more scheduled sessions on both accounts, so the real share of the corpus grows with every `gradient update`,
  and "Real or not", following Reels-only stories to see which reach the news.
