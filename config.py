

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"

TARGET_BIAS = """
Content that aligns with Indian right-wing political perspectives, specifically favoring the Bharatiya Janata Party (BJP), Rashtriya Swayamsevak Sangh (RSS), and broader nationalist ideologies.

Key indicators of positive alignment (high score):
1. Pro-BJP Governance & Leadership: Praise, positive coverage, or promotion of BJP leaders (e.g., Narendra Modi, Amit Shah, Yogi Adityanath), key government infrastructure initiatives, foreign policy, or economic achievements.
2. RSS & Cultural Nationalism: Support for RSS initiatives, Sanatan Dharma, Hindutva, Indian civilizational history, cultural heritage, and national security/defense.
3. Policy & Development Highlights: Favorable coverage of major projects and policies such as Ram Mandir, Digital India, Make in India, defense indigenization, or border security.
4. Opposition Criticism: Narratives, memes, or commentary that critique or counter opposition political parties (e.g., Congress, AAP, regional coalitions) or media perceived as anti-national/left-leaning.

Scoring Guidelines:
- High Alignment (0.75 - 1.0): Content explicitly supports BJP/RSS, praises right-wing leadership, or heavily promotes nationalist/cultural narratives.
- Moderate Alignment (0.4 - 0.74): Neutral or general patriotism, general cultural/religious posts without explicit political ties, or mild pro-government news.
- Low Alignment (0.0 - 0.39): Content supporting opposition parties, anti-government criticism, left-wing ideology, or completely unrelated/generic topics (sports, pop culture, cooking).
"""