---
description: Builds the multi-modal evaluation pipeline. Audio/Images are extracted locally, and text is scored against a target bias using the LM Studio API (OpenAI compatible).
---

## Steps

1. **Transcriber Module**
   - Create `perception/transcriber.py`. Write a function `transcribe_audio(file_path)` using `mlx-whisper` returning the transcript string.

2. **OCR Module**
   - Create `perception/ocr.py`. Write a function `extract_text_from_frame(image_path)` using `easyocr` to return joined English text.

3. **Alignment Evaluator (LM Studio Integration)**
   - Create `perception/evaluator.py`. Import `requests`, `json`, and the variables from `config.py`.
   - Write a function `evaluate_alignment(transcript, ocr_text, target_bias)`.
   - Format the payload for LM Studio's OpenAI-compatible `/v1/chat/completions` endpoint:
     - `model`: `LLM_MODEL_NAME`
     - `messages`: Array with a `system` prompt instructing it to be a JSON-only evaluator, and a `user` prompt passing the transcript, OCR, and target bias.
     - `temperature`: 0.1
     - `response_format`: `{"type": "json_object"}`
   - Use `requests.post(LM_STUDIO_URL)` to hit `http://127.0.0.1:1234/v1/chat/completions`.
   - Extract `choices[0]['message']['content']`, parse it using `json.loads()`, and return a dict with `alignment_score` (float) and `reasoning` (str). Handle JSONDecodeError with a safe fallback of `0.0`.

4. **Integration Test**
   - Write a script `tests/test_perception.py` with mock text data directly calling `evaluate_alignment()` to verify LM Studio connectivity.