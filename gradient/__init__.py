"""
PRAG — the Gradient: per-persona models trained on what each feed serves.

    corpus      store.py, taxonomy.py, generate.py, ingest.py: the training corpus
    teachers    label.py: Jev's watch / skip judgement of every item, per persona
    models      prefs.py (preference classifiers), lora.py (the persona language model)
    use         retrieval.py, chat.py: grounded chat, debate, one question two worldviews
    panel       api.py, panel.py: the panel's Corpus, Training and Talk tabs

    python -m gradient status
"""
