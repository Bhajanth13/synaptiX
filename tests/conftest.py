import os

# Keep tests fast and deterministic: never load a real language model.
os.environ["SYNAPTIX_SLM"] = "/nonexistent.gguf"
