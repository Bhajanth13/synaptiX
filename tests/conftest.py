import os

# Keep tests fast and deterministic: never load a real language model.
os.environ["SYNAPTIX_SLM"] = "/nonexistent.gguf"
os.environ["SYNAPTIX_CONFIG"] = "/nonexistent/synaptix/config.yaml"
os.environ["SYNAPTIX_REWRITES"] = "/nonexistent/rewrites.yaml"
