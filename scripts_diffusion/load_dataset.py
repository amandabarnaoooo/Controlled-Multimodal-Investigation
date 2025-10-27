import os
from datasets import load_dataset

CACHE_DIR = "/data/ecau171/mathverse_proj/MathVerse_cache"

ds_dict = load_dataset(
    "AI4Math/MathVerse",
    "testmini",
    use_auth_token="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",  # Replace with your Hugging Face read token
    cache_dir=CACHE_DIR
    )
