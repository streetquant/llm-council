"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "allenai/olmo-3.1-32b-think:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "qwen/qwen3-coder:free",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "xiaomi/mimo-v2-flash:free"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
