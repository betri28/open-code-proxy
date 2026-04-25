import os

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key not in os.environ:
                        os.environ[key] = value

load_env_file()

# Secrets (from .env or environment variables)
PROXY = os.getenv("OPENCODE_PROXY")
API_KEY = os.getenv("OPENCODE_API_KEY")

API_BASE_OPENAI    = "https://opencode.ai/zen/go/v1/chat/completions"
API_BASE_ANTHROPIC = "https://opencode.ai/zen/go/v1/messages"
HOST = "0.0.0.0"
PORT = 4000
WEB_PORT = 8082

MODELS = {
    # model_id          : config dict
    "glm-5.1"          : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
    "glm-5"            : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
    "kimi-k2.5"        : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
    "kimi-k2.6"        : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
    "deepseek-v4-pro"  : {"endpoint": API_BASE_ANTHROPIC, "protocol": "anthropic"},
    "deepseek-v4-flash": {"endpoint": API_BASE_ANTHROPIC, "protocol": "anthropic"},
    "mimo-v2-pro"      : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
    "mimo-v2-omni"     : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
    "mimo-v2.5-pro"    : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
    "mimo-v2.5"        : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
    "minimax-m2.7"     : {"endpoint": API_BASE_ANTHROPIC, "protocol": "anthropic"},
    "minimax-m2.5"     : {"endpoint": API_BASE_ANTHROPIC, "protocol": "anthropic"},
    "qwen3.6-plus"     : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
    "qwen3.5-plus"     : {"endpoint": API_BASE_OPENAI,    "protocol": "openai"},
}

def load_routes():
    """Load ROUTES from environment variables or use default."""
    opus_model = os.getenv("OPUS_MAP_MODEL", "kimi-k2.6")
    sonnet_model = os.getenv("SONNET_MAP_MODEL", "glm-5.1")
    haiku_model = os.getenv("HAIKU_MAP_MODEL", "minimax-m2.5")

    return {
        "opus":   {"match": ["opus"],   "model": opus_model},
        "sonnet": {"match": ["sonnet"], "model": sonnet_model},
        "haiku":  {"match": ["haiku"],  "model": haiku_model},
    }

ROUTES = load_routes()


def get_model_config(model_id: str) -> dict:
    """Return merged config for model_id with sensible defaults."""
    cfg = MODELS.get(model_id, {})
    defaults = {"endpoint": API_BASE_OPENAI, "protocol": "openai"}
    return {**defaults, **cfg}
