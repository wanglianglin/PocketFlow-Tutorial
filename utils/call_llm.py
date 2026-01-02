from google import genai
import os
import logging
import json
import requests
from datetime import datetime

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"


def load_cache():
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except:
        logger.warning(f"Failed to load cache.")
    return {}


def save_cache(cache):
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    except:
        logger.warning(f"Failed to save cache")


def get_llm_provider():
    provider = os.getenv("LLM_PROVIDER")
    if not provider and (os.getenv("GEMINI_PROJECT_ID") or os.getenv("GEMINI_API_KEY")):
        provider = "GEMINI"
    # if necessary, add ANTHROPIC/OPENAI
    return provider

def _call_llm_provider(prompt: str) -> str:
    """
    Call an LLM provider based on environment variables.
    Environment variables:
    - LLM_PROVIDER: "OLLAMA", "XAI", "ARK" ...
    - <provider>_MODEL: Model name (e.g., OLLAMA_MODEL, XAI_MODEL, ARK_MODEL)
    - <provider>_BASE_URL: Base URL (e.g., OLLAMA_BASE_URL, XAI_BASE_URL, ARK_BASE_URL)
    - <provider>_API_KEY: API key (e.g., OLLAMA_API_KEY, XAI_API_KEY, ARK_API_KEY; optional for providers that don't require it)
    For大多数 OpenAI 兼容服务，会在 base_url 后面自动拼接 /v1/chat/completions；
    ARK（火山方舟）单独特殊处理为 /api/v3/chat/completions。
    """
    logger.info(f"PROMPT: {prompt}")  # log the prompt

    # Read the provider from environment variable
    provider = os.environ.get("LLM_PROVIDER")
    if not provider:
        raise ValueError("LLM_PROVIDER environment variable is required")

    provider_upper = provider.upper()

    # Construct the names of the other environment variables
    model_var = f"{provider}_MODEL"
    base_url_var = f"{provider}_BASE_URL"
    api_key_var = f"{provider}_API_KEY"

    # Read the provider-specific variables
    model = os.environ.get(model_var)
    base_url = os.environ.get(base_url_var)
    api_key = os.environ.get(api_key_var, "")  # API key is optional, default to empty string

    # Validate required variables
    if not model:
        raise ValueError(f"{model_var} environment variable is required")
    if not base_url:
        raise ValueError(f"{base_url_var} environment variable is required")

    base_url = base_url.rstrip("/")

    # 🔹 构造最终 URL：通用 /v1/chat/completions，ARK 走 /api/v3/chat/completions
    if provider_upper == "ARK":
        # 允许三种写法：
        # 1) https://ark.cn-beijing.volces.com
        # 2) https://ark.cn-beijing.volces.com/api/v3
        # 3) https://ark.cn-beijing.volces.com/api/v3/chat/completions
        if "/chat/completions" in base_url:
            url = base_url
        elif base_url.endswith("/api/v3"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/api/v3/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"

    # Configure headers and payload based on provider
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:  # Only add Authorization header if API key is provided
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        # 先拿原始文本，方便出错时打印
        raw_text = response.text

        # 尝试解析 JSON（无论成功失败，都先记录一下）
        try:
            response_json = response.json()
        except ValueError as e:
            logger.error(
                "Failed to parse response as JSON from %s. status=%s, body=%s",
                provider,
                response.status_code,
                raw_text[:500],
            )
            raise Exception(
                f"Failed to parse response as JSON from {provider}. "
                f"Status: {response.status_code}, body: {raw_text[:200]}"
            ) from e

        logger.info("RESPONSE:\n%s", json.dumps(response_json, indent=2))

        # 如果 HTTP 层报错，抛出带详情的异常
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_message = f"HTTP error occurred when calling {provider}: {e}"
            try:
                error_details = response_json.get("error", "No additional details")
                error_message += f" (Details: {error_details})"
            except Exception:
                pass
            raise Exception(error_message) from e

        # 正常返回内容
        return response_json["choices"][0]["message"]["content"]

    except requests.exceptions.ConnectionError:
        raise Exception(f"Failed to connect to {provider} API. Check your network connection.")
    except requests.exceptions.Timeout:
        raise Exception(f"Request to {provider} API timed out.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"An error occurred while making the request to {provider}: {e}")

# By default, we Google Gemini 2.5 pro, as it shows great performance for code understanding
def call_llm(prompt: str, use_cache: bool = True) -> str:
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = load_cache()
        # Return from cache if exists
        if prompt in cache:
            logger.info(f"RESPONSE: {cache[prompt]}")
            return cache[prompt]

    provider = get_llm_provider()
    if provider == "GEMINI":
        response_text = _call_llm_gemini(prompt)
    else:  # generic method using a URL that is OpenAI compatible API (Ollama, ...)
        response_text = _call_llm_provider(prompt)

    # Log the response
    logger.info(f"RESPONSE: {response_text}")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = load_cache()
        # Add to cache and save
        cache[prompt] = response_text
        save_cache(cache)

    return response_text


def _call_llm_gemini(prompt: str) -> str:
    if os.getenv("GEMINI_PROJECT_ID"):
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GEMINI_PROJECT_ID"),
            location=os.getenv("GEMINI_LOCATION", "us-central1")
        )
    elif os.getenv("GEMINI_API_KEY"):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        raise ValueError("Either GEMINI_PROJECT_ID or GEMINI_API_KEY must be set in the environment")
    # model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-05-06")
    response = client.models.generate_content(
        model=model,
        contents=[prompt]
    )
    return response.text

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
