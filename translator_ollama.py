import requests
import hashlib
import logging
import time

DEFAULT_API_URL = "http://127.0.0.1:11434/api"						# Default Ollama API url
DEFAULT_MODEL = "zongwei/gemma3-translator:4b"						# Default Ollama model to use for the translation
DEFAULT_KEEPALIVE = 1 									# Clear VRAM after 1 sec. Don't put this to 0 or the time between each line translation will be too long.
DEFAULT_TIMEOUT = 120 									# Ollama API request will timeout after 120 sec. Ollama built-in timeout is 300 sec.
DEFAULT_OPTIONS = {"seed": 42, "temperature": 0.1, "top_p": 0.9, "num_predict": 256}	# Default Ollama options used in API request.
# Ollama options :
# - "seed" is an arbitrary number to keep consistency.
# - "temperature" & "top_p" setup for accuracy.
# - "num_predict" prevents infinite "word-repeat" loops.


class OllamaTranslateError(Exception):
    pass


def _prompt(text: str, src_lang: str, tgt_lang: str) -> str:
    return (
        f"Translate from {src_lang} to {tgt_lang}: {text}\n"
    )


def translate(text: str, src_lang="auto", tgt_lang="en", *, model=DEFAULT_MODEL, api_url=DEFAULT_API_URL, options=DEFAULT_OPTIONS, keep_alive=DEFAULT_KEEPALIVE, timeout=DEFAULT_TIMEOUT) -> str:
    """Translate arbitrary text using a local Ollama instance."""
    if not text.strip():
        return ""

    api_url = api_url
    api_url_endpoint = api_url + "/generate"
    model = model
    opts = options
    ka = keep_alive

    payload = {
        "model": model,
        "prompt": _prompt(text, src_lang, tgt_lang),
        "stream": False,
        "options": opts,
        "keep_alive": ka,
    }

    try:
        resp = requests.post(api_url_endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        j = resp.json()

        # Try multiple possible keys
        if isinstance(j, dict):
            for key in ("response", "content", "result", "text", "output"):
                if key in j and isinstance(j[key], str):
                    return j[key].strip()
        if isinstance(j, list):
            parts = [item.get("content", "") for item in j if isinstance(item, dict)]
            return "".join(parts).strip()

        raise OllamaTranslateError(f"Unexpected Ollama response format: {j}")

    except requests.RequestException as e:
        raise OllamaTranslateError(str(e)) from e


def translate_srt_file(in_path: str, out_path: str, src_lang="auto", tgt_lang="en", *, model=DEFAULT_MODEL, api_url=DEFAULT_API_URL, options=DEFAULT_OPTIONS, keep_alive=DEFAULT_KEEPALIVE, timeout=DEFAULT_TIMEOUT):
    """Read an .srt file, translate text lines, and write a new .srt file with same timing."""
    # Before starting a new translation, wait 5 seconds after clear VRAM operation (triggerd after keep_alive duration), to be sure VRAM is completely cleared.
    waiting_time = keep_alive + 5
    logging.debug(f"Waiting {waiting_time} seconds for VRAM to be completely cleared...")
    time.sleep(waiting_time)
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    buffer = []
    for line in lines:
        if line.strip() == "":
            if buffer:
                index, timecode, *text_lines = buffer
                text = "\n".join(text_lines).strip()
                translated = translate(text, src_lang, tgt_lang, model=model, api_url=api_url, options=options, keep_alive=keep_alive, timeout=timeout)
                
                i = index.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                t = timecode.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                txt = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                tr = translated.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                logging.debug(f"  ORIGINAL LINE ==> {i} {t} {txt}")
                logging.debug(f"TRANSLATED LINE ==> {i} {t} {tr}\n")

                new_lines.extend([index, timecode, translated + "\n", "\n"])
                buffer = []
        else:
            buffer.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

