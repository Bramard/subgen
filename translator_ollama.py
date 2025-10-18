# TRANSLATOR OLLAMA PATCH


import os
import fnmatch
import requests
import hashlib
import logging
import time
import re


# Source language
patch_translate_source_language = os.getenv('OLLAMA_TRANSLATE_SRC', 'auto')

# Target language list
patch_translate_target_languages = os.getenv('OLLAMA_TRANSLATE_TGT', 'mk|fr')

# Ollama model list to use for the translation(s)
patch_translate_models = os.getenv('OLLAMA_TRANSLATE_MODEL', 'zongwei/gemma3-translator:1b|zongwei/gemma3-translator:4b')

# Ollama API url
patch_translate_api_url = os.getenv('OLLAMA_TRANSLATE_API_URL', 'http://127.0.0.1:11434/api')

# Ollama API request will timeout after 120 sec.
# Ollama built-in timeout is 300 sec.
OLLAMA_TIMEOUT = 120

# Clear VRAM after 1 sec.
# Do not set this to 0 or the time between each line translation will be too long.
VRAM_KEEPALIVE = 1

# Wait 5 seconds between two translations.
# Safety: arbitrary time, starting after keep_alive duration, to make sure VRAM is completely cleared.
TRANSLATION_GAP_TIME = 5

# Ollama options used in API request :
# - seed: arbitrary number to keep consistency.
# - temperature & top_p: set up for accuracy.
# - num_predict: prevents infinite "word-repeat" loops.
OLLAMA_OPTIONS = {"seed": 42, "temperature": 0.1, "top_p": 0.9, "num_predict": 150}

# Ollama stream
# Do not set this to True. This patch does not handle it.
OLLAMA_STREAM = False


class OllamaTranslateError(Exception):
    pass


#----------------------------------------------
# FUNCTIONS called in translator_ollama.py
#----------------------------------------------
def _prompt(text: str, src_lang: str, tgt_lang: str) -> str:
    return (
        f"Translate from {src_lang} to {tgt_lang}: {text}\n"
    )


def translate(text: str, src_lang: str, tgt_lang: str, model: str) -> str:
    """
    Translate arbitrary text using a local Ollama instance.
    """
    if not text.strip():
        return ""

    api_url_endpoint = patch_translate_api_url + "/generate"

    payload = {
        "model": model,
        "prompt": _prompt(text, src_lang, tgt_lang),
        "stream": OLLAMA_STREAM,
        "options": OLLAMA_OPTIONS,
        "keep_alive": VRAM_KEEPALIVE,
    }

    try:
        resp = requests.post(api_url_endpoint, json=payload, timeout=OLLAMA_TIMEOUT)
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


def translate_srt_file(in_path: str, out_path: str, src_lang: str, tgt_lang: str, model: str):
    """
    Read an .srt file, translate text lines, and write a new .srt file with same timing.
    """
    waiting_time = VRAM_KEEPALIVE + TRANSLATION_GAP_TIME
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
                unfiltered_text = "\n".join(text_lines).strip()
                text = clean_for_ollama(unfiltered_text)
                translated = translate(text, src_lang, tgt_lang, model)
                
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


def clean_for_ollama(text: str) -> str:
    """
    Cleans subtitle text before feeding it to an LLM.
    Replaces music notes, singing cues, and sound effect markers with [MUSIC].
    Then, collapse multiple [MUSIC]s and strip whitespace.
    """
    pattern = (
        r'(?:'
        r'[♪♫♩♬♭♯𝄞]+.*?[♪♫♩♬♭♯𝄞]+'						# ♪ ... ♪ or ♫ ... ♫
        r'|[♪♫♩♬♭♯𝄞]{2,}'							# bare note sequences
        r'|\[.*?(music|sing|song|instrument|melody|theme).*?\]'			# [Music], [Singing]
        r'|\(.*?(music|sing|song|instrument|melody|theme).*?\)'			# (Music), (Singing)
        r')'
    )
    cleaned = re.sub(pattern, '[MUSIC]', text, flags=re.IGNORECASE)		# Replace
    cleaned = re.sub(r'(\[MUSIC\]\s*){2,}', '[MUSIC] ', cleaned).strip()	# Collapse
    return cleaned


#----------------------------------------------
# FUNCTIONS called in subgen.py
#----------------------------------------------
def gen_translated_subtitles(file_path: str) -> None:
    """Generates translated subtitles for a video file.

    External Ollama instance is useed.
    File "translator_ollama.py" is imported.

    Args:
        file_path: str - The path to the video file.
    """

    # Needed from subgen.py
    from subgen import convert_to_bool
    show_in_subname_subgen = convert_to_bool(os.getenv('SHOW_IN_SUBNAME_SUBGEN', True))

    try:
        # Translate with every model supplied
        for patch_translate_model in patch_translate_models.split("|"):
            # Construct suffix for translated file
            suffix_pre = "ollama-"											# suffix begins with this string
            suffix_post = "-translated"											# suffix ends with this string
            patch_translate_model_trimmed = patch_translate_model.replace(":","_").replace("/","_") 			# to avoid using char not compatible with OS filenames
            suffix = suffix_pre + patch_translate_model_trimmed + suffix_post 						# suffix added to the translated filename

            # Construct exclusion pattern to match previously translated files
            other_translation_pattern = f"*{suffix_pre}*{suffix_post}*" 						# previously translated files have a different suffix if they were translated with a different model

            # Get all the subtitle files from the video directory while excluding previous translations.
            directory = os.path.dirname(file_path)
            all_files_unfiltered = os.listdir(directory)
            all_files = [f for f in all_files_unfiltered if not fnmatch.fnmatch(f, other_translation_pattern)] 		# exclude previously translated files to avoid using it as source for current translation

            # Find subtitle file associated to the video, generateed by subgen, to use as source for the translation.
            file_path_without_extension = os.path.splitext(file_path)[0]
            file_name_without_extension = os.path.basename(file_path_without_extension)
            extension = ".srt"
            if show_in_subname_subgen:
                srt_filename = next(f for f in all_files if f.startswith(file_name_without_extension + ".subgen") and f.endswith(extension))
            else:
                srt_filename = next(f for f in all_files if f.startswith(file_name_without_extension) and f.endswith(extension))
            srt = f"{directory}/{srt_filename}"

            # helpful debug lines :
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - suffix = {suffix}")
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - other_translation_pattern = {other_translation_pattern}")
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - directory = {directory}")
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - all_files_unfiltered = {all_files_unfiltered}")
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - all_files = {all_files}")
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - file_path = {file_path}")
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - file_path_without_extension = {file_path_without_extension}")
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - file_name_without_extension = {file_name_without_extension}")
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - srt_filename =  : {srt_filename}")
            #logging.debug(f"TRANSLATOR OLLAMA PATCH - srt =  : {srt}")

            # Translate in every language supplied
            for patch_translate_target_language in patch_translate_target_languages.split("|"):
                try:
                    start_time = time.time()
                    logging.info(f"Translating file {srt_filename} in {patch_translate_target_language} with model {patch_translate_model}")
                    out_srt = os.path.splitext(srt)[0] + f".{suffix}.{patch_translate_target_language}{extension}"	# add suffix and language to filename
                    
                    # helpful debug lines :
                    #logging.debug(f"TRANSLATOR OLLAMA PATCH - out_srt = : {out_srt}")

                    # Translate the subtitle file
                    translate_srt_file(
                        srt,
                        out_srt,
                        patch_translate_source_language,
                        patch_translate_target_language,
                        patch_translate_model,
                    )
                    elapsed_time = time.time() - start_time
                    minutes, seconds = divmod(int(elapsed_time), 60)
                    logging.info(f"Completed translation to {patch_translate_target_language} in {minutes}m {seconds}s: {srt} --> {out_srt}")
                except Exception as e:
                    logging.info(f"Error translating subtitles in {patch_translate_target_language} for {file_path}: {e}")

    except Exception as e:
        logging.info(f"Error translating subtitles for {file_path}: {e}")

