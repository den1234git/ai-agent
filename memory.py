import json
import os
import re

PATH = "memory.json"


def load_memory():
    if not os.path.exists(PATH):
        return {}

    try:
        with open(PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError:
        return {}


def save_memory(mem):
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)


def clean_name_value(key, value):
    """name キーの値から「あなたの名前は」「です」などの余分な文言を取り除く（複数回出現にも対応）"""
    if key == "name" and isinstance(value, str):
        value = re.sub(r"あなたの名前は", "", value)
        value = re.sub(r"です", "", value)
        value = value.strip()
    return value