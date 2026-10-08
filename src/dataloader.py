import json
import pathlib
from typing import List, Tuple, Dict
from tqdm import tqdm

def load_data(path: str, key: str = None, limit: int = None) -> List[str]:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith("["):
            raw = json.loads(content)
        else:
            f.seek(0)
            raw = [json.loads(line) for line in f]

    total = min(limit, len(raw)) if limit else len(raw)
    for item in tqdm(raw, total=total, desc="Loading data"):
        if limit and len(data) >= limit: break
        text = item.get(key, "") if key else item
        if isinstance(text, str) and len(text.strip()) > 0:
            data.append(text.strip())

    return data

def load_parallel_data(path: str, human_key: str, ai_key: str, limit: int = None) -> List[Tuple[str, str]]:
    path = pathlib.Path(path)
    raw = json.loads(path.read_text(encoding='utf-8'))

    pairs = []
    total = min(limit, len(raw)) if limit else len(raw)
    for item in tqdm(raw, total=total, desc="Loading parallel data"):
        if limit and len(pairs) >= limit: break
        h = item.get(human_key)
        a = item.get(ai_key)
        if h and a:
            pairs.append((h, a))
    return pairs

def load_data_pairs(path: str, human_key: str = None, ai_key: str = None, limit: int = None) -> List[Tuple[str, str]]:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    print(f"[DataLoader] Loading data from {path}...")


    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith("["):
            raw = json.loads(content)
        else:
            f.seek(0)
            raw = [json.loads(line) for line in f]

    if not raw:
        return []

    first_item = raw[0]
    pairs = []

    is_parallel_format = (human_key is not None and ai_key is not None) and \
                         (human_key in first_item and ai_key in first_item)

    if is_parallel_format:
        print(f"[DataLoader] Detected Parallel Format (keys: {human_key}, {ai_key})")
        total = min(limit, len(raw)) if limit else len(raw)
        for item in tqdm(raw, total=total, desc="Parsing Parallel Data"):
            if limit and len(pairs) >= limit: break
            h = item.get(human_key)
            a = item.get(ai_key)
            if h and a:
                pairs.append((h, a))

    elif "text" in first_item and "label" in first_item:
        print(f"[DataLoader] Detected Mixed Label Format (keys: text, label)")

        human_texts = []
        ai_texts = []

        for item in raw:
            text = item.get("text", "").strip()
            label = str(item.get("label", "")).lower()

            if not text: continue

            if "human" in label:
                human_texts.append(text)
            else:
                ai_texts.append(text)

        print(f"    - Found Human: {len(human_texts)}")
        print(f"    - Found AI:    {len(ai_texts)}")

        min_len = min(len(human_texts), len(ai_texts))
        if limit:
            min_len = min(min_len, limit)

        for i in range(min_len):
            pairs.append((human_texts[i], ai_texts[i]))

    else:
        raise ValueError(f"Unknown data format in {path}. \n"
                         f"Expected keys '{human_key}'/'{ai_key}' OR 'text'/'label'.\n"
                         f"Found keys: {list(first_item.keys())}")

    print(f"[DataLoader] Successfully loaded {len(pairs)} pairs.")
    return pairs
