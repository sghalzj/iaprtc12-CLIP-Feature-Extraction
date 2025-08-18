import os, re
import numpy as np
import scipy.io as sio
import xml.etree.ElementTree as ET
from collections import Counter
from typing import List

# ====== Configuration ======
ANN_DIR      = r"S:\dataset\iaprtc12\unpack\iaprtc12\annotations_complete_eng"
OUT_DIR      = r"S:\SCH-main"
USE_STOPWORDS = True
MIN_DF       = 5      # minimum document frequency
TOP_K        = 255    # top-K vocabulary
EXT_FILTER   = {'.eng'}
# ===========================

# Basic English stopwords
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "at", "by", "from", "with",
    "is", "are", "was", "were", "be", "being", "been", "it", "its", "this", "that", "these", "those",
    "as", "into", "than", "then", "there", "here", "over", "under", "up", "down", "out", "off",
    "near", "behind", "front", "back", "left", "right", "between", "across", "through",
    "no", "not", "without", "within", "about", "above", "below", "again", "once", "very", "more", "most"
}

def normalize_text(text: str) -> List[str]:
    # Lowercase, remove punctuation/digits, split, filter stopwords
    text = text.lower()
    text = re.sub(r"[^\w]+", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [t for t in text.strip().split() if t]
    if USE_STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

def read_file(fp):
    # Try multiple encodings
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            with open(fp, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except: pass
    with open(fp, "rb") as f:
        return f.read().decode("latin1", "ignore")

def extract_tokens_from_annotations(ann_dir):
    token_map = {}
    total = 0
    for root, _, files in os.walk(ann_dir):
        folder = os.path.basename(root)
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in EXT_FILTER:
                continue
            key = f"{folder}/{os.path.splitext(file)[0]}"
            try:
                raw = read_file(os.path.join(root, file))
                tree = ET.fromstring(raw)
                desc = tree.findtext("DESCRIPTION", "")
                tokens = normalize_text(desc)
                if tokens:
                    token_map[key] = tokens
                    total += 1
            except ET.ParseError:
                print(f"Failed to parse: {file}")
    print(f"Parsed {total} valid .eng files.")
    return token_map

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Reading annotations from: {ANN_DIR}")

    ann_map = extract_tokens_from_annotations(ANN_DIR)
    keys = sorted(ann_map.keys(), key=lambda x: [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', x)])
    N = len(keys)
    if N == 0:
        raise RuntimeError("No valid annotations found.")

    # Build vocabulary
    token_counts = Counter(t for tokens in ann_map.values() for t in tokens)
    if MIN_DF > 0:
        token_counts = Counter({w: c for w, c in token_counts.items() if c >= MIN_DF})
    vocab = [w for w, _ in token_counts.most_common(TOP_K)]
    K = len(vocab)
    if K < TOP_K:
        print(f"[Info] Vocabulary size reduced to {K} (TOP_K={TOP_K})")
    word2id = {w: i for i, w in enumerate(vocab)}

    # Build multi-hot label matrix
    Y = np.zeros((N, K), dtype=np.uint8)
    for i, k in enumerate(keys):
        for token in ann_map[k]:
            j = word2id.get(token)
            if j is not None:
                Y[i, j] = 1

    # Save outputs
    vocab_path = os.path.join(OUT_DIR, "iaprtc12_vocab255.txt")
    mat_path   = os.path.join(OUT_DIR, "iaprtc12_labels255.mat")
    npz_path   = os.path.join(OUT_DIR, "iaprtc12_labels255.npz")

    np.savetxt(vocab_path, np.array(vocab, dtype=str), fmt="%s", encoding="utf-8")
    sio.savemat(mat_path, {
        "keys":     np.array(keys, dtype=object),
        "vocab255": np.array(vocab, dtype=object),
        "Y_multi":  Y
    }, do_compression=True)
    np.savez_compressed(npz_path,
        keys=np.array(keys, dtype=object),
        vocab255=np.array(vocab, dtype=object),
        Y_multi=Y
    )

    density = float(Y.sum()) / (N * K)
    print("Label matrix generated")
    print(f"Samples = {N}, Label dim = {K}, Density = {density:.6f}")
    print("Saved files:")
    print(" -", vocab_path)
    print(" -", mat_path)
    print(" -", npz_path)
    print("Image paths: images/{keys[i]}.jpg")

if __name__ == "__main__":
    main()
