import os, re
import numpy as np
import torch
import clip
import scipy.io as sio
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ======= Configuration =======
LABELS_NPZ   = r"S:\SCH-main\iaprtc12_labels255.npz"  # .npz file with 'keys'
ANN_FULL_DIR = r"S:\dataset\iaprtc12\unpack\iaprtc12\annotations_complete_eng"  # annotation folder
OUT_DIR      = r"S:\SCH-main"
MODEL_NAME   = "ViT-B/16"
NORMALIZE    = False      # whether to L2 normalize the embeddings
FP16         = False      # use half-precision (only on CUDA)
ENCODE_MODE  = "concat"   # options: "avg" or "concat"
MAX_SENT     = 5          # number of sentences to keep in 'concat' mode
# =============================

def read_text(fp):
    # Try reading file using common encodings
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            with open(fp, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except: pass
    with open(fp, "rb") as f:
        return f.read().decode("latin1", "ignore")

def extract_title_desc(fp):
    # Extract <TITLE> and <DESCRIPTION> from XML file
    try:
        tree = ET.fromstring(read_text(fp))
        title = tree.findtext("TITLE", "").strip()
        desc  = tree.findtext("DESCRIPTION", "").strip()
        return f"{title}. {desc}".strip()
    except: return ""

def split_sentences(text):
    # Simple sentence splitter
    return [s.strip() for s in re.split(r"[.?!；;。]+", text) if s.strip()]

def encode_texts(texts, model, device, mode="avg", max_sent=5):
    # Encode all texts using sentence average or concatenation
    if mode == "avg":
        feats = np.zeros((len(texts), 512), dtype=np.float32)
    elif mode == "concat":
        feats = np.zeros((len(texts), 512 * max_sent), dtype=np.float32)
    else:
        raise ValueError("ENCODE_MODE must be 'avg' or 'concat'")

    for i, t in enumerate(tqdm(texts, desc=f"Encoding texts [{mode}]")):
        if not t:
            continue
        sents = split_sentences(t) or ["a photo."]
        sents = sents[:max_sent]  # truncate if too many

        vecs = []
        for s in sents:
            tokens = clip.tokenize([s], truncate=True).to(device)
            if FP16 and device == "cuda":
                tokens = tokens.half()
            with torch.no_grad():
                f = model.encode_text(tokens).float().cpu().numpy()
            vecs.append(f[0])

        if mode == "avg":
            feat = np.mean(vecs, axis=0)
            if NORMALIZE:
                feat /= (np.linalg.norm(feat) + 1e-12)
            feats[i] = feat

        elif mode == "concat":
            padded = vecs + [np.zeros(512)] * (max_sent - len(vecs))  # pad with zeros
            concat_feat = np.concatenate(padded)
            if NORMALIZE:
                concat_feat /= (np.linalg.norm(concat_feat) + 1e-12)
            feats[i] = concat_feat

    return feats

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load keys from .npz
    data = np.load(LABELS_NPZ, allow_pickle=True)
    keys = data["keys"].astype(str)
    print(f"Loaded {len(keys)} keys")

    # Read annotations
    texts, missing = [], []
    for k in tqdm(keys, desc="Parsing annotations"):
        fp = os.path.join(ANN_FULL_DIR, k + ".eng")
        if not os.path.exists(fp):
            texts.append("")
            missing.append(k)
            continue
        text = extract_title_desc(fp)
        texts.append(text)
        if not text:
            missing.append(k)

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP {MODEL_NAME} on {device} ...")
    model, _ = clip.load(MODEL_NAME, device=device)
    if FP16 and device == "cuda":
        model = model.half()
    model.eval()

    # Encode text
    feats = encode_texts(texts, model, device, mode=ENCODE_MODE, max_sent=MAX_SENT)

    # Save outputs
    suffix = f"{ENCODE_MODE}{MAX_SENT}" if ENCODE_MODE == "concat" else "avg"
    npz_path = os.path.join(OUT_DIR, f"clip_vitb16_text_embeds_sentence_{suffix}.npz")
    mat_path = os.path.join(OUT_DIR, f"clip_vitb16_text_embeds_sentence_{suffix}.mat")

    np.savez_compressed(npz_path,
                        keys=keys,
                        feats=feats,
                        texts=np.array(texts, dtype=object),
                        missing=np.array(missing, dtype=object))
    
    sio.savemat(mat_path,
                {"keys": keys, "feats": feats, "texts": texts, "missing": missing},
                do_compression=True)

    print(f"Saved:\n  {npz_path}\n  {mat_path}")
    print(f"Missing annotations: {len(missing)}")

if __name__ == "__main__":
    main()
