import os
import numpy as np
from PIL import Image
import torch
import clip
from tqdm import tqdm
import scipy.io as sio


IMG_DIR = r"S:\dataset\iaprtc12\unpack\iaprtc12\images"
OUT_DIR = r"S:\SCH-main"
MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 64
NORMALIZE = False
FP16 = False

def collect_images(img_dir):
    keys, paths = [], []
    for folder in sorted(os.listdir(img_dir)):
        sub_dir = os.path.join(img_dir, folder)
        if not os.path.isdir(sub_dir): continue
        for file in sorted(os.listdir(sub_dir)):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                keys.append(f"{folder}/{os.path.splitext(file)[0]}")
                paths.append(os.path.join(sub_dir, file))
    return keys, paths

def load_tensor(path, preprocess):
    try:
        return preprocess(Image.open(path).convert("RGB"))
    except Exception:
        return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    keys, paths = collect_images(IMG_DIR)
    print(f"Collected {len(keys)} images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    if FP16 and device == "cuda": model = model.half()
    model.eval()

    feats = np.zeros((len(keys), 512), dtype=np.float32)
    bad, batch_imgs, batch_idx = [], [], []

    for i, path in enumerate(tqdm(paths, desc="Encoding images")):
        img_tensor = load_tensor(path, preprocess)
        if img_tensor is None:
            bad.append(i)
            continue
        batch_imgs.append(img_tensor)
        batch_idx.append(i)

        if len(batch_imgs) == BATCH_SIZE or i == len(keys) - 1:
            x = torch.stack(batch_imgs).to(device)
            if FP16 and device == "cuda": x = x.half()
            with torch.no_grad(): feat = model.encode_image(x).float().cpu().numpy()
            if NORMALIZE:
                feat /= (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
            for j, idx in enumerate(batch_idx):
                feats[idx] = feat[j]
            batch_imgs, batch_idx = [], []

    np.savez_compressed(os.path.join(OUT_DIR, "clip_vitb16_image_embeds_ordered.npz"),
                        keys=np.array(keys), feats=feats, missing=np.array(bad, dtype=np.int32))
    sio.savemat(os.path.join(OUT_DIR, "clip_vitb16_image_embeds_ordered.mat"),
                {"keys": keys, "feats": feats, "missing": np.array(bad, dtype=np.int32)},
                do_compression=True)

    print(f"Saved features. Missing: {len(bad)} images.")

if __name__ == "__main__":
    main()
