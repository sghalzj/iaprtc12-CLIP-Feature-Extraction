import os
import numpy as np
from PIL import Image
import torch
import clip
from tqdm import tqdm
import scipy.io as sio

# 配置路径
IMG_DIR = "S:\dataset\IAPR-TC-12\images"  # 图像文件夹路径
OUT_DIR = r"S:\SCH-main"  # 输出目录
MODEL_NAME = "ViT-B/16"  # 使用的CLIP模型
BATCH_SIZE = 64  # 批量大小
NORMALIZE = False  # 是否归一化特征

# 直接获取图像路径
def collect_all_images(img_dir):
    img_paths = []
    img_keys = []
    for file in sorted(os.listdir(img_dir)):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            full_path = os.path.join(img_dir, file)
            key = os.path.splitext(file)[0]  # 使用文件名作为 key
            img_paths.append(full_path)
            img_keys.append(key)
    return img_keys, img_paths

# 主流程
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 收集所有图像路径和键
    keys, paths = collect_all_images(IMG_DIR)
    N = len(paths)
    print(f"Collected {N} images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP {MODEL_NAME} on {device} ...")
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    feats = np.zeros((N, 512), dtype=np.float32)
    bad = []

    def load_tensor(p):
        try:
            img = Image.open(p).convert("RGB")
            t = preprocess(img)
            return t
        except Exception:
            return None

    batch_imgs, batch_idx = [], []
    for i, p in enumerate(tqdm(paths, desc="Encoding images")):
        t = load_tensor(p)
        if t is None:
            bad.append(i)
            continue
        batch_imgs.append(t)
        batch_idx.append(i)

        if len(batch_imgs) == BATCH_SIZE or i == N - 1:
            x = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                feat = model.encode_image(x)
            feat = feat.float().cpu().numpy()
            if NORMALIZE:
                norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
                feat = feat / norm
            for j, idx in enumerate(batch_idx):
                feats[idx] = feat[j]
            batch_imgs, batch_idx = [], []

    # 保存特征
    out_npz = os.path.join(OUT_DIR, "clip_vitb16_image_embeds.npz")
    out_mat = os.path.join(OUT_DIR, "clip_vitb16_image_embeds.mat")
    np.savez_compressed(out_npz, keys=np.array(keys), feats=feats, missing=np.array(bad, dtype=np.int32))
    sio.savemat(out_mat, {"keys": keys, "feats": feats, "missing": np.array(bad, dtype=np.int32)},
                do_compression=True)

    print(f"Saved:\n  {out_npz}\n  {out_mat}")
    print(f"Missing images: {len(bad)}")

if __name__ == "__main__":
    main()
