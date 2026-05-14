import numpy as np
from PIL import Image
import tensorflow as tf

import os
import pathlib

def load_image(path: str, target_size: tuple = (150, 150)) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def batch_load_images(paths: list, target_size: tuple = (150,150)) -> np.ndarray:
    return np.stack([load_image(p ,target_size) for p in paths], axis =0)

def extract_features(paths: list, encoder, target_size: tuple, output_dir: str, batch_size: int = 32):
    os.makedirs(output_dir, exist_ok=True)

    pending = []
    for p in paths:
        img_id = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(output_dir, f'{img_id}.npy')
        if not os.path.exists(out_path):
            pending.append(p)

    print(f"Total {len(pending)} dari {len(paths)} images perlu diekstraksi")

    for i in range(0, len(pending), batch_size):
        batch_paths = pending[ i : i + batch_size]
        batch_imgs = batch_load_images(batch_paths, target_size)

        batch_imgs = tf.keras.applications.inception_v3.preprocess_input(batch_imgs * 255.0)

        features = encoder.predict(batch_imgs, verbose=0)

        for j, p in enumerate(batch_paths):
            img_id = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(output_dir, f'{img_id}.npy')
            np.save(out_path, features[j])

        print(f"Extracted features for {min(i + batch_size, len(pending))} / {len(pending)} images")

# arr = load_image('../data/image.png')
# print(arr.shape)

# test_paths = [
#     '../data/image.png',
#     '../data/aino.jpg',]

# arrs = batch_load_images(test_paths)
# print(arrs.shape)

# feat = extract_features(test_paths, tf.keras.applications.InceptionV3(include_top=False, pooling='avg'), (299, 299), '../data/features')
# feat = np.load('../data/features/image.npy')
# assert feat.shape == (2048,), f"Expected feature shape (2048,), got {feat.shape}"
# print("Feature extraction successful, shape:", feat.shape)