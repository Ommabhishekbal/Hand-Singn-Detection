import os
import cv2
import pandas as pd
import numpy as np
from mediapipe import solutions as mp
from tqdm import tqdm

def normalize_landmarks(landmarks):
    """Convert to relative coordinates and normalize."""
    landmarks = np.array(landmarks)
    base_x, base_y, base_z = landmarks[0]
    landmarks -= [base_x, base_y, base_z]  # Make wrist the origin
    max_val = np.max(np.abs(landmarks))
    if max_val != 0:
        landmarks /= max_val
    return landmarks.flatten()

def process_image(img_path, hands):
    try:
        image = cv2.imread(img_path)
        if image is None:
            return None

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand.landmark]
        normalized = normalize_landmarks(landmarks)
        return normalized

    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return None

def process_all_data():
    mp_hands = mp.hands.Hands(static_image_mode=True, max_num_hands=1)
    data_list = []

    print("📂 Processing existing dataset...")
    for base in ['existing_dataset', 'custom_captures']:
        base_path = os.path.join('data', base)
        if not os.path.exists(base_path):
            continue

        for label in os.listdir(base_path):
            label_path = os.path.join(base_path, label)
            if not os.path.isdir(label_path):
                continue

            for img_file in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
                img_path = os.path.join(label_path, img_file)
                landmarks = process_image(img_path, mp_hands)
                if landmarks is not None:
                    data_list.append({'label': label, 'landmarks': landmarks})

    df = pd.DataFrame(data_list)
    os.makedirs('data/combined_dataset', exist_ok=True)
    df.to_pickle('data/combined_dataset/dataset.pkl')
    print(f"\n✅ Saved {len(df)} samples to dataset.pkl")

if __name__ == "__main__":
    process_all_data()
