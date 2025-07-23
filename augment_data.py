import numpy as np
import pandas as pd
import os

def augment_dataset():
    print("\n🚀 Starting advanced data augmentation...")

    try:
        dataset_path = 'data/combined_dataset/dataset.pkl'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"{dataset_path} not found.")

        df = pd.read_pickle(dataset_path)
        print(f"✅ Loaded {len(df)} original samples")

        augmented_data = []

        for idx, row in df.iterrows():
            landmarks = row['landmarks']
            label = row['label']

            # Check if 3D or 2D landmarks
            num_coords = len(landmarks)
            if num_coords == 42:  # 2D: (x, y) * 21
                landmarks = landmarks.reshape(21, 2)
                mode = '2D'
            elif num_coords == 63:  # 3D: (x, y, z) * 21
                landmarks = landmarks.reshape(21, 3)
                mode = '3D'
            else:
                print(f"❌ Skipped invalid landmark shape at index {idx}")
                continue

            # 1. Mirrored version (flip x-axis)
            mirrored = landmarks.copy()
            mirrored[:, 0] = 1.0 - mirrored[:, 0]
            augmented_data.append({'label': label, 'landmarks': mirrored.flatten()})

            # 2. Add Gaussian noise (jittering)
            for _ in range(3):  # More jittered versions
                noise = np.random.normal(0, 0.01, landmarks.shape)
                jittered = landmarks + noise
                augmented_data.append({'label': label, 'landmarks': jittered.flatten()})

            # 3. Simulate brightness variation by slightly distorting all points (simulate low-contrast hand blur)
            distortion = np.random.uniform(-0.005, 0.005, landmarks.shape)
            brightness_shifted = landmarks + distortion
            augmented_data.append({'label': label, 'landmarks': brightness_shifted.flatten()})

        # Combine with original data
        print("🔀 Combining with original dataset...")
        augmented_df = pd.DataFrame(augmented_data)
        final_df = pd.concat([df, augmented_df], ignore_index=True)

        # Save result
        save_path = 'data/combined_dataset/augmented_dataset.pkl'
        final_df.to_pickle(save_path)

        print(f"✅ Augmentation complete! Total samples: {len(final_df)} (from {len(df)})")
        return True

    except Exception as e:
        print(f"❌ Error during augmentation: {e}")
        return False


if __name__ == "__main__":
    augment_dataset()