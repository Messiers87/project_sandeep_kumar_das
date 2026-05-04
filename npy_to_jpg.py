import numpy as np
import os
from PIL import Image


DATA_DIR = 'data'
X_DATA_PATH = os.path.join(DATA_DIR, 'X_test.npy')
Y_DATA_PATH = os.path.join(DATA_DIR, 'y_test.npy')
SAMPLES_PER_CLASS = 10

# CLASS MAPPING
class_map = {
    0: "no_substructure",
    1: "subhalo_substructure",
    2: "vortex_substructure"
}

def save_samples_in_place():
    print(f"Looking for data in: {DATA_DIR}...")
    
    # Check if the .npy files exist inside the data folder
    if not os.path.exists(X_DATA_PATH) or not os.path.exists(Y_DATA_PATH):
        print(f"Error: Could not find .npy files at {X_DATA_PATH}")
        return

    # Load data
    x = np.load(X_DATA_PATH)
    y = np.load(Y_DATA_PATH)

    # Handle One-Hot labels
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    for class_id, folder_name in class_map.items():
        # Path to the specific category folder inside 'data'
        target_path = os.path.join(DATA_DIR, folder_name)
        
        # Ensure the subfolders exist inside 'data'
        os.makedirs(target_path, exist_ok=True)

        # Get first 10 indices
        indices = np.where(y == class_id)[0][:SAMPLES_PER_CLASS]
        
        print(f"Saving 10 samples to {target_path}...")

        for i, idx in enumerate(indices):
            # Squeeze to handle (1, 100, 100) or (100, 100, 1) shapes
            img_array = np.squeeze(x[idx])

            # Normalization 0-255
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_rescaled = 255 * (img_array - img_min) / (img_max - img_min)
            else:
                img_rescaled = img_array

            # Save as JPG
            img = Image.fromarray(img_rescaled.astype(np.uint8))
            save_path = os.path.join(target_path, f'sample_{i}.jpg')
            img.save(save_path)

    print(f"\nDone! Check your '{DATA_DIR}' subfolders for the JPGs.")

if __name__ == "__main__":
    save_samples_in_place()