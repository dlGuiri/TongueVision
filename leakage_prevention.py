import os
import shutil
import re
import random

# --- CONFIGURATION ---
# UPDATE THIS PATH to where your Mendeley data is
MASTER_DIR = r"C:\Users\User\Personal Projects\Augmented Diabetes Tongue Dataset"
OUTPUT_DIR = "TongueFusion_Mendeley_TrainVal"

# Set Test to 0.0 because you are using Kaggle for testing!
TEST_RATIO = 0.0 
# You can increase Val ratio slightly since Test is gone (e.g., 0.2 = 20%)
VAL_RATIO = 0.20 

def get_base_name(filename):
    """
    Identifies the patient ID so originals and augmentations stay together.
    """
    name = os.path.splitext(filename)[0]
    name = re.split(r'(_rotate|_flip|_resized|\.rf|_Copy)', name)[0]
    if name.endswith('_jpg'): name = name[:-4]
    return name.strip()

# Initialize containers
data_map = {"Healthy": set(), "Diabetes": set()}

print("Scanning Master folder...")
for cls in ["Healthy", "Diabetes"]:
    class_path = os.path.join(MASTER_DIR, cls)
    if not os.path.exists(class_path): continue
    
    for f in os.listdir(class_path):
        base = get_base_name(f)
        data_map[cls].add(base)

# Create Splits
for cls, bases in data_map.items():
    base_list = list(bases)
    random.shuffle(base_list)
    
    total = len(base_list)
    # Test index is 0, so we skip straight to Val split
    test_idx = int(total * TEST_RATIO) 
    val_idx = test_idx + int(total * VAL_RATIO)
    
    # Logic: No Test set. Val set gets the chunk. Train set gets the rest.
    val_set = set(base_list[:val_idx]) 
    
    print(f"Processing {cls}: {total} unique patients found.")
    print(f"   -> Assigning {len(val_set)} patients to Validation.")
    print(f"   -> Assigning {total - len(val_set)} patients to Training.")

    for s in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, s, cls), exist_ok=True)

    master_class_path = os.path.join(MASTER_DIR, cls)
    for f in os.listdir(master_class_path):
        base = get_base_name(f)
        src = os.path.join(master_class_path, f)
        
        # A. VALIDATION SET
        if base in val_set:
            shutil.copy(src, os.path.join(OUTPUT_DIR, 'val', cls, f))

        # B. TRAINING SET (Default)
        else:
            shutil.copy(src, os.path.join(OUTPUT_DIR, 'train', cls, f))

print(f"\nSuccess! Your leakage-free Train/Val dataset is in: {OUTPUT_DIR}")