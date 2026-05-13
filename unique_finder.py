import os
from PIL import Image
import imagehash

def remove_duplicates_advanced(folder_path, threshold=5, dry_run=True):
    # We store a tuple of (dhash, phash)
    hashes = {}
    deleted_count = 0
    
    if not os.path.exists(folder_path):
        print(f"Error: Path {folder_path} does not exist.")
        return

    print(f"Scanning medical dataset: {folder_path}...")

    for filename in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, filename)
        if os.path.isdir(path): continue

        try:
            with Image.open(path) as img:
                # 1. Generate two types of hashes
                d_hash = imagehash.dhash(img)
                p_hash = imagehash.phash(img)
            
            is_duplicate = False
            matched_with = ""

            # 2. Compare against existing library
            for (old_d, old_p), old_name in hashes.items():
                # Check if BOTH structural and frequency hashes match
                if (d_hash - old_d <= threshold) and (p_hash - old_p <= threshold):
                    is_duplicate = True
                    matched_with = old_name
                    break
            
            if is_duplicate:
                deleted_count += 1
                if dry_run:
                    print(f"[MATCH] {filename} is visually identical to {matched_with}")
                else:
                    os.remove(path)
                    print(f"Deleted: {filename}")
            else:
                hashes[(d_hash, p_hash)] = filename
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nScan complete. Found {deleted_count} duplicates.")

# --- EXECUTION ---
dataset_path = r"C:\Users\User\Personal Projects\Final_Segmented_Train_Dataset\diabetes"
# Start with a strict threshold (like 2 or 3) for this specific tongue dataset
remove_duplicates_advanced(dataset_path, threshold=20, dry_run=True)