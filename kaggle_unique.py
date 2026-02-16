import os
import re

def clean_dataset(directory_path):
    """
    Scans a directory for Roboflow-style exported images, keeps one unique 
    instance per patient/image ID, and removes the augmented duplicates.
    """
    # Regex to capture the base filename before the .rf. tag
    # Matches: "d_-100-_jpg.rf.5176....jpg" -> Group 1: "d_-100-_jpg"
    pattern = re.compile(r"^(.*?)\.rf\.[a-f0-9]+\.(jpg|jpeg|png|bmp|tif)$", re.IGNORECASE)
    
    # Track the unique base names we have already processed
    seen_bases = set()
    deleted_count = 0
    kept_count = 0

    # Verify directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return

    print(f"Scanning: {directory_path} ...")

    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            match = pattern.match(filename)
            
            # Only process files that match the Roboflow .rf. pattern
            if match:
                base_name = match.group(1)
                file_path = os.path.join(root, filename)

                if base_name in seen_bases:
                    # We have already seen this base name, so this is a duplicate/augmentation
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        # Optional: Print deleted file (comment out if too noisy)
                        # print(f"Removed: {filename}")
                    except OSError as e:
                        print(f"Error deleting {filename}: {e}")
                else:
                    # This is the first time we see this base name; keep it
                    seen_bases.add(base_name)
                    kept_count += 1

    print("-" * 30)
    print("Cleanup Complete.")
    print(f"Unique images preserved: {kept_count}")
    print(f"Augmented duplicates removed: {deleted_count}")

# --- EXECUTION ---
# strict usage of raw string (r"...") for Windows paths to handle backslashes
dataset_path = r"C:\Users\User\Downloads\Kaggle_Cleaned_Dataset"

# Run the function
clean_dataset(dataset_path)