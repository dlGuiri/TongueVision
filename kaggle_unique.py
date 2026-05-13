import os

def clean_dataset(directory_path):
    """
    Scans a directory and keeps ONLY files that contain 'resized' in their filename.
    All other augmentations (flip, rotate, etc.) are removed.
    """
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
            # We'll stick to checking standard image extensions just to be safe
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                file_path = os.path.join(root, filename)

                # Check if 'resized' is missing from the filename
                if "resized" not in filename.lower():
                    # It's a flip, rotate, or other augmentation -> delete it
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        # Optional: Print deleted file (comment out if too noisy)
                        # print(f"Removed: {filename}")
                    except OSError as e:
                        print(f"Error deleting {filename}: {e}")
                else:
                    # It has 'resized' in the name -> keep it
                    kept_count += 1

    print("-" * 30)
    print("Cleanup Complete.")
    print(f"Resized images preserved: {kept_count}")
    print(f"Augmented duplicates removed: {deleted_count}")

# --- EXECUTION ---
# strict usage of raw string (r"...") for Windows paths to handle backslashes
dataset_path = r"C:\Users\User\OneDrive\Documents\Final_Segmented_Train_Dataset\nondiabetes"

# Run the function
clean_dataset(dataset_path)