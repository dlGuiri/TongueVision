import os
import re
import random
import shutil

def split_dataset(train_dir, test_dir, num_patients_to_move=50):
    """
    Moves the '_resized' image for selected patients to the test directory 
    and deletes their associated augmented images from the train directory.
    """
    # Ensure the target test directory exists
    os.makedirs(test_dir, exist_ok=True)

    # Get all files in the training directory
    try:
        files = os.listdir(train_dir)
    except FileNotFoundError:
        print(f"Error: Could not find directory {train_dir}")
        return

    # Extract unique patient IDs (looking for text inside parentheses)
    patient_ids = set()
    for f in files:
        match = re.search(r'\(([^)]+)\)', f)
        if match:
            patient_ids.add(match.group(1))

    patient_ids = list(patient_ids)

    # Check if we have enough patients
    if len(patient_ids) < num_patients_to_move:
        print(f"Warning: Only {len(patient_ids)} patients found in {train_dir}. Cannot move {num_patients_to_move}.")
        return

    # Randomly select the patients to move
    # (Optional: Set a seed like random.seed(42) before calling this function for reproducible splits)
    selected_patients = random.sample(patient_ids, num_patients_to_move)

    moved_count = 0
    deleted_count = 0

    # Process the files
    for f in files:
        match = re.search(r'\(([^)]+)\)', f)
        if match:
            pid = match.group(1)
            
            # If this file belongs to one of our selected test patients
            if pid in selected_patients:
                src_path = os.path.join(train_dir, f)
                
                if "_resized" in f:
                    # Move the original/resized image to the test folder
                    dst_path = os.path.join(test_dir, f)
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                else:
                    # Delete the augmentation to prevent data leakage
                    os.remove(src_path)
                    deleted_count += 1

    print(f"Successfully processed: {os.path.basename(train_dir)}")
    print(f"  -> Moved {moved_count} '_resized' images to test folder.")
    print(f"  -> Deleted {deleted_count} augmented images from train folder.\n")

# --- Set up your directory paths ---
base_dir = r"C:\Users\User\Personal Projects\TongueVision\preprocessedcropped-20240821T085241Z-001\preprocessedcropped"

# Train folders
train_nondiabetes = os.path.join(base_dir, "train", "nondiabetes")
train_diabetes = os.path.join(base_dir, "train", "diabetes")

# Test folders (these will be created automatically if they don't exist)
test_nondiabetes = os.path.join(base_dir, "test", "nondiabetes")
test_diabetes = os.path.join(base_dir, "test", "diabetes")

# --- Execute the split ---
print("Starting dataset split...\n")

# Set a random seed if you want the exact same patients chosen every time you run this specific split
random.seed(42) 

split_dataset(train_nondiabetes, test_nondiabetes, num_patients_to_move=50)
split_dataset(train_diabetes, test_diabetes, num_patients_to_move=50)

print("Dataset split complete!")