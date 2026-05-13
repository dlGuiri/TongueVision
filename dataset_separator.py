import os
import shutil
import random

def create_split(source_base, dest_base, val_size=25, test_size=50):
    """
    Randomly splits a dataset into train, val, and test folders.
    """
    categories = ['diabetes', 'healthy']
    splits = ['train', 'val', 'test']

    # 1. Create the destination directory structure
    for split in splits:
        for category in categories:
            os.makedirs(os.path.join(dest_base, split, category), exist_ok=True)
            
    print(f"Directory structure created at: {dest_base}\n")

    # 2. Process each category
    for category in categories:
        source_dir = os.path.join(source_base, category)
        
        if not os.path.exists(source_dir):
            print(f"Error: Source directory not found -> {source_dir}")
            continue

        # Get all files and shuffle them randomly
        files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        random.shuffle(files)
        
        total_files = len(files)
        print(f"Processing '{category}' category: Found {total_files} total images.")

        if total_files < (val_size + test_size):
            print(f"  Warning: Not enough files in {category} to meet val/test requirements.")
            continue

        # 3. Slice the lists based on your required counts
        test_files = files[:test_size]
        val_files = files[test_size : test_size + val_size]
        train_files = files[test_size + val_size:]

        # 4. Copy the files to their new homes
        # TEST
        for f in test_files:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(dest_base, 'test', category, f))
        # VAL
        for f in val_files:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(dest_base, 'val', category, f))
        # TRAIN
        for f in train_files:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(dest_base, 'train', category, f))

        print(f"  -> Successfully copied: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test\n")

    print("-" * 30)
    print("Dataset splitting is complete!")

# --- EXECUTION ---
# Using raw strings (r"...") for Windows paths
source_path = r"C:\Users\User\Personal Projects\DMT and Mendeley Datasets"
destination_path = r"C:\Users\User\Personal Projects\Dataset For TongueVision"

# Set a random seed for reproducibility (so if you run it again, it splits the exact same way)
random.seed(42)

# Run the function
create_split(source_path, destination_path, val_size=25, test_size=50)