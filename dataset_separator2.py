import os
import shutil
import random

def create_split(source_base, dest_base, val_ratio=0.20):
    """
    Randomly splits a dataset into train and val folders based on a percentage.
    """
    categories = ['diabetes', 'healthy']
    splits = ['train', 'val']

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

        if total_files == 0:
            print(f"  Warning: No files found in {category}. Skipping.")
            continue

        # 3. Calculate dynamic split sizes based on the ratio
        val_size = int(total_files * val_ratio)
        
        # Slice the lists
        val_files = files[:val_size]
        train_files = files[val_size:]

        # 4. Copy the files to their new homes
        # VAL
        for f in val_files:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(dest_base, 'val', category, f))
        # TRAIN
        for f in train_files:
            shutil.copy2(os.path.join(source_dir, f), os.path.join(dest_base, 'train', category, f))

        print(f"  -> Successfully copied: {len(train_files)} Train | {len(val_files)} Val\n")

    print("-" * 30)
    print("Dataset splitting is complete!")

# --- EXECUTION ---
# Using raw strings (r"...") for Windows paths
source_path = r"C:\Users\User\OneDrive\Documents\Chinese T2DM Dataset"
destination_path = r"C:\Users\User\Personal Projects\TMC Train and Val Dataset"

# Set a random seed for reproducibility 
random.seed(42)

# Run the function with a 20% validation split (0.20)
create_split(source_path, destination_path, val_ratio=0.20)