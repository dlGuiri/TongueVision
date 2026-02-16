import os
import re

target_folders = [
    r"C:\Users\User\Downloads\preprocessedcropped-20240821T085241Z-001 (1)\preprocessedcropped\valid\diabetes",
    r"C:\Users\User\Downloads\preprocessedcropped-20240821T085241Z-001 (1)\preprocessedcropped\valid\nondiabetes"
]

def get_base_id(filename):
    """
    Extracts the unique ID by stripping the augmentation suffix.
    
    Logic:
    Everything BEFORE the first occurrence of '_flip', '_resized', or '_rotate' is the ID.
    
    Examples:
    'r_d_(10)_flip.jpg'      -> ID: 'r_d_(10)'
    'r_d_(19)c_flip.jpg'     -> ID: 'r_d_(19)c'  <-- Now correctly identified as distinct
    'r_d_(19)_flip.jpg'      -> ID: 'r_d_(19)'   <-- Now correctly identified as distinct
    """
    # Split the string at any of the known keywords
    # We use valid keywords that appear in your file lists
    split_result = re.split(r'(_flip|_resized|_rotate)', filename)
    
    # Return the first part of the split (the base name)
    if split_result:
        return split_result[0]
    return filename # Fallback

def clean_directory(folder_path):
    print(f"--- Processing: {folder_path} ---")
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # 1. Group files by their Base ID
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    groups = {}

    for f in files:
        base_id = get_base_id(f)
        if base_id:
            if base_id not in groups:
                groups[base_id] = []
            groups[base_id].append(f)
    
    deleted_count = 0

    # 2. Process each group
    for base_id, file_list in groups.items():
        # If there is only 1 file, no need to delete anything
        if len(file_list) < 2:
            continue

        # Logic: Find the file to KEEP
        file_to_keep = None
        
        # Priority A: Look for "flip"
        flip_version = next((f for f in file_list if "_flip" in f), None)
        
        # Priority B: Look for "resized" (Safety net if flip is missing)
        resized_version = next((f for f in file_list if "_resized" in f), None)

        if flip_version:
            file_to_keep = flip_version
        elif resized_version:
            file_to_keep = resized_version
        else:
            # If neither exist, just keep the first one found
            file_to_keep = file_list[0]

        # 3. Delete the others
        for f in file_list:
            if f != file_to_keep:
                file_path = os.path.join(folder_path, f)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {f}: {e}")

    print(f"Finished. Deleted {deleted_count} files in this folder.\n")

if __name__ == "__main__":
    for folder in target_folders:
        clean_directory(folder)