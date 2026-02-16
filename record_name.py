import os

def save_filenames_to_txt(folder_path, output_txt_file):
    """Gets all image names from a folder and saves them to a .txt file."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    
    # Check if the folder exists first
    if not os.path.exists(folder_path):
        print(f"⚠ Warning: Folder not found: {folder_path}")
        return []

    # Get the filenames into an array
    image_array = [
        f for f in os.listdir(folder_path) 
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(folder_path, f))
    ]
    
    # Write the array to a text file
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for name in image_array:
            f.write(f"{name}\n")
            
    print(f"✓ Saved {len(image_array)} names from '{os.path.basename(folder_path)}' to: {output_txt_file}")
    return image_array

# --- Configuration ---
base_dir = r"C:\Users\User\Downloads\preprocessedcropped-20240821T085241Z-001 (1)\preprocessedcropped\valid"
categories = ["diabetes", "nondiabetes"]

# --- Execution ---
for cat in categories:
    # Build the path to the images (e.g., .../Healthy)
    folder_path = os.path.join(base_dir, cat)
    
    # Build the name for the text file (e.g., Healthy_filenames.txt)
    output_filename = f"{cat}_filenames.txt"
    
    # Run the function
    save_filenames_to_txt(folder_path, output_filename)