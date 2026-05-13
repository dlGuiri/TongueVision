import os
import random
from PIL import Image, ImageEnhance, ImageFilter

def augment_medical_image(image):
    """
    Applies a randomized combination of clinically safe augmentations
    suitable for tongue color and texture analysis.
    """
    w, h = image.size

    # 1. Mild Camera Distance Variation (Zoom / Crop)
    # Simulates camera distance by cropping to 90%-100% of the image and resizing.
    # Applied 50% of the time.
    if random.random() > 0.5:
        scale = random.uniform(0.90, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        image = image.crop((left, top, left + new_w, top + new_h))
        image = image.resize((w, h), resample=Image.LANCZOS)

    # 2. Horizontal Flip (50% probability)
    # The tongue is relatively symmetrical, so horizontal flips preserve clinical meaning
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 3. Slight Rotation (-15 to +15 degrees)
    # Simulates natural head tilt or camera angle variations
    angle = random.uniform(-15, 15)
    image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
    
    # 4. Mild Translation (Centering Variations)
    # Shifts the image by up to +/- 5% on the X and Y axes to simulate off-center framing.
    if random.random() > 0.5:
        tx = random.uniform(-0.05 * w, 0.05 * w)
        ty = random.uniform(-0.05 * h, 0.05 * h)
        # PIL Affine transformation maps pixels from destination back to source.
        # Matrix: (1, 0, -tx, 0, 1, -ty) translates the image by tx, ty.
        image = image.transform((w, h), Image.AFFINE, (1, 0, -tx, 0, 1, -ty), resample=Image.BILINEAR)

    # 5. Subtle Brightness & Contrast Shifts (+/- 15%)
    # Crucial for color analysis; simulates different clinic lighting
    bright_enhancer = ImageEnhance.Brightness(image)
    image = bright_enhancer.enhance(random.uniform(0.85, 1.15))
    
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(random.uniform(0.85, 1.15))

    # 6. Mild Blur (20% probability)
    # Simulates slight out-of-focus shots or minor patient tremors.
    if random.random() < 0.2:
        radius = random.uniform(0.5, 1.5)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
    return image

def process_training_folder(train_directory, num_augmentations=5):
    """
    Scans the training directory, keeps the original image, and 
    generates 'num_augmentations' new images for each original.
    """
    print(f"Starting augmentation for: {train_directory}")
    
    # Supported image formats
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    
    total_originals = 0
    total_generated = 0

    for root, _, files in os.walk(train_directory):
        for filename in files:
            # Skip files that are already augmented (prevents infinite loops if re-run)
            if filename.startswith("aug_"):
                continue
                
            if filename.lower().endswith(valid_extensions):
                total_originals += 1
                file_path = os.path.join(root, filename)
                
                try:
                    with Image.open(file_path) as img:
                        # Convert to RGB in case some images have an alpha channel (RGBA)
                        img = img.convert('RGB')
                        
                        # Generate the requested number of augmentations
                        for i in range(1, num_augmentations + 1):
                            aug_img = augment_medical_image(img)
                            
                            # Create a clear, traceable filename
                            base_name, ext = os.path.splitext(filename)
                            aug_filename = f"aug_{i}_{base_name}{ext}"
                            aug_filepath = os.path.join(root, aug_filename)
                            
                            aug_img.save(aug_filepath, quality=95)
                            total_generated += 1
                            
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    print("-" * 30)
    print("Augmentation Complete!")
    print(f"Original images processed: {total_originals}")
    print(f"New augmented images generated: {total_generated}")
    print(f"Total training images now available: {total_originals + total_generated}")

# --- EXECUTION ---
# IMPORTANT: Point this ONLY to your 'train' folder. 
# Do NOT run this on your 'val' or 'test' folders.

train_dataset_path = r"C:\Users\User\Personal Projects\Dataset For TongueVision\train"

# Run the function
process_training_folder(train_dataset_path, num_augmentations=5)