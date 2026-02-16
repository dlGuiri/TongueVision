import cv2
import numpy as np
import os
import torch
from rembg import remove, new_session
from datetime import datetime
from PIL import Image
import io

# Configuration
INPUT_DIR = r"C:\Users\User\Downloads\preprocessedcropped-20240821T085241Z-001 (1)\preprocessedcropped\valid"
OUTPUT_DIR = "Final_Segmented_Val_Dataset"

# ============== GPU VERIFICATION ==============
print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"✓ Using GPU acceleration")
else:
    print("✗ GPU not available - using CPU")
    providers = ['CPUExecutionProvider']

print("=" * 60)
print()

# Initialize RemBG session
print("Loading RemBG model...")
session = new_session(providers=providers)
print("✓ Model loaded successfully\n")

# ============== COUNT TOTAL IMAGES ==============
total_images = 0
category_counts = {}

for category in ["diabetes", "nondiabetes"]:
    path = os.path.join(INPUT_DIR, category)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        category_counts[category] = count
        total_images += count
    else:
        category_counts[category] = 0
        print(f"⚠ Warning: {path} does not exist")

print(f"Total images to process: {total_images}")
for cat, count in category_counts.items():
    print(f"  - {cat}: {count} images")
print()

# ============== PROCESSING WITH PROGRESS ==============
processed_count = 0
start_time = datetime.now()

for category in ["diabetes", "nondiabetes"]:
    path = os.path.join(INPUT_DIR, category)
    out_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(out_path, exist_ok=True)
    
    if not os.path.exists(path):
        continue
    
    images_in_category = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    category_total = len(images_in_category)
    
    print(f"\n{'=' * 60}")
    print(f"Processing category: {category} ({category_total} images)")
    print(f"{'=' * 60}")
    
    for idx, img_name in enumerate(images_in_category, 1):
        try:
            img_path = os.path.join(path, img_name)
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"  ✗ Skipped (invalid): {img_name}")
                continue
            
            # Convert BGR to RGB for RemBG
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Remove background using RemBG
            output = remove(pil_image, session=session)
            
            # Convert back to numpy array
            output_np = np.array(output)
            
            # RemBG returns RGBA, extract RGB and alpha channel
            rgb = output_np[:, :, :3]
            alpha = output_np[:, :, 3]
            
            # Create mask from alpha channel
            mask = alpha > 0
            
            # Apply mask: Set everything outside the mask to BLACK
            clean_img = image.copy()
            clean_img[~mask] = 0
            
            # Save result
            cv2.imwrite(os.path.join(out_path, img_name), clean_img)
            
            # Update progress
            processed_count += 1
            percentage = (processed_count / total_images) * 100
            images_left = total_images - processed_count
            
            # Calculate ETA
            elapsed = (datetime.now() - start_time).total_seconds()
            if processed_count > 0:
                avg_time_per_image = elapsed / processed_count
                eta_seconds = avg_time_per_image * images_left
                eta_minutes = eta_seconds / 60
                eta_str = f"{int(eta_minutes)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "calculating..."
            
            # Print progress
            print(f"  [{idx}/{category_total}] {img_name[:40]:<40} | "
                  f"Overall: {processed_count}/{total_images} ({percentage:.1f}%) | "
                  f"Left: {images_left} | ETA: {eta_str}")
        
        except Exception as e:
            print(f"  ✗ Error processing {img_name}: {str(e)}")
            continue

# ============== COMPLETION SUMMARY ==============
end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()

print("\n" + "=" * 60)
print("PROCESSING COMPLETE!")
print("=" * 60)
print(f"✓ Total images processed: {processed_count}/{total_images}")
print(f"✓ Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
if processed_count > 0:
    print(f"✓ Average time per image: {total_time/processed_count:.2f}s")
print(f"✓ Output directory: {OUTPUT_DIR}")
if torch.cuda.is_available():
    print(f"✓ Device used: GPU ({torch.cuda.get_device_name(0)})")
else:
    print(f"✓ Device used: CPU")
print("=" * 60)