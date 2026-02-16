"""
Script to delete Kaggle images from the Master Dataset
Keeps only Mendeley images (those starting with 'r_d_' or 'r_nd_')
"""

import os
import re
from pathlib import Path

# Define the master dataset path
MASTER_DATASET_PATH = r"C:\Users\User\Personal Projects\Augmented Diabetes Tongue Dataset"

def is_kaggle_file(filename):
    """
    Check if a file is from Kaggle dataset.
    
    Kaggle files have patterns like:
    - d_-XXX-_jpg.rf.HASH.jpg (diabetes)
    - nd_-XXX-_jpg.rf.HASH.jpg (healthy)
    - IMGXXXXXXX_jpg.rf.HASH.jpg (healthy)
    
    Mendeley files start with:
    - r_d_ (diabetes)
    - r_nd_ (healthy)
    """
    # Kaggle files contain '.rf.' pattern
    if '.rf.' in filename:
        return True
    
    # Additional pattern check: starts with 'd_-' or 'nd_-' or 'IMG'
    if re.match(r'^(d_-|nd_-|IMG)', filename):
        return True
    
    return False

def is_mendeley_file(filename):
    """
    Check if a file is from Mendeley dataset.
    Mendeley files start with 'r_d_' or 'r_nd_'
    """
    return filename.startswith('r_d_') or filename.startswith('r_nd_')

def delete_kaggle_images(base_path, dry_run=True):
    """
    Delete all Kaggle images from the dataset.
    
    Args:
        base_path: Path to the master dataset
        dry_run: If True, only show what would be deleted without deleting
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_files': 0,
        'kaggle_deleted': 0,
        'mendeley_kept': 0,
        'other_files': 0,
        'diabetes_kaggle': 0,
        'healthy_kaggle': 0
    }
    
    folders = ['Diabetes', 'Healthy']
    
    for folder in folders:
        folder_path = Path(base_path) / folder
        
        if not folder_path.exists():
            print(f"⚠️  Warning: Folder '{folder}' not found at {folder_path}")
            continue
        
        print(f"\n📁 Processing folder: {folder}")
        print(f"   Path: {folder_path}")
        print("-" * 70)
        
        # Get all files in the folder
        files = [f for f in folder_path.iterdir() if f.is_file()]
        stats['total_files'] += len(files)
        
        for file_path in files:
            filename = file_path.name
            
            if is_kaggle_file(filename):
                stats['kaggle_deleted'] += 1
                
                if folder == 'Diabetes':
                    stats['diabetes_kaggle'] += 1
                else:
                    stats['healthy_kaggle'] += 1
                
                if dry_run:
                    print(f"   [DRY RUN] Would delete: {filename}")
                else:
                    try:
                        file_path.unlink()
                        print(f"   ✓ Deleted: {filename}")
                    except Exception as e:
                        print(f"   ✗ Error deleting {filename}: {e}")
            
            elif is_mendeley_file(filename):
                stats['mendeley_kept'] += 1
                # Don't print kept files to reduce clutter
            
            else:
                stats['other_files'] += 1
                print(f"   ⚠️  Unknown file pattern: {filename}")
    
    return stats

def print_summary(stats, dry_run=True):
    """Print summary of the operation"""
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total files found:          {stats['total_files']}")
    print(f"Kaggle images (to delete):  {stats['kaggle_deleted']}")
    print(f"  - From Diabetes folder:   {stats['diabetes_kaggle']}")
    print(f"  - From Healthy folder:    {stats['healthy_kaggle']}")
    print(f"Mendeley images (to keep):  {stats['mendeley_kept']}")
    print(f"Other/Unknown files:        {stats['other_files']}")
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No files were actually deleted")
        print("   Run with dry_run=False to perform actual deletion")
    else:
        print("\n✓ Files have been deleted")
    print("=" * 70)

def main():
    """Main function"""
    print("=" * 70)
    print("🗑️  KAGGLE IMAGE DELETION SCRIPT")
    print("=" * 70)
    print(f"Master Dataset Path: {MASTER_DATASET_PATH}")
    print()
    
    # Check if path exists
    if not Path(MASTER_DATASET_PATH).exists():
        print(f"❌ Error: Path does not exist: {MASTER_DATASET_PATH}")
        print("\nPlease update the MASTER_DATASET_PATH variable in the script")
        return
    
    # First run in dry-run mode to show what would be deleted
    print("\n🔍 STEP 1: DRY RUN (Preview)")
    print("Analyzing files to identify Kaggle images...")
    stats = delete_kaggle_images(MASTER_DATASET_PATH, dry_run=True)
    print_summary(stats, dry_run=True)
    
    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("\n⚠️  Do you want to proceed with deletion? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n🗑️  STEP 2: ACTUAL DELETION")
        print("Deleting Kaggle images...")
        stats = delete_kaggle_images(MASTER_DATASET_PATH, dry_run=False)
        print_summary(stats, dry_run=False)
    else:
        print("\n❌ Deletion cancelled. No files were deleted.")

if __name__ == "__main__":
    main()