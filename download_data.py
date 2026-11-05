import os
import shutil
import kagglehub

def download_and_setup():
    print("Downloading HAM10000 dataset...")
    # Download dataset
    path = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
    print(f"Dataset downloaded to: {path}")

    # Destination directory
    dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(dest_dir, exist_ok=True)
    
    # Check for expected files
    expected_files = ['HAM10000_metadata.csv']
    has_metadata = False
    
    for root, _, files in os.walk(path):
        for file in files:
            if file == 'HAM10000_metadata.csv':
                has_metadata = True
                break
                
    if not has_metadata:
        print("Warning: HAM10000_metadata.csv not found in downloaded files.")
    else:
        print("Verified dataset contains metadata.")

    # Create symlink to data directory
    dataset_link = os.path.join(dest_dir, 'ham10000')
    if os.path.exists(dataset_link):
        if os.path.islink(dataset_link):
            os.unlink(dataset_link)
        else:
            shutil.rmtree(dataset_link)
            
    try:
        os.symlink(path, dataset_link)
        print(f"Created symlink: {dataset_link} -> {path}")
    except OSError:
        # Fallback to copy if symlink fails
        print("Symlink failed, copying files instead...")
        shutil.copytree(path, dataset_link)
        print(f"Copied files to {dataset_link}")
        
    print("Download and setup complete!")

if __name__ == "__main__":
    download_and_setup()
