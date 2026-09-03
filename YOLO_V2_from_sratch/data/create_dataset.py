import os
import csv
import argparse
import urllib.request
import tarfile
import zipfile
import shutil

from pathlib import Path
from tqdm import tqdm


def init():
    """
        Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="YOLO V2 Dataset Preparation Script", formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
                    Examples:
                    # Download and prepare dataset
                    python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv
                    
                    # Use existing dataset (skip download)
                    python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv --no_download
                    
                    # Skip validation (faster)
                    python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv --no_validate
            """)
    
    parser.add_argument('--voc_root', 
                        type=str, 
                        required=True, 
                        help='Path to VOC2012 folder (e.g., ./data/VOCdevkit/VOC2012)')
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./data',
                        help='Output directory for CSV files (default: ./data)')
    
    parser.add_argument('--no_download', 
                        action='store_true',
                        help='Skip downloading dataset if not found')
    
    parser.add_argument('--no_validate', 
                        action='store_true',
                        help='Skip validation of dataset structure')
    
    parser.add_argument('--download_only', 
                        action='store_true',
                        help='Only download dataset, don\'t create CSV files')
    
    return parser.parse_args()


class DownloadProgressBar(tqdm):
    """Custom progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, description="Downloading"):
    """Download a file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, 
                            miniters=1, desc=description) as t:
        urllib.request.urlretrieve(url, filename=output_path, 
                                  reporthook=t.update_to)


def download_voc_dataset(output_dir="./data"):
    """
        Download and extract Pascal VOC 2012 dataset
        
        Args:
            output_dir: Directory to download and extract to
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for VOC 2012
    voc_train_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    voc_test_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOC2012test.tar"
    
    # Download training/validation data
    train_tar_path = output_dir / "VOCtrainval_11-May-2012.tar"
    if not train_tar_path.exists():
        print("Downloading VOC 2012 train/val dataset...")
        download_file(voc_train_url, str(train_tar_path), "Downloading train/val")
    else:
        print("Train/val tar file already exists, skipping download...")
    
    # Download test data (optional)
    test_tar_path = output_dir / "VOC2012test.tar"
    if not test_tar_path.exists():
        print("\nDownloading VOC 2012 test dataset...")
        download_file(voc_test_url, str(test_tar_path), "Downloading test")
    else:
        print("Test tar file already exists, skipping download...")
    
    # Extract training/validation data
    voc_root = output_dir / "VOCdevkit" / "VOC2012"
    if not voc_root.exists():
        print("\nExtracting train/val dataset...")
        with tarfile.open(train_tar_path, "r") as tar:
            # Extract with progress
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting train/val"):
                tar.extract(member, path=output_dir)
        print(f"Extracted to {voc_root}")
    else:
        print(f"VOC root already exists at {voc_root}")
    
    # Extract test data (optional)
    test_extract_path = output_dir / "VOC2012test"
    if not test_extract_path.exists() and test_tar_path.exists():
        print("\nExtracting test dataset...")
        with tarfile.open(test_tar_path, "r") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting test"):
                tar.extract(member, path=output_dir)
        print(f"Extracted test to {test_extract_path}")
    
    return voc_root


def validate_voc_dataset(voc_root_path):
    """
    Validate that the VOC dataset is properly structured
    
    Args:
        voc_root_path: Path to VOC2012 folder
    """
    required_dirs = ['Annotations', 'ImageSets', 'JPEGImages']
    for dir_name in required_dirs:
        dir_path = Path(voc_root_path) / dir_name
        if not dir_path.exists():
            raise ValueError(f"Required directory {dir_path} not found in VOC dataset")
    
    # Check for train.txt
    main_dir = Path(voc_root_path) / 'ImageSets' / 'Main'
    for split_file in ['train.txt', 'val.txt', 'test.txt']:
        file_path = main_dir / split_file
        if not file_path.exists():
            print(f"Warning: {split_file} not found in {main_dir}")
    
    print(f"VOC dataset validated at {voc_root_path}")
    return True


def run(voc_root_path, output_dir, download=True, validate=True):
    """
    Prepare Pascal VOC dataset for YOLO training
    
    Args:
        voc_root_path: Path to VOCdevkit/VOC2012 folder
        output_dir: Where to save CSV files
        download: Whether to download dataset if not found
        validate: Whether to validate dataset structure
    """
    voc_root = Path(voc_root_path)
    
    # Check if VOC dataset exists, download if not
    if not voc_root.exists() and download:
        print(f"VOC dataset not found at {voc_root}. Downloading...")
        voc_root = download_voc_dataset(Path(output_dir).parent)
        voc_root_path = str(voc_root)
    
    # Validate dataset structure
    if validate:
        try:
            validate_voc_dataset(voc_root_path)
        except ValueError as e:
            print(f"Error: {e}")
            if download:
                print("Attempting to re-download dataset...")
                shutil.rmtree(voc_root.parent if 'VOCdevkit' in str(voc_root) else voc_root)
                voc_root = download_voc_dataset(Path(output_dir).parent)
                voc_root_path = str(voc_root)
            else:
                raise
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define splits
    splits = {
        'train': 'train.txt',
        'val': 'val.txt',
        'test': 'test.txt'
    }
    
    image_sets_dir = Path(voc_root_path) / 'ImageSets' / 'Main'
    
    for split_name, split_file in splits.items():
        split_file_path = image_sets_dir / split_file
        
        # Check if split file exists
        if not split_file_path.exists():
            print(f"Warning: {split_file_path} not found. Skipping {split_name} split.")
            continue
        
        # Read image names
        with open(split_file_path) as f:
            image_names = [line.strip() for line in f.readlines() if line.strip()]
        
        if not image_names:
            print(f"Warning: No images found in {split_file}. Skipping {split_name} split.")
            continue
        
        # Create CSV entries
        entries = []
        for image_name in image_names:
            image_file = f"{image_name}.jpg"
            label_file = f"{image_name}.txt"
            entries.append([image_file, label_file])
        
        # Save to CSV
        csv_path = Path(output_dir) / f"{split_name}.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(entries)
        
        print(f"Created {csv_path} with {len(entries)} entries")
    
    # Create a summary file
    summary_path = Path(output_dir) / "dataset_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("VOC Dataset Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"VOC Root: {voc_root_path}\n")
        f.write(f"Output Dir: {output_dir}\n\n")
        
        for split_name in splits.keys():
            csv_path = Path(output_dir) / f"{split_name}.csv"
            if csv_path.exists():
                with open(csv_path, 'r') as csvfile:
                    count = sum(1 for _ in csvfile)
                f.write(f"{split_name}: {count} images\n")
            else:
                f.write(f"{split_name}: Not created\n")
    
    print(f"\nDataset preparation complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    args = init()
    
    # Handle download only mode
    if args.download_only:
        print("Downloading VOC dataset only...")
        download_voc_dataset(Path(args.output_dir).parent)
        print(f"Dataset downloaded to {Path(args.output_dir).parent / 'VOCdevkit' / 'VOC2012'}")
        exit(0)
    
    # Run the full preparation
    run(voc_root_path=args.voc_root,
        output_dir=args.output_dir,
        download=not args.no_download,
        validate=not args.no_validate)