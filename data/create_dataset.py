import os
import csv
import time
import argparse
import http.client
import urllib.request
import urllib.error
import tarfile
import shutil

from pathlib import Path
from tqdm import tqdm


VOC_TRAIN_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_TEST_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOC2012test.tar"

# Progress bar shows percentage, size, elapsed time, TIME REMAINING and speed
BAR_FORMAT = ("{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
              "[elapsed {elapsed} | remaining {remaining} | {rate_fmt}]")


def init():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="YOLO V1 Dataset Preparation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv
  python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv --no_download
  python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv --no_validate
""")

    parser.add_argument('--voc_root', type=str, required=True,
                        help='Path to VOC2012 folder (e.g., ./data/VOCdevkit/VOC2012) or root folder (e.g., ./data)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for CSV files (default: ./data)')
    parser.add_argument('--no_download', action='store_true',
                        help='Skip downloading dataset if not found')
    parser.add_argument('--no_validate', action='store_true',
                        help='Skip validation of dataset structure')
    parser.add_argument('--download_only', action='store_true',
                        help="Only download dataset, don't create CSV files")
    return parser.parse_args()


# Downloading (resume + retry + ETA)
def get_remote_size(url):
    """Return file size in bytes from a HEAD request, or None if unknown."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            size = int(r.headers.get("Content-Length", 0))
            return size or None
    except Exception:
        return None


def download_file(url, output_path, description="Downloading", max_retries=200):
    """
    Download a file, resuming from a .part file after any interruption.
    Retries until the download is complete. The final filename only appears
    once the download is fully finished, so a truncated file can never be
    mistaken for a complete one.
    """
    output_path = Path(output_path)
    part_path = output_path.with_name(output_path.name + ".part")
    remote_size = get_remote_size(url)

    for attempt in range(1, max_retries + 1):
        try:
            done = part_path.stat().st_size if part_path.exists() else 0

            if remote_size and done > remote_size:      # bad partial file -> restart
                part_path.unlink()
                done = 0
            if remote_size and done == remote_size:     # already fully downloaded
                break

            req = urllib.request.Request(url)
            if done:
                req.add_header("Range", f"bytes={done}-")

            with urllib.request.urlopen(req, timeout=30) as resp:
                if done and resp.status != 206:         # server ignored Range -> restart
                    done = 0
                length = int(resp.headers.get("Content-Length", 0))
                total = (done + length) if length else remote_size

                with open(part_path, "ab" if done else "wb") as f, tqdm(
                        total=total, initial=done, unit="B", unit_scale=True,
                        unit_divisor=1024, miniters=1, smoothing=0.05,
                        desc=description, bar_format=BAR_FORMAT) as bar:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))

            size = part_path.stat().st_size
            if total and size < total:
                raise IOError(f"Incomplete download ({size}/{total} bytes)")
            break  # success

        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise RuntimeError(f"HTTP {e.code} for {url}: not retrying") from e
            print(f"\nAttempt {attempt}/{max_retries} failed: {e}. Resuming in 5s...")
            time.sleep(5)
        except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException) as e:
            print(f"\nAttempt {attempt}/{max_retries} failed: {e}. Resuming in 5s...")
            time.sleep(5)
    else:
        raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")

    part_path.replace(output_path)


def is_valid_tar(path, expected_size=None):
    """Check that a tar file is complete and readable."""
    path = Path(path)
    if expected_size and path.stat().st_size != expected_size:
        return False
    try:
        with tarfile.open(path, "r") as tar:
            tar.getmembers()
        return True
    except (tarfile.TarError, EOFError, OSError):
        return False


def ensure_tar(url, tar_path, description, max_cycles=5):
    """
    Make sure a complete, valid tar exists at tar_path.
    - If a corrupt/truncated tar is found, delete it (and any .part file)
      and download again from the beginning.
    - Interrupted downloads resume automatically until finished.
    """
    tar_path = Path(tar_path)
    part_path = tar_path.with_name(tar_path.name + ".part")
    expected_size = get_remote_size(url)

    for cycle in range(1, max_cycles + 1):
        if tar_path.exists():
            print(f"Checking {tar_path.name}...")
            if is_valid_tar(tar_path, expected_size):
                print("Archive is complete and valid.")
                return
            print(f"{tar_path.name} is incomplete or corrupt. Deleting and downloading from the beginning...")
            tar_path.unlink()
            if part_path.exists():
                part_path.unlink()

        print(f"Downloading {tar_path.name} (cycle {cycle}/{max_cycles})...")
        download_file(url, tar_path, description)

    raise RuntimeError(f"Could not get a valid {tar_path.name} after {max_cycles} cycles")


def extract_tar(tar_path, output_dir, description):
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=description):
            try:
                tar.extract(member, path=output_dir, filter="data")  # Python 3.12+
            except TypeError:
                tar.extract(member, path=output_dir)


def download_voc_dataset(output_dir="./data"):
    """
    Download and extract Pascal VOC 2012.
    The tar contains a VOCdevkit/VOC2012 folder, so output_dir should be the
    folder that will CONTAIN VOCdevkit (e.g. ./data).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    voc_root = output_dir / "VOCdevkit" / "VOC2012"

    # Train/val
    train_tar_path = output_dir / "VOCtrainval_11-May-2012.tar"
    if not (voc_root / "Annotations").exists():
        ensure_tar(VOC_TRAIN_URL, train_tar_path, "Downloading train/val")
        print("\nExtracting train/val dataset...")
        extract_tar(train_tar_path, output_dir, "Extracting train/val")
        print(f"Extracted to {voc_root}")
    else:
        print(f"VOC root already exists at {voc_root}")

    # Test (optional; the link is often dead)
    if not (voc_root / "ImageSets" / "Main" / "test.txt").exists():
        test_tar_path = output_dir / "VOC2012test.tar"
        try:
            ensure_tar(VOC_TEST_URL, test_tar_path, "Downloading test")
            print("\nExtracting test dataset...")
            extract_tar(test_tar_path, output_dir, "Extracting test")
        except Exception as e:
            print(f"Skipping test set: {e}")

    return voc_root


# Dataset validation / path helpers
def validate_voc_dataset(voc_root_path):
    """Validate that the VOC dataset is properly structured"""
    for dir_name in ['Annotations', 'ImageSets', 'JPEGImages']:
        dir_path = Path(voc_root_path) / dir_name
        if not dir_path.exists():
            raise ValueError(f"Required directory {dir_path} not found in VOC dataset")

    main_dir = Path(voc_root_path) / 'ImageSets' / 'Main'
    for split_file in ['train.txt', 'val.txt', 'test.txt']:
        if not (main_dir / split_file).exists():
            print(f"Warning: {split_file} not found in {main_dir}")

    print(f"VOC dataset validated at {voc_root_path}")
    return True


def find_voc_root(voc_root_path):
    """Find the actual VOC2012 directory from the given path"""
    path = Path(voc_root_path)

    if path.name == "VOC2012" and (path / "Annotations").exists():
        return path

    if path.name == "VOCdevkit":
        voc2012_path = path / "VOC2012"
        return voc2012_path if voc2012_path.exists() else path

    voc2012_path = path / "VOCdevkit" / "VOC2012"
    if voc2012_path.exists():
        return voc2012_path

    vocdevkit_path = path / "VOCdevkit"
    if vocdevkit_path.exists():
        return vocdevkit_path

    return path


def get_download_dir(original_path):
    """Folder that should CONTAIN VOCdevkit (the tar extracts VOCdevkit/VOC2012 inside it)"""
    original_path = Path(original_path)
    if original_path.name == "VOC2012" and original_path.parent.name == "VOCdevkit":
        return original_path.parent.parent      # ./data/VOCdevkit/VOC2012 -> ./data
    if original_path.name == "VOCdevkit":
        return original_path.parent             # ./data/VOCdevkit -> ./data
    return original_path                        # ./data -> ./data


def safe_delete_directory(path):
    """Safely delete a directory with confirmation"""
    path = Path(path)
    if not path.exists():
        return True

    dangerous_paths = ['/', '/home', '/Users', 'C:\\', 'C:/', '.', '']
    if str(path) in dangerous_paths or str(path.resolve()) in dangerous_paths:
        print(f"ERROR: Refusing to delete dangerous path: {path}")
        return False
    if len(str(path)) < 5:
        print(f"ERROR: Refusing to delete path with short name: {path}")
        return False

    response = input(f"WARNING: About to delete {path}. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Deletion cancelled")
        return False

    try:
        shutil.rmtree(path)
        print(f"Deleted: {path}")
        return True
    except Exception as e:
        print(f"Error deleting {path}: {e}")
        return False


# Main pipeline
def run(voc_root_path, output_dir, download=True, validate=True):
    """
    Prepare Pascal VOC dataset for YOLO training

    Args:
        voc_root_path: Path to VOCdevkit/VOC2012 folder or root folder
        output_dir: Where to save CSV files
        download: Whether to download dataset if not found
        validate: Whether to validate dataset structure
    """
    original_path = Path(voc_root_path)
    voc_root = find_voc_root(voc_root_path)

    print(f"Looking for VOC dataset at: {voc_root}")

    if not (voc_root / "Annotations").exists() and download:
        print(f"VOC dataset not found at {voc_root}. Downloading...")
        voc_root = download_voc_dataset(get_download_dir(original_path))
        voc_root_path = str(voc_root)

    if not (voc_root / "Annotations").exists():
        if (voc_root / "VOC2012" / "Annotations").exists():
            voc_root = voc_root / "VOC2012"
            voc_root_path = str(voc_root)
        else:
            print(f"ERROR: Could not find VOC dataset at {voc_root_path}")
            print("Please ensure the dataset is properly structured or remove --no_download")
            return

    if validate:
        try:
            validate_voc_dataset(voc_root_path)
        except ValueError as e:
            print(f"Error: {e}")
            if download and 'VOCdevkit' in str(voc_root_path):
                print("Attempting to fix dataset...")
                if safe_delete_directory(Path(voc_root_path)):
                    voc_root = download_voc_dataset(get_download_dir(original_path))
                    voc_root_path = str(voc_root)
                else:
                    raise
            else:
                raise

    os.makedirs(output_dir, exist_ok=True)

    splits = {'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt'}
    image_sets_dir = Path(voc_root_path) / 'ImageSets' / 'Main'

    if not image_sets_dir.exists():
        print(f"ERROR: ImageSets directory not found at {image_sets_dir}")
        return

    for split_name, split_file in splits.items():
        split_file_path = image_sets_dir / split_file

        if not split_file_path.exists():
            print(f"Warning: {split_file_path} not found. Skipping {split_name} split.")
            continue

        with open(split_file_path) as f:
            image_names = [line.strip() for line in f.readlines() if line.strip()]

        if not image_names:
            print(f"Warning: No images found in {split_file}. Skipping {split_name} split.")
            continue

        entries = [[f"{name}.jpg", f"{name}.txt"] for name in image_names]

        csv_path = Path(output_dir) / f"{split_name}.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            csv.writer(csvfile).writerows(entries)

        print(f"Created {csv_path} with {len(entries)} entries")

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

    if args.download_only:
        print("Downloading VOC dataset only...")
        target = get_download_dir(Path(args.voc_root))
        voc_root = download_voc_dataset(target)
        print(f"Dataset downloaded to {voc_root}")
        raise SystemExit(0)

    run(voc_root_path=args.voc_root,
        output_dir=args.output_dir,
        download=not args.no_download,
        validate=not args.no_validate)