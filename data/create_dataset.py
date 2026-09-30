import os
import csv
import time
import argparse
import http.client
import urllib.request
import urllib.error
import tarfile
import shutil
import xml.etree.ElementTree as ET

from pathlib import Path
from tqdm import tqdm


VOC_TRAIN_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
VOC_TEST_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOC2012test.tar"

# Class order defines the class index written into the YOLO label files
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Progress bar shows percentage, size, elapsed time, TIME REMAINING and speed
BAR_FORMAT = ("{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
              "[elapsed {elapsed} | remaining {remaining} | {rate_fmt}]")


def init():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YOLO V1 Dataset Preparation Script", formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
            Examples:
            # Everything: download (if missing) -> convert XML to YOLO txt -> create CSVs
            python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv

            # Separate steps
            python create_dataset.py --voc_root ./data --download_only
            python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --convert_only
            python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv --csv_only

            # Other options
            python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv --no_download
            python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --output_dir ./data/csv --no_validate
            python create_dataset.py --voc_root ./data/VOCdevkit/VOC2012 --convert_only --label_dir ./data/labels --overwrite_labels
    """)
    parser.add_argument('--voc_root', 
                        type=str, 
                        required=True,
                        help='Path to VOC2012 folder (e.g., ./data/VOCdevkit/VOC2012) or root folder (e.g., ./data)')
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./data',
                        help='Output directory for CSV files (default: ./data)')

    parser.add_argument('--label_dir',
                        type=str,
                        default=None,
                        help='Output directory for YOLO .txt labels (default: <voc_root>/labels)')
    
    parser.add_argument('--no_download', 
                        action='store_true',
                        help='Skip downloading dataset if not found')
    
    parser.add_argument('--no_validate', 
                        action='store_true',
                        help='Skip validation of dataset structure')

    parser.add_argument('--no_convert',
                        action='store_true',
                        help='Skip XML -> YOLO txt label conversion in the full run')

    parser.add_argument('--overwrite_labels',
                        action='store_true',
                        help='Re-create label .txt files that already exist (default: skip existing)')

    # Run a single step on its own (only one of these can be used at a time)
    step = parser.add_mutually_exclusive_group()
    step.add_argument('--download_only', 
                      action='store_true',
                      help="Only download dataset, don't convert labels or create CSV files")
    step.add_argument('--convert_only',
                      action='store_true',
                      help="Only convert XML annotations to YOLO .txt labels (no CSV files)")
    step.add_argument('--csv_only',
                      action='store_true',
                      help="Only create the CSV files (no label conversion)")
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


def download_file(url, output_path, description="Downloading", max_retries=50):
    """
    Download a file with automatic resume.

    - Data is written to "<name>.part". If the download drops mid-way, or you
      press Ctrl+C, the .part file is kept.
    - Running again with the SAME output path continues from where it stopped.
    - Connection drops are retried automatically. The retry counter resets
      whenever new data arrives, so only repeated failures with NO progress
      count towards max_retries.
    - The final filename only appears when the download is fully complete.
    """
    output_path = Path(output_path)
    part_path = output_path.with_name(output_path.name + ".part")
    remote_size = get_remote_size(url)
    failures = 0

    while True:
        done = part_path.stat().st_size if part_path.exists() else 0
        start_size = done
        try:
            if remote_size and done > remote_size:      # bad partial file -> restart
                part_path.unlink()
                done = start_size = 0
            if remote_size and done == remote_size:     # already fully downloaded
                break

            if done:
                pct = f" ({done / remote_size:.0%})" if remote_size else ""
                print(f"Resuming from {done / 1024 / 1024:.1f} MB{pct}...")

            req = urllib.request.Request(url)
            if done:
                req.add_header("Range", f"bytes={done}-")

            with urllib.request.urlopen(req, timeout=30) as resp:
                if done and resp.status != 206:         # server ignored Range -> restart
                    print("Server does not support resume; restarting from the beginning.")
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
                raise IOError(f"Connection closed early ({size}/{total} bytes)")
            break  # success

        except KeyboardInterrupt:
            print("\nDownload paused. Run the same command again to resume from where it stopped.")
            raise

        except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException) as e:
            if isinstance(e, urllib.error.HTTPError):
                if e.code in (401, 403, 404):
                    raise RuntimeError(f"HTTP {e.code} for {url}: not retrying") from e
                if e.code == 416 and done:
                    break   # range beyond end of file: .part is already complete

            now = part_path.stat().st_size if part_path.exists() else 0
            failures = 0 if now > start_size else failures + 1   # progress resets the counter
            if failures >= max_retries:
                raise RuntimeError(f"Failed to download {url}: no progress after {max_retries} retries") from e
            print(f"\nDownload interrupted ({e}). Retrying in 5s and resuming "
                  f"from {now / 1024 / 1024:.1f} MB... (retry {failures}/{max_retries})")
            time.sleep(5)

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


# XML -> YOLO label conversion
def convert_xml_to_yolo(voc_root_path, label_dir=None, overwrite=False):
    """
    Convert Pascal VOC XML annotations into YOLO-format .txt files.

    Each output line is:  class_index x_center y_center width height
    with all four box values normalized to 0-1 (what dataset.py expects).

    Args:
        voc_root_path: Path to the VOC2012 folder (must contain Annotations/)
        label_dir: Where to write the .txt files (default: <voc_root>/labels)
        overwrite: Re-create files that already exist (default: skip them)

    Returns:
        Path to the label directory.
    """
    voc_root = Path(voc_root_path)
    ann_dir = voc_root / "Annotations"
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations folder not found at {ann_dir}")

    label_dir = Path(label_dir) if label_dir else voc_root / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ann_dir.glob("*.xml"))
    converted = skipped = failed = 0

    bar = tqdm(xml_files, desc="Converting XML -> YOLO txt", unit="file",
               bar_format=BAR_FORMAT, smoothing=0.05)
    for xml_path in bar:
        bar.set_postfix(converted=converted, skipped=skipped, failed=failed)
        out_path = label_dir / f"{xml_path.stem}.txt"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            root = ET.parse(xml_path).getroot()
            W = float(root.find("size/width").text)
            H = float(root.find("size/height").text)
            if W <= 0 or H <= 0:
                raise ValueError(f"invalid image size {W}x{H}")

            lines = []
            for obj in root.iter("object"):
                cls = obj.find("name").text.strip()
                if cls not in VOC_CLASSES:
                    continue
                bb = obj.find("bndbox")
                xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
                xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)

                x_center = (xmin + xmax) / 2 / W
                y_center = (ymin + ymax) / 2 / H
                width = (xmax - xmin) / W
                height = (ymax - ymin) / H
                lines.append(f"{VOC_CLASSES.index(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            with open(out_path, "w") as f:
                f.write("\n".join(lines))
            converted += 1

        except (ET.ParseError, AttributeError, ValueError) as e:
            failed += 1
            tqdm.write(f"Warning: could not convert {xml_path.name}: {e}")
    bar.set_postfix(converted=converted, skipped=skipped, failed=failed)
    bar.close()

    print(f"Labels: {converted} converted, {skipped} already existed, {failed} failed -> {label_dir}")
    return label_dir


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


def run(voc_root_path, output_dir, download=True, validate=True,
        convert=True, make_csv=True, label_dir=None, overwrite_labels=False):
    """
    Prepare Pascal VOC dataset for YOLO training

    Args:
        voc_root_path: Path to VOCdevkit/VOC2012 folder or root folder
        output_dir: Where to save CSV files
        download: Whether to download dataset if not found
        validate: Whether to validate dataset structure
        convert: Whether to convert XML annotations to YOLO .txt labels
        make_csv: Whether to create the CSV files
        label_dir: Where to write .txt labels (default: <voc_root>/labels)
        overwrite_labels: Re-create label files that already exist
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

    if convert:
        label_dir = convert_xml_to_yolo(voc_root_path, label_dir, overwrite=overwrite_labels)
        print(f"Set label_dir in your train config to: {label_dir}")

    if not make_csv:
        return

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

        entries = [[f"{name}.jpg", f"{name}.txt"]
                   for name in tqdm(image_names, desc=f"Building {split_name}.csv",
                                    unit="img", bar_format=BAR_FORMAT)]

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
        validate=not args.no_validate,
        convert=not args.no_convert and not args.csv_only,
        make_csv=not args.convert_only,
        label_dir=args.label_dir,
        overwrite_labels=args.overwrite_labels)