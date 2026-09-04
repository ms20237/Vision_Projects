import os
import csv
import argparse


def init():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="YOLO V2 generate data Script")
    
    parser.add_argument('--train_path', 
                        type=str, 
                        default="train.txt",
                        help="path to train text file with image paths")
    
    parser.add_argument('--test_path', 
                        type=str, 
                        default="test.txt",
                        help="path to test text file with image paths")
    
    parser.add_argument('--train_output', 
                        type=str, 
                        default="train.csv",
                        help="output path for train CSV file")
    
    parser.add_argument('--test_output', 
                        type=str, 
                        default="test.csv",
                        help="output path for test CSV file")
    
    return parser.parse_args()


def run(train_path: str,
        test_path: str,
        train_output: str,
        test_output: str):
    """
    Generate CSV files from text files containing image paths
    
    Args:
        train_path: Path to train.txt file
        test_path: Path to test.txt file
        train_output: Output path for train.csv
        test_output: Output path for test.csv
    """
    
    # Process train data
    try:
        with open(train_path, "r") as f:
            read_train = f.readlines()
    except FileNotFoundError:
        print(f"Error: {train_path} not found")
        return
    
    with open(train_output, mode="w", newline="") as train_file:
        writer = csv.writer(train_file)  # Move writer outside the loop
        
        for line in read_train:
            image_file = line.strip().split("/")[-1]  # Use strip() to remove whitespace
            text_file = image_file.replace(".jpg", ".txt")
            data = [image_file, text_file]
            writer.writerow(data)
    
    print(f"Created {train_output} with {len(read_train)} entries")
    
    # Process test data
    try:
        with open(test_path, "r") as f:
            read_test = f.readlines()
    except FileNotFoundError:
        print(f"Error: {test_path} not found")
        return
    
    with open(test_output, mode="w", newline="") as test_file:
        writer = csv.writer(test_file)  # Move writer outside the loop
        
        for line in read_test:
            image_file = line.strip().split("/")[-1]  # Use strip() to remove whitespace
            text_file = image_file.replace(".jpg", ".txt")
            data = [image_file, text_file]
            writer.writerow(data)
    
    print(f"Created {test_output} with {len(read_test)} entries")


if __name__ == "__main__":
    args = init()
    run(args.train_path, 
        args.test_path, 
        args.train_output, 
        args.test_output)
    

    