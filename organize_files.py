# src/organize_files.py

import pandas as pd
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_dataset(raw_data_path="CUB_200_2011", organized_data_path="data"):
    """
    Reads the raw CUB_200_2011 text files and organizes the images into
    a clean train/test directory structure.

    Args:
        raw_data_path (str): The path to the downloaded, unzipped CUB_200_2011 directory.
        organized_data_path (str): The path where the clean 'train' and 'test' folders will be created.
    """
    print("--- Starting Dataset Organization ---")
    
    # Ensure the raw data path exists
    if not os.path.isdir(raw_data_path):
        print(f"Error: Raw data directory not found at '{raw_data_path}'")
        print("Please make sure you have downloaded and unzipped CUB_200_2011.tgz in your project folder.")
        return

    # Create the target directories
    train_dir = Path(organized_data_path) / "train"
    test_dir = Path(organized_data_path) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the instruction files using pandas
    print("Loading instruction files...")
    images_df = pd.read_csv(Path(raw_data_path) / 'images.txt', sep=' ', names=['image_id', 'filepath'])
    split_df = pd.read_csv(Path(raw_data_path) / 'train_test_split.txt', sep=' ', names=['image_id', 'is_train'])
    labels_df = pd.read_csv(Path(raw_data_path) / 'image_class_labels.txt', sep=' ', names=['image_id', 'class_id'])
    
    # 2. Merge them into one master dataframe
    master_df = images_df.merge(split_df, on='image_id').merge(labels_df, on='image_id')
    print("Successfully merged metadata.")

    # 3. Loop through the master dataframe and copy files
    print(f"Organizing {len(master_df)} images into train/test folders...")
    
    # Using tqdm for a nice progress bar
    for index, row in tqdm(master_df.iterrows(), total=master_df.shape[0]):
        image_path = Path(raw_data_path) / 'images' / row['filepath']
        
        # The class folder name is part of the original filepath
        # e.g., '001.Black_footed_Albatross/...'
        class_folder_name = image_path.parent.name
        
        # Determine if it's a train or test image
        if row['is_train'] == 1:
            target_dir = train_dir / class_folder_name
        else:
            target_dir = test_dir / class_folder_name
            
        # Create the class directory if it doesn't exist
        target_dir.mkdir(exist_ok=True)
        
        # Define the destination path
        destination_path = target_dir / image_path.name
        
        # Copy the file
        shutil.copy(image_path, destination_path)
        
    print("--- Dataset Organization Complete! ---")
    print(f"Clean dataset is ready at: '{os.path.abspath(organized_data_path)}'")


if __name__ == "__main__":
    # Make sure you've run the wget/tar commands from a previous step
    # to have the 'CUB_200_2011' folder.
    # If not, you can run them in your terminal:
    # !wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
    # !tar -xzf CUB_200_2011.tgz
    organize_dataset()