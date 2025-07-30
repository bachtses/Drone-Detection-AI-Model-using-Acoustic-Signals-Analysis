import os
import random
import shutil

# ---------------------- PARAMETERS ----------------------
DATASET_PATH = "dataset"
NUM_FILES_TO_MOVE = 200  # Change this number as needed

# Paths to train and test folders
TRAIN_DRONE_PATH = os.path.join(DATASET_PATH, "train", "drone")
TEST_DRONE_PATH = os.path.join(DATASET_PATH, "test", "drone")
TRAIN_BACKGROUND_PATH = os.path.join(DATASET_PATH, "train", "background")
TEST_BACKGROUND_PATH = os.path.join(DATASET_PATH, "test", "background")

# Ensure test folders exist
os.makedirs(TEST_DRONE_PATH, exist_ok=True)
os.makedirs(TEST_BACKGROUND_PATH, exist_ok=True)

# ---------------------- FUNCTION TO MOVE FILES ----------------------
def move_random_files(source_folder, target_folder, num_files):
    """Moves a random selection of .wav files from source_folder to target_folder."""
    files = [f for f in os.listdir(source_folder) if f.endswith(".wav")]
    
    if len(files) < num_files:
        print(f"⚠ Warning: {source_folder} only has {len(files)} files, moving all available.")
        num_files = len(files)  # Adjust to available files

    selected_files = random.sample(files, num_files)

    for file in selected_files:
        src_path = os.path.join(source_folder, file)
        dest_path = os.path.join(target_folder, file)
        shutil.move(src_path, dest_path)

    print(f"✔ Moved {num_files} files from {source_folder} to {target_folder}")

# ---------------------- MOVE FILES ----------------------
move_random_files(TRAIN_DRONE_PATH, TEST_DRONE_PATH, NUM_FILES_TO_MOVE)
move_random_files(TRAIN_BACKGROUND_PATH, TEST_BACKGROUND_PATH, NUM_FILES_TO_MOVE)

print("\n✅ File transfer complete.")
