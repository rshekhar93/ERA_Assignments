import os
import shutil
import tarfile
from tqdm import tqdm

def create_val_img_folder(val_dir):
    """
    Move validation images to labeled subfolders
    """
    # Read the validation labels file
    val_dict = {}
    with open('ILSVRC2012_validation_ground_truth.txt', 'r') as f:
        for i, line in enumerate(f.readlines(), 1):
            val_dict[f'ILSVRC2012_val_{i:08d}.JPEG'] = line.strip()

    # Create class folders and move images
    for img, label in tqdm(val_dict.items()):
        label_folder = os.path.join(val_dir, label)
        os.makedirs(label_folder, exist_ok=True)
        src = os.path.join(val_dir, img)
        dst = os.path.join(label_folder, img)
        if os.path.exists(src):
            shutil.move(src, dst)

def extract_train_dataset(train_tar_path, train_dir):
    """
    Extract training images and organize them into class folders
    """
    with tarfile.open(train_tar_path) as tar:
        for member in tqdm(tar.getmembers()):
            if member.name.endswith('.tar'):
                class_name = member.name.split('.')[0]
                class_dir = os.path.join(train_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Extract the class tar file
                tar.extract(member, path=train_dir)
                class_tar_path = os.path.join(train_dir, member.name)
                
                # Extract images from the class tar
                with tarfile.open(class_tar_path) as class_tar:
                    class_tar.extractall(class_dir)
                
                # Remove the class tar file
                os.remove(class_tar_path)

def main():
    # Set your paths here
    data_root = "/path/to/imagenet"
    train_tar = "ILSVRC2012_img_train.tar"
    val_tar = "ILSVRC2012_img_val.tar"

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print("Extracting and organizing training data...")
    extract_train_dataset(train_tar, train_dir)

    print("Extracting validation data...")
    with tarfile.open(val_tar) as tar:
        tar.extractall(val_dir)

    print("Organizing validation data...")
    create_val_img_folder(val_dir)

if __name__ == "__main__":
    main() 