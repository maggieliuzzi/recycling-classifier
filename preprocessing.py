import argparse
import csv
import glob
import math
import os
from random import shuffle
import shutil



parser = argparse.ArgumentParser(
    description="Retrains top layers of MobileNetV2 for classifying images into various object materials for accurate recyling.",
    epilog="")
args = parser.parse_args()

# Data Augmentation: first manually duplicate the images to reach _ in total in each category + added images sourced by web scrapping, then apply rotations, etc

home_path = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(home_path, "trashnet-dataset-full/")
csv_path = os.path.join(source_path,'data.csv')

cardboard_data_list = glob.glob(os.path.join(source_path,'cardboard/*.jpg'))
glass_data_list = glob.glob(os.path.join(source_path,'glass/*.jpg'))
metal_data_list = glob.glob(os.path.join(source_path,'metal/*.jpg'))
paper_data_list = glob.glob(os.path.join(source_path,'paper/*.jpg'))
plastic_data_list = glob.glob(os.path.join(source_path,'plastic/*.jpg'))
trash_data_list = glob.glob(os.path.join(source_path,'trash/*.jpg'))

with open(csv_path, 'w') as wf:
    writer = csv.writer(wf, delimiter=',', quoting=csv.QUOTE_ALL)
    for item in cardboard_data_list:
        writer.writerow([item] + ["cardboard"])
    for item in glass_data_list:
        writer.writerow([item] + ["glass"])
    for item in metal_data_list:
        writer.writerow([item] + ["metal"])
    for item in paper_data_list:
        writer.writerow([item] + ["paper"])
    for item in plastic_data_list:
        writer.writerow([item] + ["plastic"])
    for item in trash_data_list:
        writer.writerow([item] + ["trash"])


dataset_path = os.path.join(home_path, "training-data-formatted/")
train_path = os.path.join(dataset_path, "train")
validate_path = os.path.join(dataset_path, "validate")

train_path_1 = os.path.join(train_path, "cardboard")
train_path_2 = os.path.join(train_path, "glass")
train_path_3 = os.path.join(train_path, "metal")
train_path_4 = os.path.join(train_path, "paper")
train_path_5 = os.path.join(train_path, "plastic")
train_path_6 = os.path.join(train_path, "trash")

validate_path_1 = os.path.join(validate_path, "cardboard")
validate_path_2 = os.path.join(validate_path, "glass")
validate_path_3 = os.path.join(validate_path, "metal")
validate_path_4 = os.path.join(validate_path, "paper")
validate_path_5 = os.path.join(validate_path, "plastic")
validate_path_6 = os.path.join(validate_path, "trash")

test_path = os.path.join(dataset_path, "test")

processed_paths = [train_path_1, train_path_2, train_path_3, train_path_4, train_path_5, train_path_6, 
                   validate_path_1, validate_path_2, validate_path_3, validate_path_4, validate_path_5, validate_path_6,
                   test_path]

for path in processed_paths:
    if not os.path.isdir(path):
        os.makedirs(path)


with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)

    total_images = 0
    usable = []
    # Assuming all training images are usable - Add filtering
    for row in reader:
        usable.append(row)
        total_images += 1
    print("Loaded data from " + csv_path)

    # Randomising data to ensure even distribution between train, validate and test folders
    usable = [i for i in usable]
    shuffle(usable)

    usable_images = len(usable)
    train_cutoff = 0.64
    validate_cutoff = 0.8
    train_images = int(math.floor(usable_images * train_cutoff))
    validate_images = int(math.floor(usable_images * (validate_cutoff - train_cutoff)))
    test_images = usable_images - (train_images + validate_images)

    train_images_data = []
    validate_images_data = []
    test_images_data = []
    current_point = 0

    with open(dataset_path + "form_data.csv", "w") as form_data_file, open(dataset_path + "test.csv", "w") as test_data_file:
        # TOFIX (maggie.liuzzi): eliminate redundancy
        # For all training images
        for i in range(0, train_images):
            train_images_data.append(usable[i])
            role = "train"
            material = usable[i][1]
            source = os.path.join(source_path, material+"-data", usable[i][0])
            destination = os.path.join(train_path, material)
            shutil.copy(source, destination)
            writer = csv.writer(form_data_file, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow([usable[i]] + [material] + [role])
            current_point += 1
        # For all validation images
        for i in range(train_images, train_images + validate_images):
            validate_images_data.append(usable[i])
            role = "validate"
            material = usable[i][1]
            source = os.path.join(source_path, material+"-data", usable[i][0])
            destination = os.path.join(validate_path, material)
            shutil.copy(source, destination)
            writer = csv.writer(form_data_file, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow([usable[i]] + [material] + [role])
            current_point += 1
        # For all testing images
        for i in range(train_images + validate_images, usable_images):
            test_images_data.append(usable[i])
            role = "test"
            material = usable[i][1]
            source = os.path.join(source_path, material+"-data", usable[i][0])
            destination = os.path.join(test_path)
            shutil.copy(source, destination)
            writer = csv.writer(form_data_file, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow([usable[i]] + [material] + [role])
            writer = csv.writer(test_data_file, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow([usable[i]] + [material] + [role])
            current_point += 1

    print("Total images: " + str(total_images))
    print("Usable images: " + str(usable_images))
    print("Training images: " + str(len(train_images_data)))
    print("Validation images: " + str(len(validate_images_data)))
    print("Testing images: " + str(len(test_images_data)))

print("Pre-processing done.")
