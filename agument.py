import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# Augmentation Pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.GaussNoise(p=0.2)
])

# Function to Resize and Pad Image (Maintains Aspect Ratio)
def resize_and_pad(image, target_size=224):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a blank square image with padding
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    padded_img = np.full((target_size, target_size, 3), 128, dtype=np.uint8)  # Gray background
    padded_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return padded_img

input_folder = r"D:\RESUME\Deep_learning\agument"
output_folder = r"D:\RESUME\Deep_learning\agumented_dataset"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each person's folder
for person in os.listdir(input_folder):
    person_path = os.path.join(input_folder, person)
    output_person_path = os.path.join(output_folder, person)
    
    if not os.path.exists(output_person_path):
        os.makedirs(output_person_path)
    
    for img_name in tqdm(os.listdir(person_path)):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue
        
        # Resize & Pad Image
        image = resize_and_pad(image)

        # Create multiple augmented versions
        for i in range(13):  
            augmented = transform(image=image)["image"]
            aug_filename = f"{person}_{img_name.split('.')[0]}_aug{i}.jpg"
            aug_path = os.path.join(output_person_path, aug_filename)
            cv2.imwrite(aug_path, augmented)

print("Augmentation & Resizing Complete!")
