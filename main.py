import os

import imageio
import pandas as pd
from tqdm import tqdm

data_dir = "data"

# Read train.csv
# TODO: rm n-rows
train = pd.read_csv(os.path.join(data_dir, "train.csv"), sep=',', nrows=10)
pet_ids = train["PetID"]

# Read 1 image per pet
# TODO: there are more than 1 image per pet in the folder.
image_paths = [os.path.join(data_dir, "train_images", pet_id + "-1.jpg") for pet_id in pet_ids]
images = [imageio.imread(path) if os.path.isfile(path) else None for path in tqdm(image_paths)]
train = train.assign(image_1=images)



#print(len(images))
# print(list(train))


# im = imageio.imread(os.path.join(data_dir, "train_images", pet_id + "-1.jpg"))

#print(images)
