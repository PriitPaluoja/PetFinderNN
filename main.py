import os

import imageio
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_dir = "data"

# Read train.csv
# TODO: rm n-rows
train = pd.read_csv(os.path.join(data_dir, "train.csv"), sep=',')
pet_ids = train["PetID"]

# Read 1 image per pet
# TODO: there are more than 1 image per pet in the folder.
# image_paths = [os.path.join(data_dir, "train_images", pet_id + "-1.jpg") for pet_id in pet_ids]
# images = [imageio.imread(path) if os.path.isfile(path) else None for path in tqdm(image_paths)]
# train = train.assign(image_1=images)


selected_columns = ["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                    "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health",
                    "Quantity", "Fee", "State", "VideoAmt", "PhotoAmt", "AdoptionSpeed"]

label_column = "AdoptionSpeed"

y = train["AdoptionSpeed"]
X = train[selected_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
rf = RandomForestClassifier().fit(X_train, y_train)
predictions = rf.predict(X_test)

print(cohen_kappa_score(y_test, predictions, weights="quadratic"))
