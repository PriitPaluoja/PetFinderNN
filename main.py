import os

import imageio
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import keras.utils
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

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


# "Age"
# "Fee"
selected_columns = ["Type",
                    "Breed1",
                    "Breed2",
                    "Gender",
                    "Color1",
                    "Color2",
                    "Color3",
                    "MaturitySize",
                    "FurLength",
                    "Vaccinated",
                    "Dewormed",
                    "Sterilized",
                    "Health",
                    "State",
                    "MaturitySize"]

# "VideoAmt"
# "PhotoAmt"
# "Quantity"

label_column = "AdoptionSpeed"

y = train[label_column]

X = pd.get_dummies(train[selected_columns], columns=selected_columns)

# Normalize age:
to_normalize = ["Age", "Fee", "Quantity"]
for to_norm in to_normalize:
    X[to_norm] = (train[to_norm] - train[to_norm].mean()) / train[to_norm].std()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

inputs = Input(shape=(len(list(X_train)),))
x = Dense(150, activation='relu')(inputs)
x = Dense(250, activation='relu')(x)
x = Dense(350, activation='relu')(x)
x = Dense(450, activation='relu')(x)
x = Dense(700, activation='relu')(x)
x = Dense(500)(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, keras.utils.to_categorical(y_train),
          batch_size=1200,
          epochs=15,
          validation_split=0.04,
          callbacks=[EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=5,
                                   verbose=0, mode='auto')])

test_predictions = model.predict(X_test)
test_predictions = [np.argmax(pred) for pred in test_predictions]
print(test_predictions)

train_pred = [np.argmax(pred) for pred in model.predict(X_train)]

print("Kappa on train: {}".format(round(cohen_kappa_score(y_train, train_pred, weights="quadratic"), 4)))
print("Accuracy on train: {}".format(round(accuracy_score(y_train, train_pred), 4)))
print("________________")
print("Kappa on test: {}".format(round(cohen_kappa_score(y_test, test_predictions, weights="quadratic"), 4)))
print("Accuracy on test: {}".format(round(accuracy_score(y_test, test_predictions), 4)))
