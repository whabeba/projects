import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import svm 

dir = 'C:\\Users\\mohammed\\Desktop\\python course\\dogs-vs-cats\\train'
categories = ['cats', 'dogs']
data = []
for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        pet_img = cv2.imread(imgpath, 0)  # Read image in grayscale
        try:
            pet_img = cv2.resize(pet_img, (50, 50))  # Resize to 50x50
            image = np.array(pet_img).flatten()
            data.append([image, label])  # Add flattened image and label
        except Exception as e:
            pass

print("Total data samples:", len(data))

pick_out = open('data1.pickle', 'wb')
pickle.dump(data, pick_out)
pick_out.close()

pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)

features = []
labels = []
for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.01)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

svc = svm.SVC(probability=True)


model = GridSearchCV(svc, param_grid)
model.fit(xtrain, ytrain)

with open('model.sav', 'wb') as pick_out:
    pickle.dump(model, pick_out)

pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()
predictions = model.predict(xtest)

accuracy = model.score(xtest, ytest)
categories = ["cat", "dog"]
print("Accuracy:", accuracy)
print("Prediction:", categories[predictions[0]])

mypet = xtest[0].reshape(50, 50)
plt.imshow(mypet, cmap='gray')
plt.show()
