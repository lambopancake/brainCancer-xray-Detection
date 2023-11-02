import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import os, cv2


mainPath = os.getcwd() + "\\BrainTumorDect\\brainCancer-xray-Detection\\"
modelPath = mainPath + "MedicalDect.h5"
model = load_model(modelPath)
file = os.listdir(mainPath + "Image\\")

a = 3
img = cv2.imread(mainPath + "\\Image\\" + file[a])
imgAlt = cv2.resize(img, (150,150),interpolation = cv2.INTER_AREA) / 255
x = np.expand_dims(imgAlt, axis = 0)

labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

pred = model.predict(x)
print("\n************************")
print("File: " + file[a] + "\n")
plt.imshow(img)
plt.title("Predicted: " + labels[np.argmax(pred)])
plt.axis("off")
plt.show()

for prob in range(len(pred[0])):
    print(labels[prob] + ": " + str(round(pred[0][prob] * 100 , 3)))
print("************************\n")

