import os
import numpy as np
from PIL import Image

IMGS = 32
LR = 0.1
EPOCHS = 1000

X = []
Y = []

def load_images(folder, label):
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        img = Image.open(path).convert("L")
        img = img.resize((IMGS, IMGS))
        arr = np.array(img).flatten() / 255
        X.append(arr)
        Y.append(label)

load_images("A/", 1)
load_images("Not_A/", 0)


X = np.array(X)
Y = np.array(Y)

print("Training Samples:", len(X))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

np.random.seed(0)
w = np.random.randn(IMGS * IMGS) * 0.01
b = 0

for epoch in range(EPOCHS):
    z = np.dot(X,w) + b
    yy = sigmoid(z)

    dz = yy - Y
    dW = np.dot(X.T, dz) / len(X)
    db = np.mean(dz)
    
    w -= LR * dW
    b -= LR * db


img = Image.open("char.png").convert("L")
img = img.resize((IMGS, IMGS))
x = np.array(img).flatten() / 255

pr = sigmoid(np.dot(x, w) + b)
