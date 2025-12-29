import os
import numpy as np
from PIL import Image

IMGS = 32
LR = 0.1
EPOCHS = 500
DATASET = "dataset"

x = []
y = []

labels = sorted(os.listdir(DATASET))
labelid = {l: i for i, l in enumerate(labels)}
idlabel = {i:l for l, i in labelid.items()}


def load_images():
    for label in labels:
        folder = os.path.join(DATASET, label)
        for f in os.listdir(folder):
            fl = os.path.join(folder, f)
            if os.path.isfile(fl):    
                img = Image.open(fl).convert("L")
                img = img.resize((IMGS, IMGS))
                arr = np.array(img).flatten() / 255
                x.append(arr)
                y.append(labelid[label])

load_images()
x = np.array(x)
y = np.array(y)

print("classes:", labels)


nc = len(labels)
y_onehot = np.zeros((len(y), nc))
y_onehot[np.arange(len(y)), y] = 1

def softmax(z):
    e = np.exp(z - np.max(z, axis = 1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

np.random.seed(0)
w = np.random.randn(IMGS * IMGS, nc) * 0.01
b = np.zeros((1, nc))


for epoch in range(EPOCHS):
    z = np.dot(x, w) + b
    y_hat = softmax(z)

    dz = y_hat - y_onehot
    dw = np.dot(x.T, dz) / len(x)
    db = np.mean(dz, axis=0, keepdims=True)

    w -= LR * dw
    b -= LR * db
    print(w)
while True:
    j = input("Enter a name: ")
    img = Image.open(j).convert("L")
    img = img.resize((IMGS, IMGS))
    x = np.array(img).flatten() / 255

    scores = np.dot(x, w) + b

    probs = softmax(scores.reshape(1, -1))

    predid = np.argmax(probs)
    predchar = idlabel[predid]

    print(probs[0][predid] * 100 , "%")
    print(predchar)
