import os
import numpy as np
from PIL import Image

IMGS = 32
LR = 0.1
EPOCHS = 100
DATASET = "dataset"
BATCH = 128

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

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

np.random.seed(0)

h1 = 256
h2 = 128

w1 = np.random.randn(x.shape[1], 256) * 0.01
b1 = np.zeros((1, h1))

w2 = np.random.randn(h1, h2) * 0.01
b2 = np.zeros((1, h2))

w3 = np.random.randn(h2, nc) * 0.01
b3 = np.zeros((1, nc))
n = len(x)
for epoch in range(EPOCHS):
    perm = np.random.permutation(n)
    x_shuf = x[perm]
    y_shuf = y_onehot[perm]

    epoch_loss = 0
    for i in range(0, n, BATCH):
        xb = x_shuf[i:i+BATCH]
        yb = y_shuf[i:i+BATCH]


        z1 = np.dot(xb, w1) + b1
        a1 = relu(z1)

        z2 = np.dot(a1, w2) + b2
        a2 = relu(z2)

        z3 = np.dot(a2, w3) + b3
        y_hat = softmax(z3)

        dz3 = y_hat - yb
        dw3 = np.dot(a2.T, dz3) / len(xb)
        db3 = np.mean(dz3, axis=0, keepdims=True)
    
        da2 = np.dot(dz3, w3.T)
        dz2 = da2 * relu_derivative(z2)
        dw2 = np.dot(a1.T , dz2) / len(xb)
        db2 = np.mean(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, w2.T)
        dz1 = da1 * relu_derivative(z1)
        dw1 = np.dot(xb.T, dz1) / len(xb)
        db1 = np.mean(dz1, axis=0, keepdims=True)

        w3 -= LR * dw3
        b3 -= LR * db3
        w2 -= LR * dw2
        b2 -= LR * db2
        w1 -= LR * dw1
        b1 -= LR * db1
    print(w1)

while True:
    j = input("Enter image path: ")

    img = Image.open(j).convert("L")
    img = img.resize((IMGS, IMGS))
    x_test = np.array(img).flatten() / 255.0

    z1 = x_test @ w1 + b1
    a1 = relu(z1)

    z2 = a1 @ w2 + b2
    a2 = relu(z2)

    z3 = a2 @ w3 + b3
    probs = softmax(z3.reshape(1, -1))

    predid = np.argmax(probs, axis=1)[0]   # 🔥 THIS LINE FIXES EVERYTHING
    predchar = idlabel[predid]

    print(f"Prediction: {predchar}")
    print(f"Confidence: {probs[0, predid] * 100:.2f}%")
