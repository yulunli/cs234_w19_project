'''
linUCB algorithm based on http://rob.schapire.net/papers/www10.pdf
'''
import pandas as pd
import numpy as np

K = 3

df = pd.read_csv("data/warfarin_imputed.csv")
labels = df["Classified Dose of Warfarin"]
# using 1 feature for now
rs9923231 = df["VKORC1 -1639 consensus"]
rs8050894 = df["VKORC1 1542 consensus"]
# features: [5700, 4]
d = 8
features = []
for row in rs9923231:
    if row == "A/A":
        features.append([1,0,0,0])
    elif row == "A/G":
        features.append([0,1,0,0])
    elif row == "G/G":
        features.append([0,0,1,0])
    else:
        features.append([0,0,0,1])

for i in range(len(rs8050894)):
    row = rs8050894[i]
    if row == "C/C":
        features[i] += [1,0,0,0]
    elif row == "A/G":
        features[i] += [0,1,0,0]
    elif row == "G/G":
        features[i] += [0,0,1,0]
    else:
        features[i] += [0,0,0,1]
features = np.array(features)

As = [np.identity(d) for _ in range(K)]
bs = [np.zeros(d) for _ in range(K)]
alpha = 0.7 # hyperparameter
total = 0
correct = 0
for f, y in zip(features, labels):
    total += 1
    invAs = [np.linalg.inv(As[i]) for i in range(K)]
    thetas = [np.matmul(invAs[i], bs[i]) for i in range(K)]
    ps = [np.dot(thetas[i], f) + alpha * np.sqrt(np.dot(np.squeeze(np.matmul(f[None, :], invAs[i])), f)) for i in range(K)]
    # a: chosen arm
    a = np.argmax(ps)
    if a == y:
        r = 0
        correct += 1
    else:
        r = -1
    As[a] += np.matmul(f[:, None], f[None, :])
    bs[a] += r * f

print("Correct:", correct)
print("Total:", total)
print("Performance:", 1.0 * correct / total)

# evaluation
total = 0
correct = 0
for f, y in zip(features, labels):
    total += 1
    invAs = [np.linalg.inv(As[i]) for i in range(K)]
    thetas = [np.matmul(invAs[i], bs[i]) for i in range(K)]
    ps = [np.dot(thetas[i], f) + alpha * np.sqrt(np.dot(np.squeeze(np.matmul(f[None, :], invAs[i])), f)) for i in range(K)]
    # a: chosen arm
    a = np.argmax(ps)
    if a == y:
        correct += 1

print("Correct:", correct)
print("Total:", total)
print("Performance:", 1.0 * correct / total)
