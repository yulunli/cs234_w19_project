'''
linUCB algorithm based on http://rob.schapire.net/papers/www10.pdf
'''
import pandas as pd
import numpy as np
import util

K = 3

df = pd.read_csv("data/warfarin_imputed.csv")
labels = df["Classified Dose of Warfarin"]
# 3 features
rs9923231 = df["VKORC1 -1639 consensus"]
rs8050894 = df["VKORC1 1542 consensus"]
cyp2c9 = df["CYP2C9 consensus"]
# features: [5700, 16], last index is bias
d = 16
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
    elif row == "C/G":
        features[i] += [0,1,0,0]
    elif row == "G/G":
        features[i] += [0,0,1,0]
    else:
        features[i] += [0,0,0,1]

for i in range(len(cyp2c9)):
    row = cyp2c9[i]
    if row == "*1/*1":
        features[i] += [1,0,0,0,0,0,0]
    elif row == "*1/*2":
        features[i] += [0,1,0,0,0,0,0]
    elif row == "*1/*3":
        features[i] += [0,0,1,0,0,0,0]
    elif row == "*2/*2":
        features[i] += [0,0,0,1,0,0,0]
    elif row == "*2/*3":
        features[i] += [0,0,0,0,1,0,0]
    elif row == "*3/*3":
        features[i] += [0,0,0,0,0,1,0]
    else:
        features[i] += [0,0,0,0,0,0,1]

for feature in features:
    feature.append(1)
features = np.array(features)

data = np.array(list(zip(features, labels)))
# split into train and test
np.random.shuffle(data)
train, test = data[:5000, :], data[5000:, :]
alpha = 0.7 # hyperparameter

def main():
    regrets = []
    fractions = []
    for _ in range(10): # train 10 times in different order to get avg and error bars
        np.random.shuffle(train)
        regret = []
        fraction = []
        As = [np.identity(d) for _ in range(K)]
        bs = [np.zeros(d) for _ in range(K)]
        regret.append(0)
        fraction.append(evaluate(As, bs)) # performance without training
        total = 0
        correct = 0
        for f, y in train:
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
            # evaluate every 500 examples
            if total % 250 == 0:
                regret.append(total - correct)
                fraction.append(evaluate(As, bs))
        print(fraction)
        regrets.append(regret)
        fractions.append(fraction)
        print("Correct:", correct)
        print("Total:", total)
        print("Performance:", 1.0 * correct / total)

    # plot
    util.plot(range(0, 5001, 250), regrets, fractions)

# evaluation of fraction of incorrect dosing decisions with fixed weights
def evaluate(As, bs):
    invAs = [np.linalg.inv(As[i]) for i in range(K)]
    thetas = [np.matmul(invAs[i], bs[i]) for i in range(K)]
    incorrect = 0
    for f, y in test:
        ps = [np.dot(thetas[i], f) + alpha * np.sqrt(np.dot(np.squeeze(np.matmul(f[None, :], invAs[i])), f)) for i in range(K)]
        # a: chosen arm
        a = np.argmax(ps)
        if a != y:
            incorrect += 1
    return 1.0 * incorrect / len(test)

if __name__ == '__main__':
    main()
