'''
TS algorithm based on http://proceedings.mlr.press/v28/agrawal13.pdf
'''
import pandas as pd
import numpy as np
import util

np.random.seed(1)
K = 3

df = pd.read_csv("data/warfarin_imputed.csv")
labels = df["Classified Dose of Warfarin"]
# 3 features
rs9923231 = df["VKORC1 -1639 consensus"]
rs8050894 = df["VKORC1 1542 consensus"]
cyp2c9 = df["CYP2C9 consensus"]
# features: [5700, 15]
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
v_sq = 0.01 # hyperparameter

def main():
    regrets = []
    fractions = []
    for _ in range(10): # train 10 times in different order to get avg and error bars
        np.random.shuffle(train)
        regret = []
        fraction = []
        regret.append(0)
        Bs = [np.identity(d) for _ in range(K)]
        mu_hats = [np.zeros(d) for _ in range(K)]
        fs = [np.zeros(d) for _ in range(K)]
        fraction.append(evaluate(mu_hats, Bs)) # performance without training
        total = 0
        correct = 0
        for f, y in train:
            total += 1
            mu_ts = [np.random.multivariate_normal(mu_hats[i], v_sq * np.linalg.inv(Bs[i])) for i in range(K)]
            expects = [np.dot(f.T, mu_ts[i]) for i in range(K)]
            a = np.argmax(expects)
            if a == y:
                r = 1
                correct += 1
            else:
                r = 0
            Bs[a] += np.dot(f, f)
            fs[a] += f * r
            mu_hats[a] = np.dot(np.linalg.inv(Bs[a]), fs[a])
            # evaluate every 500 examples
            if total % 500 == 0:
                regret.append(total - correct)
                fraction.append(evaluate(mu_hats, Bs))
        print(fraction)
        regrets.append(regret)
        fractions.append(fraction)
        print("Correct:", correct)
        print("Total:", total)
        print("Performance:", 1.0 * correct / total)

    # plot
    util.plot(range(0, 5001, 500), regrets, fractions)

# evaluation of fraction of incorrect dosing decisions with fixed weights
def evaluate(mu_hats, Bs):
    incorrect = 0
    for f, y in test:
        mu_ts = [np.random.multivariate_normal(mu_hats[i], v_sq * np.linalg.inv(Bs[i])) for i in range(K)]
        expects = [f * mu_ts[i] for i in range(K)]
        # a: chosen arm
        a = np.argmax(expects)
        if a != y:
            incorrect += 1
    return 1.0 * incorrect / 700

if __name__ == '__main__':
    main()
