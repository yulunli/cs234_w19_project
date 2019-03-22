'''
Lasso Bandit based on http://web.stanford.edu/~bayati/papers/LassoBandit.pdf
'''
import pandas as pd
import numpy as np
from sklearn import linear_model
import util

np.random.seed(seed=0)

K = 3

df = pd.read_csv("data/warfarin_imputed.csv")
labels = df["Classified Dose of Warfarin"]
# 3 features
rs9923231 = df["VKORC1 -1639 consensus"]
rs8050894 = df["VKORC1 1542 consensus"]
cyp2c9 = df["CYP2C9 consensus"]
# features: [5700, 16], last index is bias
d = 4
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

# for feature in features:
#     feature.append(1)
features = np.array(features)

data = np.array(list(zip(features, labels)))
# split into train and test
np.random.shuffle(data)
train, test = data[:5000, :], data[5000:, :]

# number of arms
K = 3
# forced sampling parameter
q = 1
# localization parameter
h_0 = 10

def main():
    regrets = []
    fractions = []
    for _ in range(5): # train 10 times in different order to get avg and error bars
        np.random.shuffle(train)
        h = h_0
        lambda_1 = 0.1
        lambda_2_0 = 0.1
        lambda_2 = lambda_2_0
        T = train.shape[0]
        Tau = {}
        # forced sample set
        Tau_i = dict([(i, []) for i in range(1, K + 1)])
        # all sample set
        S_i = dict([(i, []) for i in range(1, K + 1)])

        Y_Tau = dict([(i, []) for i in range(1, K + 1)])
        Y_S = dict([(i, []) for i in range(1, K + 1)])
        for i in range(1, K + 1):
            for j in range(q * (i - 1), q * i):
                n = 0
                while True:
                    tau_i = (np.power(2, n) - 1) * K * q + j
                    Tau[tau_i] = i
                    n += 1
                    if tau_i >= T:
                        break

        regret = []
        fraction = []
        regret.append(0)
        fit_intercept = True
        fraction.append(1.)
        pi_hist = []
        total = 0
        correct = 0
        for f, y in train:
            t = total + 1
            # print('-------------------------------')
            # print(t)
            X_t = f
            h = h_0 / np.log(max(t, 2))
            pi_t = None
            if t in Tau:
                pi_t = Tau[t]
            else:
                # find subset
                model_1 = linear_model.Lasso(alpha=lambda_1, fit_intercept=fit_intercept)
                y_s = []
                for arm_idx in range(1, K + 1):
                    fitted = model_1.fit(Tau_i[arm_idx], Y_Tau[arm_idx])
                    y_hat = model_1.predict([X_t])[0]
                    y_s.append(y_hat)
                actions = []
                max_y = np.max(y_s)
                for arm_idx in range(1, K + 1):
                    if y_s[arm_idx - 1] >= max_y - h / 2:
                        actions.append(arm_idx)
                # find max in subset
                model_2 = linear_model.Lasso(alpha=lambda_2, fit_intercept=fit_intercept)
                y_ = -np.inf
                for a in actions:
                    fitted = model_2.fit(S_i[arm_idx], Y_S[arm_idx])
                    y_hat = model_2.predict([X_t])[0]
                    if y_hat > y_:
                        pi_t = a
                        y_ = y_hat
            pi_hist.append(pi_t)
            a = pi_t
            if a == y:
                r = 0
                correct += 1
            else:
                r = -1
            reward = -1
            if pi_t == y:
                reward = 0
            # update s
            if t in Tau:
                Tau_i[pi_t].append(X_t)
                Y_Tau[pi_t].append(reward)
            S_i[pi_t].append(X_t)
            Y_S[pi_t].append(reward)
            lambda_2 = lambda_2_0 * np.sqrt((np.log(t) + np.log(d)) / t)
            # play pi_t
            # if pi_t == y:
            #     regret.append(0)
            # else:
            #     regret.append(-1)
            total += 1
            # evaluate every 500 examples
            if total % 500 == 0:
                regret.append(total - correct)
                ev = evaluate(lambda_1, fit_intercept, Tau_i, Y_Tau, h, lambda_2, S_i, Y_S)
                fraction.append(ev)
        print(fraction)
        regrets.append(regret)
        fractions.append(fraction)
        print("Correct:", correct)
        print("Total:", total)
        print("Performance:", 1.0 * correct / total)

    print(np.array(regrets).shape, np.array(fractions).shape)
    util.plot(range(0, 5001, 500), regrets, fractions)

# evaluation of fraction of incorrect dosing decisions with fixed weights
def evaluate(lambda_1, fit_intercept, Tau_i, Y_Tau, h, lambda_2, S_i, Y_S):
    incorrect = 0
    for f, y in test:
        # find subset
        model_1 = linear_model.Lasso(alpha=lambda_1, fit_intercept=fit_intercept)
        y_s = []
        pi_t = None
        for arm_idx in range(1, K + 1):
            fitted = model_1.fit(Tau_i[arm_idx], Y_Tau[arm_idx])
            y_hat = model_1.predict([f])[0]
            y_s.append(y_hat)
        actions = []
        max_y = np.max(y_s)
        for arm_idx in range(1, K + 1):
            if y_s[arm_idx - 1] >= max_y - h / 2:
                actions.append(arm_idx)
        # find max in subset
        model_2 = linear_model.Lasso(alpha=lambda_2, fit_intercept=fit_intercept)
        y_ = -np.inf
        for a in actions:
            fitted = model_2.fit(S_i[arm_idx], Y_S[arm_idx])
            y_hat = model_2.predict([f])[0]
            if y_hat > y_:
                pi_t = a
                y_ = y_hat
        # a: chosen arm
        if pi_t != y:
            incorrect += 1
    return 1.0 * incorrect / test.shape[0]

if __name__ == '__main__':
    main()

