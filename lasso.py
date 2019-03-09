'''
Lasso Bandit based on http://web.stanford.edu/~bayati/papers/LassoBandit.pdf
'''
import pandas as pd
import numpy as np
from sklearn import linear_model

np.random.seed(seed=0)

K = 3

df = pd.read_csv("data/warfarin_imputed.csv")
labels = df["Classified Dose of Warfarin"]
# using 1 feature for now
rs9923231 = df["VKORC1 -1639 consensus"]
# features: [5700, 4]
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
features = np.array(features)

# number of arms
K = 3
# forced sampling parameter
q = 1
# localization parameter
h_0 = 10
h = h_0
lambda_1 = 0.1
lambda_2_0 = 0.1
lambda_2 = lambda_2_0
T = df.shape[0]
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
pi_hist = []
fit_intercept = True
for t in range(1, T + 1):
    # print('-------------------------------')
    # print(t)
    X_t = features[t - 1]
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
    # update s
    if t in Tau:
        Tau_i[pi_t].append(X_t)
        Y_Tau[pi_t].append(labels[t - 1])
    S_i[pi_t].append(X_t)
    Y_S[pi_t].append(labels[t - 1])
    lambda_2 = lambda_2_0 * np.sqrt((np.log(t) + np.log(d)) / t)
    # play pi_t
    if pi_t == labels[t - 1]:
        regret.append(0)
    else:
        regret.append(-1)


# evaluation
total = 0
correct = 0
for a_hat, a in zip(pi_hist, labels):
    total += 1
    if a_hat == a:
        correct += 1
    print(correct / total)

print("Correct:", correct)
print("Total:", total)
print("Performance:", 1.0 * correct / total)
