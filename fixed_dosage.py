'''
Evaluate the performance of fixed-dosage baseline.
'''

import pandas as pd

df = pd.read_csv("data/warfarin_discrete.csv")

correct = 0
total = 0
for dose in df["Classified Dose of Warfarin"]:
    total += 1
    if dose == 1:
        correct += 1

print("Correct:", correct)
print("Total:", total)
print("Performance:", 1.0 * correct / total)
