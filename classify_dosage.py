import pandas as pd

dataFrame = pd.read_csv("data/warfarin.csv")

classified_dose = []
for dose in dataFrame["Therapeutic Dose of Warfarin"]:
    if dose < 21:
        classified_dose.append(0)
    elif dose > 49:
        classified_dose.append(2)
    else:
        classified_dose.append(1)
dataFrame["Classified Dose of Warfarin"] = classified_dose

dataFrame.to_csv("data/warfarin_discrete.csv")
