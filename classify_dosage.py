import pandas as pd

df = pd.read_csv("data/warfarin.csv")

classified_dose = []
rows_to_drop = [] # drop rows without known optimal dose
doses = df["Therapeutic Dose of Warfarin"]
for i, dose in enumerate(doses):
    if pd.isnull(dose):
        rows_to_drop.append(i)
    if dose < 21:
        classified_dose.append(0)
    elif dose > 49:
        classified_dose.append(2)
    else:
        classified_dose.append(1)
df["Classified Dose of Warfarin"] = classified_dose
df.drop(df.index[rows_to_drop], inplace=True)
df.to_csv("data/warfarin_discrete.csv", index=False)
