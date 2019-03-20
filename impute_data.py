'''
Impute missing rs9923231 based on S4 of appx.pdf
'''
import pandas as pd

df = pd.read_csv("data/warfarin_discrete.csv")

rs9923231 = df["VKORC1 -1639 consensus"].values.tolist()
race = df["Race"]
rs2359612 = df["VKORC1 2255 consensus"]
rs9934438 = df["VKORC1 1173 consensus"]
rs8050894 = df["VKORC1 1542 consensus"]

initial_empty, final_empty = 0, 0
for i in range(len(rs9923231)):
    if pd.isnull(rs9923231[i]):
        initial_empty += 1
        if race[i] != "Black or African American" and race[i] != "Missing or Mixed Race" and rs2359612[i] == "C/C":
            rs9923231[i] = "G/G"
        elif race[i] != "Black or African American" and race[i] != "Missing or Mixed Race" and rs2359612[i] == "T/T":
            rs9923231[i] = "A/A"
        elif race[i] != "Black or African American" and race[i] != "Missing or Mixed Race" and rs2359612[i] == "C/T":
            rs9923231[i] = "A/G"
        elif rs9934438[i] == "C/C":
            rs9923231[i] = "G/G"
        elif rs9934438[i] == "T/T":
            rs9923231[i] = "A/A"
        elif rs9934438[i] == "C/T":
            rs9923231[i] = "A/G"
        elif race[i] != "Black or African American" and race[i] != "Missing or Mixed Race" and rs8050894[i] == "G/G":
            rs9923231[i] = "G/G"
        elif race[i] != "Black or African American" and race[i] != "Missing or Mixed Race" and rs8050894[i] == "C/C":
            rs9923231[i] = "A/A"
        elif race[i] != "Black or African American" and race[i] != "Missing or Mixed Race" and rs8050894[i] == "C/G":
            rs9923231[i] = "A/G"
        else:
            final_empty += 1

df["VKORC1 -1639 consensus"] = rs9923231
df.to_csv("data/warfarin_imputed.csv")
print("Initial empty:", initial_empty)
print("Final empty:", final_empty)
