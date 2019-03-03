import numpy as np
import pandas as pd

df = pd.read_csv('data/warfarin.csv')
df = df.drop(df[df['Age'].isnull()].index)
# df = df.drop(df[df['Amiodarone (Cordarone)'].isnull()].index)
print(df.shape)

df['is_asian'] = df.Race == 'Asian'
df['is_afr_am'] = df.Race == 'Black or African American'
df['is_m'] = df.Race == 'Unknown'#  | df.Race.isnull()
df['has_inducer'] = (df['Carbamazepine (Tegretol)'] == 1) | (df['Phenytoin (Dilantin)'] == 1) | (df['Rifampin or Rifampicin'] == 1)
df['has_amio'] = df['Amiodarone (Cordarone)'] == 1
df.loc[df['Age'] == '10 - 19', 'age'] = 1
df.loc[df['Age'] == '20 - 29', 'age'] = 2
df.loc[df['Age'] == '30 - 39', 'age'] = 3
df.loc[df['Age'] == '40 - 49', 'age'] = 4
df.loc[df['Age'] == '50 - 59', 'age'] = 5
df.loc[df['Age'] == '60 - 69', 'age'] = 6
df.loc[df['Age'] == '70 - 79', 'age'] = 7
df.loc[df['Age'] == '80 - 89', 'age'] = 8
df.loc[df['Age'] == '90+', 'age'] = 9

coefficients = {
    'age': 0.2546,
    'Height (cm)': 0.0118,
    'Weight (kg)': 0.0134,
    'is_asian': 0.6752,
    'is_afr_am': 0.4060, 
    'is_m': 0.0443,
    'has_inducer': 1.2799,
    'has_amio': 0.5695,
}

y = np.zeros(df.shape[0])
for col in coefficients:
    d = np.multiply(df[col].values[0], coefficients[col])
    y += d

df['raw_predicted'] = y
df.loc[df['raw_predicted'] < 21, 'predicted'] = 0
df.loc[(df['raw_predicted'] > 21) & (df['raw_predicted'] < 49), 'predicted'] = 1
df.loc[df['raw_predicted'] > 49, 'predicted'] = 2

df.loc[df['Therapeutic Dose of Warfarin'] < 21, 'label'] = 0
df.loc[(df['Therapeutic Dose of Warfarin'] > 21) & (df['Therapeutic Dose of Warfarin'] < 49), 'label'] = 1
df.loc[df['Therapeutic Dose of Warfarin'] > 49, 'label'] = 2


res = (df['predicted'] != df['label']).values
print('predicted', np.sum(res))
print('total', df.shape[0])
print('rate', np.sum(res) / df.shape[0])
