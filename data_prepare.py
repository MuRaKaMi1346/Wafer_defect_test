import pandas as pd
from feature import extract_features  

df = pd.read_pickle("LSWMD_fixed.pkl")

print(df.columns)
print(df['failureType'].head())

import numpy as np

def clean_label(x):

    while isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return "none"
        x = x[0]

    return str(x)

df['label'] = df['failureType'].apply(clean_label)

print(df['label'].value_counts())

df = df[df['label'] != 'none']
df = df.reset_index(drop=True)

for i in range(10):
    print(df['failureType'].iloc[i], type(df['failureType'].iloc[i]))