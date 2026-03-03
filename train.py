import pandas as pd
import numpy as np
from feature import extract_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

print("Loading dataset...")
df = pd.read_pickle("LSWMD_fixed.pkl")

# -------- clean label --------
def clean_label(x):
    while isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return "none"
        x = x[0]
    return str(x)

df['label'] = df['failureType'].apply(clean_label)

# เอาเฉพาะ defect
df = df[df['label'] != 'none']
df = df.reset_index(drop=True)

# -------- feature extraction --------
print("Extracting features...")
X = []
y = []

for _, row in df.iterrows():
    X.append(extract_features(row['waferMap']))
    y.append(row['label'])

# -------- train test split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------- model --------
print("Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------- evaluate --------
print("\n=== RESULT ===")
pred = model.predict(X_test)

print(classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\naccuracy:",accuracy_score(y_test,pred))