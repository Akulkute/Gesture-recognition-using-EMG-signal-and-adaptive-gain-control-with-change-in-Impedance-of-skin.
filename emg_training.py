import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import warnings
import glob
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Only 3 gestures ───────────────────────────────────────────
gesture_map = {
    0: 'HandOpen',
    1: 'HandClose',
    2: 'Rest'
}
inv_map  = {v: k for k, v in gesture_map.items()}
GESTURE_COLS = ['HandOpen', 'HandClose', 'Rest']

# ── Load volunteer CSVs from emg_dataset ─────────────────────
dataset_path = "emg_dataset"
csv_files    = glob.glob(os.path.join(dataset_path, "*.csv"))
print(f"Found {len(csv_files)} files: {csv_files}")

all_dfs = []
for file in csv_files:
    df = pd.read_csv(file, index_col=0)
    df.columns = df.columns.str.strip()
    available  = [c for c in GESTURE_COLS if c in df.columns]
    df         = df[available].dropna()
    all_dfs.append(df)
    print(f"  Loaded {file}: {df.shape}")

df_combined = pd.concat(all_dfs, ignore_index=True)

# Label = column with highest value at each row
df_combined['class'] = df_combined[GESTURE_COLS].idxmax(axis=1).map(inv_map)
df_combined.dropna(subset=['class'], inplace=True)
df_combined['class'] = df_combined['class'].astype(int)

print(f"\nDataset shape : {df_combined.shape}")
print("Class counts  :")
print(df_combined['class'].value_counts().rename(gesture_map))

# ── Train / Test split ────────────────────────────────────────
X = df_combined[GESTURE_COLS].values
y = df_combined['class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

# ── Train ML models ───────────────────────────────────────────
models = {
    "Random Forest" : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Decision Tree" : DecisionTreeClassifier(random_state=42),
    "KNN"           : KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    t0     = time.time()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = {
        "model"   : model,
        "accuracy": acc,
        "time"    : round(time.time()-t0, 2),
        "y_pred"  : y_pred
    }
    print(f"\n{name}: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred,
          target_names=[gesture_map[i] for i in sorted(gesture_map)]))

# ── Accuracy bar chart ────────────────────────────────────────
names = list(results.keys())
accs  = [results[n]['accuracy'] * 100 for n in names]

plt.figure(figsize=(8, 5))
bars = plt.bar(names, accs,
               color=['#2ecc71','#3498db','#e74c3c'], edgecolor='black')
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{acc:.2f}%", ha='center', fontweight='bold')
plt.ylim(0, 110)
plt.title("ML Accuracy Comparison — HandOpen / HandClose / Rest")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

# ── Confusion matrices ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
tick_labels = [gesture_map[i] for i in sorted(gesture_map)]
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels)
    ax.set_title(f"{name}  |  {res['accuracy']*100:.2f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.show()

# ── Save best model ───────────────────────────────────────────
with open("emg_gesture_model.pkl","wb") as f: pickle.dump(results['Random Forest']['model'], f)
with open("emg_scaler.pkl","wb")        as f: pickle.dump(scaler, f)
with open("emg_gesture_map.pkl","wb")   as f: pickle.dump(gesture_map, f)
with open("emg_columns.pkl","wb")       as f: pickle.dump(GESTURE_COLS, f)
print("\nSaved: model, scaler, gesture_map, columns")