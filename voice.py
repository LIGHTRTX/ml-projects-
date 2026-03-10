import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

DATA_PATH="/home/cheonma/PROJECTS/innospark/team_1/archive"

def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    pitch = np.mean(librosa.yin(y=y, fmin=50, fmax=500, sr=sr))  # ✅ fixed
    return mfcc, zcr, centroid, rms, pitch

rows = []

for root, dirs, files in os.walk(DATA_PATH):
    for f in tqdm(files):
        if f.endswith(".wav") and "Zone.Identifier" not in f:  # ✅ skip metadata files
            parts = f.split(".")[0].split("-")
            emotion = int(parts[2])
            actor = int(parts[-1])
            path = os.path.join(root, f)
            mfcc, zcr, centroid, rms, pitch = extract_features(path)
            row = {
                "emotion": emotion,
                "actor": actor,
                "zcr": zcr,
                "centroid": centroid,
                "rms": rms,
                "pitch": pitch
            }
            for i, v in enumerate(mfcc):
                row[f"mfcc_{i}"] = v
            rows.append(row)

df = pd.DataFrame(rows)

actor_stats = df.groupby("actor")[["pitch", "rms", "centroid", "zcr"]].mean().reset_index()
emotion_stats = df.groupby("emotion")[["pitch", "rms", "centroid", "zcr"]].mean().reset_index()

plt.figure()
sns.barplot(data=actor_stats,x="actor",y="pitch")
plt.title("Average Pitch per Actor")
plt.xticks(rotation=90)
plt.savefig("graphs/pitch_per_actor.png")
plt.close()

plt.figure()
sns.barplot(data=emotion_stats,x="emotion",y="pitch")
plt.title("Average Pitch per Emotion")
plt.savefig("graphs/pitch_per_emotion.png")
plt.close()

plt.figure()
sns.boxplot(data=df,x="emotion",y="rms")
plt.title("Energy distribution across emotions")
plt.savefig("graphs/emotion_energy_boxplot.png")
plt.close()

plt.figure()
sns.scatterplot(data=df,x="pitch",y="centroid",hue="emotion")
plt.title("Pitch vs Spectral Centroid")
plt.savefig("graphs/pitch_vs_centroid.png")
plt.close()