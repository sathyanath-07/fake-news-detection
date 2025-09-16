import pandas as pd

# Load datasets
true = pd.read_csv("E:\\fake\\data\\true.csv")
fake = pd.read_csv("E:\\fake\\data\\Fake.csv") 

# Add labels
true["label"] = 0
fake["label"] = 1

# Merge
df = pd.concat([true, fake], ignore_index=True)

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save as train.csv
df.to_csv("train.csv", index=False)

print("✅ train.csv created with", df.shape[0], "rows")
