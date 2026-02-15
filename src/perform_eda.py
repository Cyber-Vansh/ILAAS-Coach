import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data-cleaned/processed_student_mat.csv")

print(df.shape)
print(df.isnull().sum())

print(df["risk_level"].value_counts())

sns.countplot(x=df["risk_level"])
plt.show()

print(df.groupby("risk_level").mean(numeric_only=True))
