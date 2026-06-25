import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV file
df = pd.read_csv("iris.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Basic Information
print("\nDataset Info:")
print(df.info())

# Average of selected column
avg_sepal_length = df['sepal_length'].mean()
print("\nAverage Sepal Length:", avg_sepal_length)

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# -------------------
# Bar Chart
# -------------------

species_count = df['species'].value_counts()

plt.figure(figsize=(6,4))
species_count.plot(kind='bar')
plt.title("Count of Flower Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

# -------------------
# Scatter Plot
# -------------------

plt.figure(figsize=(6,4))
plt.scatter(df['sepal_length'], df['petal_length'])
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

# -------------------
# Heatmap
# -------------------

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()