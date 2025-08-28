# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Basic info
print("\n--- Dataset Info ---")
print(df.shape)
print(df.head())
print(df['Class'].value_counts())

# Fraud ratio
fraud_ratio = df['Class'].mean() * 100
print(f"\nFraudulent transactions: {fraud_ratio:.4f}%")

# Transaction Amount distribution
plt.figure()
df['Amount'].plot(kind='hist', bins=100, title='Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

# Transaction Time distribution
plt.figure()
df['Time'].plot(kind='hist', bins=100, title='Transaction Time Distribution')
plt.xlabel('Time (seconds since first transaction)')
plt.ylabel('Frequency')
plt.show()

# Fraud vs Non-Fraud counts
plt.figure()
df['Class'].value_counts().plot(kind='bar', title='Class Distribution (0 = Legit, 1 = Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Fraud rate by hour of day
df['hour'] = (df['Time'] // 3600) % 24
fraud_rate = df.groupby('hour')['Class'].mean() * 100

plt.figure()
fraud_rate.plot(kind='bar', title='Fraud Rate by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Fraud Rate (%)')
plt.show()
