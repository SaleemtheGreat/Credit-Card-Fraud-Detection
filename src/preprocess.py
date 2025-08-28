# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess(path="data/creditcard.csv", test_size=0.2, random_state=42, smote=True):
    # Load dataset
    df = pd.read_csv(path)

    # Features & target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Train-test split (stratify to keep fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Standardize 'Time' and 'Amount'
    scaler = StandardScaler()
    cols_to_scale = ['Time', 'Amount']
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # Handle imbalance with SMOTE
    if smote:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler
