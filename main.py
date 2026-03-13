import os
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# --- 1. CONFIGURATION ---
# Ensure this folder name matches the one you unzipped
DATASET_DIR = './nsl-kdd'
TRAIN_FILE = 'KDDTrain+.txt'
TEST_FILE = 'KDDTest+.txt'
N_FEATURES = 20

# Column names for the NSL-KDD dataset
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Mapping granular attacks to 5 categories as per academic standards
ATTACK_MAP = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos',
    'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 'ps': 'u2r'
}


# --- 2. PIPELINE FUNCTIONS ---
def preprocess_data(train_path, test_path, n_feats):
    # Load and map labels
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, header=None, names=COLUMN_NAMES)

    for df in [train_df, test_df]:
        df['attack_category'] = df['label'].str.strip().str.lower().map(lambda x: ATTACK_MAP.get(x, 'other'))
        df['binary_label'] = df['attack_category'].apply(lambda x: 0 if x == 'normal' else 1)
        if 'difficulty' in df.columns: df.drop(columns=['difficulty'], inplace=True)

    # Encode Categorical features
    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = test_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Separate Features and Labels
    feature_cols = [c for c in COLUMN_NAMES if c not in ['label', 'difficulty', 'attack_category', 'binary_label']]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['binary_label'].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['binary_label'].values

    # Scale and Select Top K Features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    selector = SelectKBest(score_func=f_classif, k=n_feats)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_and_eval(variant, X_tr, y_tr, X_te, y_te):
    # Defining architectures: Full (128-64-32) vs Lightweight (64-32)
    config = {
        'full': {'hidden_layer_sizes': (128, 64, 32), 'max_iter': 100},
        'light': {'hidden_layer_sizes': (64, 32), 'max_iter': 50}
    }
    model = MLPClassifier(**config[variant], random_state=42, early_stopping=True)

    start = time.time()
    model.fit(X_tr, y_tr)
    t_time = time.time() - start

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)

    return acc, t_time, params


# --- 3. RUN PIPELINE ---
if __name__ == "__main__":
    train_p = os.path.join(DATASET_DIR, TRAIN_FILE)
    test_p = os.path.join(DATASET_DIR, TEST_FILE)

    if os.path.exists(train_p):
        X_tr, X_te, y_tr, y_te = preprocess_data(train_p, test_p, N_FEATURES)

        # Training both models for comparison
        f_acc, f_time, f_params = train_and_eval('full', X_tr, y_tr, X_te, y_te)
        l_acc, l_time, l_params = train_and_eval('light', X_tr, y_tr, X_te, y_te)

        print("\n" + "=" * 30)
        print(" FINAL SUMMARY (GROUP 4)")
        print("=" * 30)
        print(f"{'Metric':<15} {'Full':<10} {'Light'}")
        print(f"{'Accuracy':<15} {f_acc * 100:.2f}% {l_acc * 100:.2f}%")
        print(f"{'Train Time':<15} {f_time:.2f}s    {l_time:.2f}s")
        print(f"{'Params':<15} {f_params:<10} {l_params}")
    else:
        print(f"[ERROR] Dataset not found at {train_p}")