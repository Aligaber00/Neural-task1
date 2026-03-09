import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── import everything from your notebook module ──
from penguin_model import (
    load_data,
    perceptron_train,
    perceptron_test,
    adaline_train,
    adaline_test,
    plot_decision_boundary,
    plot_confusion_matrix
)

st.set_page_config(page_title="Penguin Classifier", layout="centered")
st.title("🐧 Penguin Neural Network Classifier")

# ─────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────
df = load_data()

# ─────────────────────────────────────────
# GUI Inputs
# ─────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

all_features = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass", "OriginLocation"]
all_classes  = ["Adelie", "Chinstrap", "Gentoo"]

feat1    = st.sidebar.selectbox("Feature 1", all_features, index=0)
feat2    = st.sidebar.selectbox("Feature 2", all_features, index=1)
class1   = st.sidebar.selectbox("Class 1",   all_classes,  index=0)
class2   = st.sidebar.selectbox("Class 2",   all_classes,  index=2)
lr       = st.sidebar.number_input("Learning Rate (eta)",  value=0.01,  format="%.4f")
epochs   = st.sidebar.number_input("Epochs (m)",           value=100,   min_value=1, step=10)
mse_thr  = st.sidebar.number_input("MSE Threshold",        value=0.01,  format="%.4f")
use_bias = st.sidebar.checkbox("Add Bias", value=True)
algo     = st.sidebar.radio("Algorithm", ["Perceptron", "Adaline"])

# ─────────────────────────────────────────
# Validation
# ─────────────────────────────────────────
if feat1 == feat2:
    st.warning("Please select two different features.")
    st.stop()

if class1 == class2:
    st.warning("Please select two different classes.")
    st.stop()

# ─────────────────────────────────────────
# Prepare Data
# ─────────────────────────────────────────
bias = 1 if use_bias else 0

filtered = df[df["Species"].isin([class1, class2])].copy()
filtered["Species"] = filtered["Species"].map({class1: 1, class2: -1})

train_df, test_df = train_test_split(
    filtered, test_size=0.4, stratify=filtered["Species"], random_state=42
)

x_train = train_df[[feat1, feat2]]
y_train  = train_df["Species"]
x_test   = test_df[[feat1, feat2]]
y_test   = test_df["Species"]

# scale for adaline
scaler         = StandardScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=[feat1, feat2], index=x_train.index)
x_test_scaled  = pd.DataFrame(scaler.transform(x_test),      columns=[feat1, feat2], index=x_test.index)

# ─────────────────────────────────────────
# Train & Evaluate
# ─────────────────────────────────────────
if st.sidebar.button("🚀 Train & Evaluate"):

    w0, w1, w2 = np.random.rand() * 0.01, np.random.rand() * 0.01, np.random.rand() * 0.01

    if algo == "Perceptron":
        w1, w2, w0           = perceptron_train(x_train, y_train, w0, w1, w2, bias, lr, int(epochs))
        preds, acc, cm       = perceptron_test(x_test, y_test, w0, w1, w2, bias)
        x_tr_plot, x_te_plot = x_train, x_test

    else:  # Adaline
        w1, w2, w0, mse      = adaline_train(x_train_scaled, y_train, w0, w1, w2, bias, lr, int(epochs), mse_thr)
        preds, acc, cm       = adaline_test(x_test_scaled, y_test, w0, w1, w2, bias)
        x_tr_plot, x_te_plot = x_train_scaled, x_test_scaled
        st.metric("Final MSE", f"{mse:.5f}")

    # ── Show Results ──
    st.subheader(f"📊 Results — {algo}")
    st.metric("Accuracy", f"{acc * 100:.2f}%")

    st.subheader("🔢 Confusion Matrix")
    st.pyplot(plot_confusion_matrix(cm, class1, class2))

    st.subheader("📈 Decision Boundary")
    st.pyplot(plot_decision_boundary(x_tr_plot, y_train, x_te_plot, y_test,
                                     w0, w1, w2, bias, feat1, feat2, class1, class2))