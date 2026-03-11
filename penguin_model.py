import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import combinations


# ─────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────
def load_data():
    df = pd.read_csv("penguins.csv")

    num_col = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']

    for i in num_col:
        mask = (df["Species"] == "Adelie") & (df["OriginLocation"] == "Torgersen")
        mean_fill = df.loc[mask, i].mean()
        df.loc[mask, i] = df.loc[mask, i].fillna(mean_fill).round(1)

    for i in num_col:
        mask = (df["Species"] == "Gentoo") & (df["OriginLocation"] == "Biscoe")
        mean_fill = df.loc[mask, i].mean()
        df.loc[mask, i] = df.loc[mask, i].fillna(mean_fill).round(1)

    origin_location = {"Dream": 1, "Biscoe": 2, "Torgersen": 3}
    df["OriginLocation"] = df["OriginLocation"].map(origin_location)

    return df


def prepare_data(df, feat1, feat2, class1, class2):
    filtered = df[df["Species"].isin([class1, class2])].copy()
    filtered["Species"] = filtered["Species"].map({class1: 1, class2: -1})

    train_df, test_df = train_test_split(
        filtered, test_size=0.4, stratify=filtered["Species"], random_state=42
    )

    x_train = train_df[[feat1, feat2]]
    y_train  = train_df["Species"]
    x_test   = test_df[[feat1, feat2]]
    y_test   = test_df["Species"]

    return x_train, y_train, x_test, y_test


def scale_data(x_train, x_test):
    scaler         = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_scaled  = pd.DataFrame(scaler.transform(x_test),      columns=x_test.columns,  index=x_test.index)
    return x_train_scaled, x_test_scaled


# ─────────────────────────────────────────
# Helper
# ─────────────────────────────────────────
def signum(x):
    return 1 if x >= 0 else -1


# ─────────────────────────────────────────
# Perceptron
# ─────────────────────────────────────────
def perceptron_train(x, y, w0, w1, w2, bias, lr, epochs):
    for _ in range(epochs):
        for x1, x2, target in zip(x.iloc[:, 0], x.iloc[:, 1], y):
            net    = w1 * x1 + w2 * x2 + w0 * bias
            y_pred = signum(net)
            if y_pred != target:
                error = target - y_pred
                w1 = w1 + lr * error * x1
                w2 = w2 + lr * error * x2
                w0 = w0 + lr * error * bias
    return w1, w2, w0


def perceptron_test(x_test, y_test, w0, w1, w2, bias):
    predictions = []
    TP = TN = FP = FN = 0

    for x1, x2, target in zip(x_test.iloc[:, 0], x_test.iloc[:, 1], y_test):
        net  = w1 * x1 + w2 * x2 + w0 * bias
        pred = signum(net)
        predictions.append(pred)

        if   pred ==  1 and target ==  1: TP += 1
        elif pred == -1 and target == -1: TN += 1
        elif pred ==  1 and target == -1: FP += 1
        elif pred == -1 and target ==  1: FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    cm = np.array([[TP, FN],
                   [FP, TN]])
    return predictions, accuracy, cm


# ─────────────────────────────────────────
# Adaline
# ─────────────────────────────────────────
def adaline_train(x, y, w0, w1, w2, bias, lr, epochs, mse_threshold):
    mse = float('inf')
    for _ in range(epochs):
        total_error = 0
        for x1, x2, target in zip(x.iloc[:, 0], x.iloc[:, 1], y):
            net    = w1 * x1 + w2 * x2 + w0 * bias
            error  = target - net
            w1 = w1 + lr * error * x1
            w2 = w2 + lr * error * x2
            w0 = w0 + lr * error * bias
            total_error += error ** 2
        mse = total_error / len(y)
        if mse < mse_threshold:
            break
    return w1, w2, w0, mse


def adaline_test(x_test, y_test, w0, w1, w2, bias):
    predictions = []
    TP = TN = FP = FN = 0

    for x1, x2, target in zip(x_test.iloc[:, 0], x_test.iloc[:, 1], y_test):
        net  = w1 * x1 + w2 * x2 + w0 * bias
        pred = signum(net)
        predictions.append(pred)

        if   pred ==  1 and target ==  1: TP += 1
        elif pred == -1 and target == -1: TN += 1
        elif pred ==  1 and target == -1: FP += 1
        elif pred == -1 and target ==  1: FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    cm = np.array([[TP, FN],
                   [FP, TN]])
    return predictions, accuracy, cm


# ─────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────
def plot_confusion_matrix(cm, class1, class2):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Pred {class1}", f"Pred {class2}"])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"Actual {class1}", f"Actual {class2}"])
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, fontweight="bold")
    plt.colorbar(im)
    plt.tight_layout()
    return fig


def plot_decision_boundary(x_train, y_train, x_test, y_test, w0, w1, w2, bias, feat1, feat2, class1, class2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {1: "blue", -1: "green"}
    labels = {1: class1, -1: class2}

    for ax, x_data, y_data, title in [
        (axes[0], x_train, y_train, "Training Data"),
        (axes[1], x_test,  y_test,  "Test Data")
    ]:
        for cls, color in colors.items():
            mask = y_data == cls
            ax.scatter(x_data.loc[mask, feat1], x_data.loc[mask, feat2],
                       color=color, label=labels[cls], alpha=0.6)

        x1_vals = np.linspace(x_data[feat1].min(), x_data[feat1].max(), 100)
        if w2 != 0:
            x2_vals = -(w1 * x1_vals + w0 * bias) / w2
            ax.plot(x1_vals, x2_vals, color="red", linewidth=2, label="Decision Boundary")

        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_title(f"{title}: {feat1} vs {feat2}")
        ax.legend()

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    FEAT1, FEAT2   = "CulmenLength", "CulmenDepth"
    CLASS1, CLASS2 = "Adelie", "Gentoo"
    BIAS, LR, EPOCHS, MSE_THRESHOLD = 1, 0.01, 100, 0.01

    df = load_data()
    x_train, y_train, x_test, y_test = prepare_data(df, FEAT1, FEAT2, CLASS1, CLASS2)
    x_train_scaled, x_test_scaled    = scale_data(x_train, x_test)

    # Perceptron
    w0, w1, w2 = np.random.rand() * 0.01, np.random.rand() * 0.01, np.random.rand() * 0.01
    w1, w2, w0 = perceptron_train(x_train, y_train, w0, w1, w2, BIAS, LR, EPOCHS)
    preds, acc, cm = perceptron_test(x_test, y_test, w0, w1, w2, BIAS)
    print("Perceptron Accuracy:", acc)

    # Adaline
    w0, w1, w2 = np.random.rand() * 0.01, np.random.rand() * 0.01, np.random.rand() * 0.01
    w1, w2, w0, mse = adaline_train(x_train_scaled, y_train, w0, w1, w2, BIAS, LR, EPOCHS, MSE_THRESHOLD)
    preds, acc, cm  = adaline_test(x_test_scaled, y_test, w0, w1, w2, BIAS)
    print("Adaline Accuracy:", acc)
    print("MSE:", mse)