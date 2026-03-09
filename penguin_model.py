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


def plot_feature_combinations(df):
    features = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
    combos = list(combinations(features, 2))
    colors = {"Adelie": "blue", "Chinstrap": "orange", "Gentoo": "green"}

    fig, axes = plt.subplots(2, 3, figsize=(30, 12))
    axes = axes.flatten()

    for i, (feat1, feat2) in enumerate(combos):
        ax = axes[i]
        for species, color in colors.items():
            subset = df[df["Species"] == species]
            ax.scatter(subset[feat1], subset[feat2], label=species, color=color, alpha=0.6)
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_title(f"{feat1} vs {feat2}")
        ax.legend()

    plt.suptitle("All Feature Combinations by Species", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("penguin_all_combinations.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────
# Helper
# ─────────────────────────────────────────
def signum(x):
    if x >= 0:
        return 1
    else:
        return -1


# ─────────────────────────────────────────
# Perceptron
# ─────────────────────────────────────────
def perceptron_train(x, y, w0, w1, w2, bias, lr=0.01, epochs=100):
    for _ in range(epochs):
        for x1, x2, target in zip(x.iloc[:, 0], x.iloc[:, 1], y):
            net = w1 * x1 + w2 * x2 + w0 * bias
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
        net = w1 * x1 + w2 * x2 + w0 * bias
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
def adaline_train(x, y, w0, w1, w2, bias, lr=0.0001, epochs=1000, mse_threshold=0.01):
    mse = float('inf')
    for _ in range(epochs):
        total_error = 0
        for x1, x2, target in zip(x.iloc[:, 0], x.iloc[:, 1], y):
            net = w1 * x1 + w2 * x2 + w0 * bias
            error = target - net
            w1 = w1 + lr * error * x1
            w2 = w2 + lr * error * x2
            w0 = w0 + lr * error * bias
            total_error += pow(error, 2)
        mse = total_error / len(y)
        if mse < mse_threshold:
            break
    return w1, w2, w0, mse


def adaline_test(x_test, y_test, w0, w1, w2, bias):
    predictions = []
    TP = TN = FP = FN = 0

    for x1, x2, target in zip(x_test.iloc[:, 0], x_test.iloc[:, 1], y_test):
        net = w1 * x1 + w2 * x2 + w0 * bias
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


# ─────────────────────────────────────────
# runs only when executing this file directly
# ignored when imported by app.py
# ─────────────────────────────────────────
if __name__ == "__main__":

    df = load_data()
    plot_feature_combinations(df)

    # split
    train_df, test_df = train_test_split(df, test_size=0.4, stratify=df["Species"], random_state=42)

    # filter 2 classes
    train_df_1 = train_df[train_df["Species"].isin(["Adelie", "Gentoo"])]
    test_df_1  = test_df[test_df["Species"].isin(["Adelie", "Gentoo"])]

    x       = train_df_1.drop(columns="Species")
    y       = train_df_1["Species"].map({"Adelie": 1, "Gentoo": -1})
    x_test  = test_df_1.drop(columns="Species")
    y_test  = test_df_1["Species"].map({"Adelie": 1, "Gentoo": -1})

    bias = 1  # change to 0 to disable bias

    # ── Perceptron ──
    w0, w1, w2 = np.random.rand(), np.random.rand(), np.random.rand()
    w1, w2, w0 = perceptron_train(x[["CulmenLength", "CulmenDepth"]], y, w0, w1, w2, bias, lr=0.01, epochs=100)
    preds, acc, cm = perceptron_test(x_test[["CulmenLength", "CulmenDepth"]], y_test, w0, w1, w2, bias)
    print("Perceptron Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

    # ── Adaline ──
    scaler = StandardScaler()
    x_scaled      = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

    w0, w1, w2 = np.random.rand(), np.random.rand(), np.random.rand()
    w1, w2, w0, mse = adaline_train(x_scaled[["CulmenLength", "CulmenDepth"]], y, w0, w1, w2, bias, lr=0.0001, epochs=1000, mse_threshold=0.01)
    preds, acc, cm  = adaline_test(x_test_scaled[["CulmenLength", "CulmenDepth"]], y_test, w0, w1, w2, bias)
    print("Adaline Accuracy:", acc)
    print("MSE:", mse)
    print("Confusion Matrix:\n", cm)