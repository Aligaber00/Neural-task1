# 🐧 Penguin Species Classifier — Perceptron & Adaline

A single-layer neural network implementation from scratch using the **Perceptron** and **Adaline** learning algorithms to classify penguin species from the Palmer Penguins dataset.

---

## 📋 Project Overview

This project implements two classic neural network algorithms without using any ML libraries for the learning logic:

- **Perceptron** — uses the sign of the output for weight updates
- **Adaline** — uses the raw linear output for weight updates (gradient descent on MSE)

Both algorithms are binary classifiers — the user selects any two features and any two classes to train and evaluate the model.

---

## 📁 Project Structure

```
📁 project/
├── neural_proj.ipynb   # Notebook for exploration, analysis, and visualization
├── penguin_model.py    # All functions (train, test, plot) — no hardcoded inputs
├── app.py              # Streamlit GUI app
└── penguins.csv        # Dataset
```

---

## 📊 Dataset

The **Palmer Penguins** dataset contains 150 samples across 3 species (50 each):

| Species | Location |
|---|---|
| Adelie | Torgersen |
| Chinstrap | Dream |
| Gentoo | Biscoe |

**Features available:**
- CulmenLength
- CulmenDepth
- FlipperLength
- BodyMass
- OriginLocation

**Missing values** are filled using the group mean per species and location.

---

## 🧠 Algorithms

### Perceptron
- Activation: `signum(net)` during both training and testing
- Weight update: `w = w + lr * (target - signum(net)) * x`
- Stops after fixed number of epochs

### Adaline
- Activation: linear `net` during training, `signum(net)` during testing
- Weight update: `w = w + lr * (target - net) * x`
- Stops early if MSE drops below threshold
- Requires **feature normalization** (StandardScaler) to avoid overflow

---

## 🖥️ GUI — Streamlit App

The GUI allows full control over the experiment without touching any code.

**User Inputs:**
- Select 2 features
- Select 2 classes
- Learning rate (eta)
- Number of epochs
- MSE threshold (Adaline only)
- Add bias or not (checkbox)
- Choose algorithm: Perceptron or Adaline (radio button)

**Outputs:**
- Accuracy score
- Confusion matrix
- Decision boundary plot (training + test data)

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib scikit-learn streamlit
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```
Make sure `penguins.csv` and `penguin_model.py` are in the same folder as `app.py`.

### 3. Run the notebook
Open `neural_proj.ipynb` in Jupyter or VS Code.
Change the variables in the **User Input Variables** cell at the top:
```python
FEAT1         = "CulmenLength"
FEAT2         = "CulmenDepth"
CLASS1        = "Adelie"
CLASS2        = "Gentoo"
LR            = 0.01
EPOCHS        = 100
MSE_THRESHOLD = 0.01
BIAS          = 1
```
Then run all cells.

---

## 📈 Train / Test Split

Each class has **50 samples**:
- **30 samples** per class → training
- **20 samples** per class → testing

Split is done using stratified sampling to ensure equal class distribution.

---

## 📉 Evaluation

- **Accuracy** — percentage of correctly classified samples
- **Confusion Matrix** — manually computed TP, TN, FP, FN (no sklearn)

---

## ⚙️ File Responsibilities

| File | Responsibility |
|---|---|
| `penguin_model.py` | Pure functions only — load data, train, test, plot. No hardcoded inputs, no running code at import time |
| `app.py` | Streamlit GUI — takes user inputs, calls functions from `penguin_model.py` |
| `neural_proj.ipynb` | EDA, visualization, and experimentation using the same functions |
