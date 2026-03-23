# Customer Behaviour Prediction

A machine learning project that predicts whether a customer will purchase a product based on demographic and financial features. Four classification models are built and compared: Naive Bayes, Support Vector Machine, Decision Tree, and Neural Network.

## Dataset

**Source:** [Customer Behaviour Dataset — Kaggle](https://www.kaggle.com/datasets/denisadutca/customer-behaviour)  
**Size:** 400 rows × 5 columns

| Feature | Type | Description |
|---|---|---|
| User ID | int | Unique identifier (dropped before modeling) |
| Gender | categorical | Male / Female → encoded as 0 / 1 |
| Age | int | Age of the customer (18–60) |
| EstimatedSalary | int | Annual estimated salary (15k–150k) |
| Purchased | int | Target variable — 0 (No) or 1 (Yes) |

**Class distribution:** ~64% did not purchase, ~36% purchased.

## Project Structure

```
Consumer Behavior Modeling/
│
├── customer-behaviour-prediction.ipynb   # Main notebook
├── Customer_Behaviour.csv                # Dataset
└── README.md
```

## Methodology

### 1. Data Preparation
- Dropped `User ID` (irrelevant to prediction)
- Encoded `Gender`: Male → 0, Female → 1
- Confirmed no missing values

### 2. Exploratory Data Analysis (EDA)
- KDE plots for `Age` and `EstimatedSalary` by purchase decision and gender
- Key findings:
  - Customers over 43 are more likely to purchase regardless of salary
  - Customers earning over $100k tend to purchase regardless of age
  - Gender has minimal influence on purchasing behavior
  - Highest correlation with target: `Age` (0.62)

### 3. Models

| Model | Test Accuracy | Notes |
|---|---|---|
| Naive Bayes (MultinomialNB) | ~81% | Baseline; `EstimatedSalary` scaled to (18, 60) range for compatibility |
| Support Vector Machine (SVC) | ~94% | RBF kernel; best overall performance |
| Decision Tree | ~90% | Good interpretability |
| Neural Network (MLPClassifier) | ~93% | 2 hidden layers (64 → 32), ReLU, Adam optimizer |

### 4. Model Improvement
- Tested multiple `test_size` values (0.1, 0.15, 0.2, 0.3) — 0.2 performed best
- Used K-Fold cross-validation (k=10) to identify and remove outlier folds
- Removed ~80 problematic data points, improving Naive Bayes accuracy from 78% → 81%

## Setup & Usage

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
```

### Run
Open the notebook in Jupyter:
```bash
jupyter notebook customer-behaviour-prediction.ipynb
```

Run cells sequentially. Note: the Neural Network section uses `sklearn.neural_network.MLPClassifier` — no TensorFlow required.

## Key Insights

- **Age is the strongest predictor** of purchase behavior (correlation: 0.62)
- **SVM outperforms all other models** at 94% accuracy on this dataset
- Naive Bayes, while fast, is limited here due to the mixed feature types and scale sensitivity
- The dataset is small (400 rows) — sklearn's `MLPClassifier` is recommended over TensorFlow/Keras for the neural network to avoid kernel crashes
