# Fitness Prediction Using Logistic Regression

**A machine learning project that predicts whether a person is "fit" (Yes/No) from health, lifestyle and body data, and explains which habits matter most.**

![Python](https://img.shields.io/badge/Python-3.x-blue) ![statsmodels](https://img.shields.io/badge/statsmodels-stats%20model-informational) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![pandas](https://img.shields.io/badge/pandas-data%20cleaning-green)

---

## 1. Project at a Glance

| | |
|---|---|
| **Goal** | Predict `is_fit` (1 = fit, 0 = not fit) for a person |
| **Type of problem** | Binary classification (two possible answers) |
| **Model** | Logistic Regression |
| **Data size** | 2,000 records, 1,806 after cleaning |
| **Inputs** | 10 features: age, height, weight, heart rate, blood pressure, sleep, nutrition, activity, smoking, gender |
| **Accuracy** | ~79% (see the note on testing in Section 8) |
| **Separation power (KS)** | 57.5% |
| **Biggest drivers** | Smoking, activity level, nutrition quality, age |

---

## 2. Business Question

- Can we tell if a person is fit using simple health and lifestyle data?
- Which habits push a person towards "fit" and which push them away?
- Who should be given extra attention first (for example in a wellness programme or health insurance plan)?

---

## 3. Dataset

File used: `fitness_dataset.csv`

| Column | Meaning | Type |
|---|---|---|
| `age` | Age in years (18 to 79) | Number |
| `height_cm` | Height in cm | Number |
| `weight_kg` | Weight in kg | Number |
| `heart_rate` | Resting heart rate | Number |
| `blood_pressure` | Blood pressure reading | Number |
| `sleep_hours` | Average hours of sleep | Number |
| `nutrition_quality` | Diet score from 0 to 10 | Number |
| `activity_index` | Activity score from 1 to 5 | Number |
| `smokes` | Does the person smoke? | Yes/No |
| `gender` | M or F | Category |
| **`is_fit`** | **Target: 1 = fit, 0 = not fit** | **Yes/No** |

- About **40%** of people in the cleaned data are labelled fit, so the two classes are reasonably balanced.

---

## 4. Approach (Step by Step)

1. **Loaded and explored the data**: checked columns, types and summary statistics.
2. **Ranked variables using Information Value (IV)**: wrote a custom function that splits each variable into 10 groups and measures how well it separates fit from not fit people.
3. **Removed outliers**: used the IQR rule with box plots on all 8 number columns.
4. **Handled missing values**: `sleep_hours` had 157 missing rows, which were dropped.
5. **Converted text to numbers**: `smokes` became 0/1 and `gender` became a `gender_M` column.
6. **Built the model with statsmodels**: to get p-values, confidence intervals and odds ratios (to explain the results).
7. **Checked for overlap between variables (VIF)**: all values were close to 1, so no variable was removed.
8. **Built the prediction model with scikit-learn**: to get a probability of being fit for every person.
9. **Evaluated the model**: confusion matrix, precision/recall, ROC curve and a KS decile table.

---

## 5. Key Findings

### Which variables matter most (Information Value)

| Rank | Variable | IV | Strength |
|---|---|---|---|
| 1 | activity_index | 0.57 | Very strong |
| 2 | smokes | 0.40 | Strong |
| 3 | nutrition_quality | 0.31 | Strong |
| 4 | age | 0.20 | Medium |
| 5 | gender | 0.09 | Weak |
| 6 | weight_kg | 0.08 | Weak |
| 7 | sleep_hours | 0.08 | Weak |
| 8-10 | height, heart rate, blood pressure | below 0.03 | Very weak |

### What the model says (odds ratios)

| Factor | Effect on the chance of being fit |
|---|---|
| **Smoking** | Odds drop by about **85%** (odds ratio 0.15) |
| **Activity index** (+1 point) | Odds go up about **2.5 times** |
| **Nutrition quality** (+1 point) | Odds go up about **32%** |
| **Sleep** (+1 hour) | Odds go up about **21%** |
| **Age** (+1 year) | Odds drop about **4%** |
| **Blood pressure** (+1 unit) | Odds drop about **2.4%** |
| **Weight** (+1 kg) | Odds drop about **0.8%** |
| **Height and heart rate** | No clear effect (p-values 0.76 and 0.48) |

- Smoking is the biggest single negative. Activity is the biggest single positive.
- The model's overall fit: pseudo R-squared of **0.33**, and the model as a whole is statistically significant.

---

## 6. Model Performance

**Confusion matrix** (1,806 people: 1,086 not fit, 720 fit)

| | Predicted Not Fit | Predicted Fit |
|---|---|---|
| **Actually Not Fit** | 919 | 167 |
| **Actually Fit** | 213 | 507 |

| Measure | Result |
|---|---|
| Accuracy | ~79% |
| Precision (fit) | ~75% (when the model says "fit", it is right 3 out of 4 times) |
| Recall (fit) | ~70% (it finds 7 out of 10 truly fit people) |
| KS statistic | **57.5%** (at the 5th decile) |

**Ranking power (decile table):**
- The top 30% of people ranked by the model contain about **60%** of all fit people.
- The top 50% contain about **85%** of all fit people.

---

## 7. Tech Stack

- **Language:** Python
- **Data handling:** pandas, NumPy
- **Statistical modelling:** statsmodels (Logit, OLS, VIF)
- **Machine learning:** scikit-learn (LogisticRegression, metrics)
- **Charts:** Matplotlib (box plots, ROC curve)
- **Custom code written:** Information Value / Weight of Evidence function, IQR outlier function, KS decile table

---

## 8. Limitations (Be Honest About These)

- **No train/test split.** The model was trained and scored on the same data, so the 79% accuracy and 57.5% KS are likely higher than what new data would give.
- **Likely synthetic data.** Most columns are spread evenly across their range (for example height from 150 to 199), which is typical of generated data. Results show the method works, not that these exact effects hold in real life.
- **Correlation is not cause.** The model shows links, not proof that changing a habit changes fitness.
- **Rows were dropped, not filled.** 157 rows with missing sleep data were removed, which may add bias.
- **The label `is_fit` is not defined in the data.** It is unclear how "fit" was decided, which limits how far the results can be trusted.
- **Gender has a large effect in the model.** This should be questioned before any real-world use.

---

## 9. Next Steps

- Split the data into train and test sets (for example 70/30) and report test results.
- Use cross-validation for more stable numbers.
- Compute the ROC curve and AUC from predicted probabilities.
- Scale the features so the scikit-learn model converges cleanly.
- Fill missing sleep values instead of dropping them.
- Compare with Random Forest or XGBoost.
- Choose a decision cut-off based on business cost, not the default 0.5.
- Build a small Streamlit app where a user enters their details and sees their fitness probability.

---

## 10. How to Run

```bash
# 1. Clone the repo
git clone <your-repo-link>
cd <your-repo-folder>

# 2. Install the libraries
pip install pandas numpy statsmodels scikit-learn matplotlib jupyter

# 3. Place the dataset
#    Put fitness_dataset.csv in the same folder as the notebook

# 4. Open the notebook
jupyter notebook Fitness_Prediction_Logistic_Regression.ipynb
```

> Note: change the file path inside the notebook to a relative path such as `pd.read_csv("fitness_dataset.csv")`.

---

## 11. Project Structure

```
├── Fitness_Prediction_Logistic_Regression.ipynb   # Full analysis and model
├── fitness_dataset.csv                            # Data
└── README.md                                      # This file
```

---

## 12. What I Learned

- How to rank variables using Information Value before modelling.
- How to clean data using the IQR rule and handle missing values.
- How to read p-values, odds ratios and VIF to explain a model in plain words.
- How to judge a model using a confusion matrix, precision, recall and KS.
- Why testing on unseen data matters.

---

## 13. Author

**Rajdip Sanyal**
[GitHub](https://github.com/RajdipSanyalofficial) | [Mail Id](rajdipsanyal43@gmail.com)
