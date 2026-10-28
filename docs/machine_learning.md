# Customer Churn Prediction — Model Evaluation and Interpretability Report

## 1. Project Objective

The objective of this project is to predict customer churn using the Telco Customer Churn dataset and identify the factors associated with customers leaving the service.

The project includes data preparation, exploratory analysis, feature selection experiments, model comparison, hyperparameter tuning, decision-threshold optimization, and model interpretability using Permutation Importance and SHAP.

The final model was selected based on predictive performance and the project's business objective: identifying as many potential churners as reasonably possible while maintaining a manageable number of false positives.

---

## 2. Dataset and Preprocessing

The dataset contains 7,043 customer records and 21 columns, including the target variable `Churn` and the customer identifier.

The target variable was encoded as:

- `No` = 0
- `Yes` = 1

The class distribution was:

| Class    | Number of customers | Percentage |
| -------- | ------------------: | ---------: |
| No churn |               5,174 |     73.46% |
| Churn    |               1,869 |     26.54% |

The dataset is therefore moderately imbalanced, with approximately 26.5% of customers belonging to the churn class.

The following preprocessing steps were applied:

- `TotalCharges` was converted to a numeric data type. The 11 blank values were converted to zero.
- The customer identifier (`User_id`, previously `customerID`) was excluded from the predictors.
- The target variable `Churn` was excluded from the input features.
- The remaining 19 original features were used as model inputs.
- Categorical variables were encoded using One-Hot Encoding with `handle_unknown="ignore"`.
- Numerical features and the binary `SeniorCitizen` feature were passed through the tree-model preprocessing pipeline.
- The preprocessing and classification steps were combined into a scikit-learn `Pipeline`.

The data was divided into training and test sets using a stratified split:

- Training set: 5,634 observations.
- Test set: 1,409 observations.
- Test size: 20%.
- Random state: 42.
- Stratification: based on `Churn`.

Stratification preserves approximately the same target-class proportions in both subsets.

The tree-model preprocessing pipeline transforms the 19 original predictors into 45 model-input columns. These include numerical features, the binary feature, and the columns created by One-Hot Encoding.

P.S.:
The original Kaggle dataset **did not include a predefined validation set**. Therefore, the data were split into stratified training (80%, 5,634 observations) and test (20%, 1,409 observations) sets. The training set was used for model development and selection, while the test set was reserved for final evaluation

---

## 3. Exploratory Analysis and Feature Relationships

Before model training, statistical analyses were conducted to investigate relationships among the original input features.

### 3.1 Numerical correlations

**Pearson** and **Spearman** correlation coefficients were used to examine linear and monotonic relationships between numerical variables.

Selected results:

| Feature pair                      | Pearson | Spearman |
| --------------------------------- | ------: | -------: |
| `tenure` – `MonthlyCharges`       |   0.248 |    0.276 |
| `tenure` – `TotalCharges`         |   0.826 |    0.890 |
| `MonthlyCharges` – `TotalCharges` |   0.651 |    0.638 |

The strong relationship between `tenure` and `TotalCharges` is expected because total charges are influenced by the amount of time a customer has been subscribed and by the charges incurred over that period.

These correlations are relevant to feature interpretation because correlated predictors can share predictive information.

### 3.2 Categorical associations

Chi-square tests and Cramér's V were used to investigate associations between categorical features.

Selected results:

| Feature pair                           | Cramér's V |
| -------------------------------------- | ---------: |
| `PhoneService` – `MultipleLines`       |      1.000 |
| `StreamingTV` – `StreamingMovies`      |      0.771 |
| `InternetService` – `OnlineSecurity`   |      0.724 |
| `DeviceProtection` – `StreamingMovies` |      0.734 |
| `InternetService` – `StreamingTV`      |      0.723 |
| `TechSupport` – `OnlineBackup`         |      0.720 |
| `OnlineSecurity` – `OnlineBackup`      |      0.718 |
| `Partner` – `Dependents`               |      0.453 |
| `PhoneService` – `InternetService`     |      0.425 |

The perfect association between `PhoneService` and `MultipleLines` reflects a structural relationship in the dataset: when a customer does not have phone service, the `MultipleLines` variable can take the value `No phone service`.

Other service-related variables are also associated. For example, customers with particular Internet services may have different patterns of security, backup, technical support, and streaming subscriptions.

These associations motivated additional feature-selection experiments and, later, the use of group-based Permutation Importance.

### 3.3 Feature-selection experiments

Feature-selection experiments were conducted with the **Logistic regression** model to determine whether removing less informative or potentially redundant predictors could improve model performance.

The resulting models did not show a meaningful improvement over models trained with the original 19-feature input set. Therefore, the full original feature set was retained.

This decision does not imply that all features contribute equally. It means that removing features did not demonstrate a measurable performance benefit in the experiments conducted.

---

## 4. Model Comparison — Final tuned model Evaluation on the Test Set with threshold standard (0.5)

The candidate models iperparameters were tuned using **cross-validation** on the training set. The following results represent final test-set evaluation at the standard classification threshold of **0.50**.

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |  Brier score |
| ------------------- | -------: | --------: | -----: | -------: | ------: | -----------: |
| Logistic Regression |   0.8024 |    0.6535 | 0.5438 |   0.5936 |  0.8392 |       0.1351 |
| Random Forest       |   0.7956 |    0.6378 | 0.5321 |   0.5802 |  0.8392 | Not recorded |
| XGBoost             |   0.8048 |    0.6667 | 0.5294 |   0.5902 |  0.8472 | Not recorded |

The Brier scores for Random Forest and XGBoost were not available in the recorded results and have not been estimated.

### 4.1 Confusion matrices

The confusion matrices use the format `[[TN, FP], [FN, TP]]`.

| Model               | Confusion matrix           |
| ------------------- | -------------------------- |
| Logistic Regression | `[[926, 109], [165, 209]]` |
| Random Forest       | `[[922, 113], [175, 199]]` |
| XGBoost             | `[[936, 99], [176, 198]]`  |

At the standard threshold, **XGBoost achieved the highest accuracy and ROC-AUC among the three models, while Logistic Regression achieved a slightly higher F1-score.**

However, all three models had recall values below 55%. This means that they missed a substantial proportion of actual churners.

Consequently, the standard threshold was not necessarily the most appropriate operating point for the business objective.

---

## 5. Decision-Threshold Optimization

A classification threshold determines how predicted probabilities are converted into binary predictions.

The default threshold is 0.50. Lowering the threshold generally identifies more positive cases, potentially increasing recall while also increasing false positives.

Because the project prioritizes identifying potential churners, threshold optimization was investigated.

### 5.1 Logistic Regression threshold experiments

The decision thresholds were selected using **out-of-fold (OOF)** predictions generated from the training data

**Logistic Regression** — Threshold Tuning Results:

| Model / Strategy             | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ---------------------------- | --------: | -------: | --------: | -----: | -------: | ------: |
| Baseline                     |      0.36 |   0.7665 |    0.5470 | 0.7005 |   0.6143 |  0.8420 |
| Balanced                     |      0.60 |   0.7622 |    0.5399 | 0.7059 |   0.6118 |  0.8416 |
| RandomOverSampler            |      0.61 |   0.7679 |    0.5495 | 0.6979 |   0.6148 |  0.8421 |
| SMOTENC                      |      0.53 |   0.7509 |    0.5230 | 0.6979 |   0.5979 |  0.8210 |
| L2 Regularization (`C=0.01`) |      0.36 |   0.7672 |    0.5485 | 0.6952 |   0.6132 |  0.8378 |

The corresponding confusion matrices were:

| Model / Strategy             | Confusion matrix           |
| ---------------------------- | -------------------------- |
| Baseline                     | `[[818, 217], [112, 262]]` |
| Balanced                     | `[[810, 225], [110, 264]]` |
| RandomOverSampler            | `[[821, 214], [113, 261]]` |
| SMOTENC                      | `[[797, 238], [113, 261]]` |
| L2 Regularization (`C=0.01`) | `[[821, 214], [114, 260]]` |

These experiments improved recall compared with the standard-threshold model comparison. However, their F1-scores remained below the F1-score achieved by the selected XGBoost model at its "best" tuned threshold.

### 5.2 XGBoost — Threshold Tuning Results:

XGBoost was selected for additional hyperparameter tuning and decision-threshold optimization.

The final tuned pipeline was selected using `RandomizedSearchCV` with five-fold stratified cross-validation and F1-score as the tuning metric.

The best recorded cross-validation F1-score was approximately 0.5928.

The selected hyperparameters were:

| Hyperparameter     | Selected value |
| ------------------ | -------------: |
| `colsample_bytree` |         0.8212 |
| `gamma`            |         1.1854 |
| `learning_rate`    |        0.01274 |
| `max_depth`        |              4 |
| `min_child_weight` |              1 |
| `n_estimators`     |            354 |
| `subsample`        |         0.9213 |

The final XGBoost results at the threshold of 0.35 were:
(The decision thresholds were selected using **out-of-fold (OOF)** predictions generated from the training data)

| Model   | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------- | --------: | -------: | --------: | -----: | -------: | ------: |
| XGBoost |      0.35 |   0.7800 |    0.5684 | 0.7112 |   0.6318 |  0.8472 |

Confusion matrix:

`[[833, 202], [108, 266]]`

### 5.3 Comparison of XGBoost thresholds

| Metric          | Threshold 0.50 | Threshold 0.35 |
| --------------- | -------------: | -------------: |
| Accuracy        |         0.8048 |         0.7800 |
| Precision       |         0.6667 |         0.5684 |
| Recall          |         0.5294 |         0.7112 |
| F1-score        |         0.5902 |         0.6318 |
| ROC-AUC         |         0.8472 |         0.8472 |
| True negatives  |            936 |            833 |
| False positives |             99 |            202 |
| False negatives |            176 |            108 |
| True positives  |            198 |            266 |

Reducing the threshold from 0.50 to 0.35 resulted in:

- Recall increasing from 52.94% to 71.12%.
- F1-score increasing from 0.5902 to 0.6318.
- False negatives decreasing from 176 to 108.
- False positives increasing from 99 to 202.
- Accuracy decreasing from 80.48% to 78.00%.

ROC-AUC remains unchanged because it evaluates ranking performance across thresholds rather than the binary predictions at one specific threshold.

The selected threshold identifies 68 additional actual churners in the test set, at the cost of 103 additional false positives.

### 5.4 Business interpretation

**If** the company prioritizes identifying as many potential churners as possible, accepting additional false positives may be reasonable.

A false negative represents a customer who actually churns but is not flagged by the model. A false positive represents a customer flagged as a potential churner who does not actually churn.

The threshold of 0.35 reduces missed churners substantially, but requires more retention-campaign contacts.

The economic value of this decision depends on:

- The cost of contacting and incentivizing customers.
- The effectiveness of retention interventions.
- The expected value of customers retained.
- The company's capacity to act on the predictions.

A cost-benefit analysis would be necessary to determine whether the lower threshold maximizes actual business value.

---

## 6. Evaluation Metrics

The following metrics were used to evaluate the classification models:

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of customers predicted to churn who actually churn.
- **Recall:** proportion of actual churners correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** ability to distinguish churners from non-churners across different classification thresholds.
- **Brier score:** mean squared difference between predicted probabilities and binary outcomes; lower values indicate better probabilistic predictions.

F1-score was used as the primary optimization metric for the selected XGBoost model, while recall was also important because missing potential churners can be costly for a retention strategy.

---

## 7. Model Interpretability

After selecting and evaluating the final model, interpretability analyses were performed to understand which features contribute to its predictions.

Two complementary techniques were used:

1. Permutation Importance, evaluated using F1-score at the selected threshold.
2. SHAP, used to examine the contribution of individual transformed features and summarize their importance at the original-feature level.

These analyses investigate model behavior. They do not establish that the identified features cause customer churn.

### 7.1 Why use both Permutation Importance and SHAP?

The methods answer different questions.

**Permutation Importance** measures how much predictive performance deteriorates when a feature's association with the observations is disrupted. It evaluates how much the fitted model relies on that feature for its current predictions.

**SHAP** attributes contributions to individual model predictions. It helps explain which features push predictions toward churn or away from churn.

Using both methods provides complementary information:

- Permutation Importance evaluates the effect of disrupting feature information on model performance.
- SHAP describes the direction and magnitude of feature contributions to model predictions.

Their numerical values are not directly comparable because they measure different quantities and use different scales.

---

## 8. Permutation Importance — Individual Features

### 8.1 Methodology

Permutation Importance was calculated using the complete fitted XGBoost pipeline and the held-out test set.

The evaluation settings were:

- Evaluation dataset: `X_test`, `y_test`.
- Scoring metric: F1-score.
- Classification threshold: 0.35.
- Number of repetitions: 20.
- Random state: 42.
- Parallel processing: `n_jobs=-1`.

The threshold of 0.35 was fixed before the permutation analysis.

For each repetition, one feature was shuffled across observations while the other features were left unchanged. The fitted pipeline then generated new predictions and the resulting F1-score was recorded.

Importance was defined as:

`Baseline F1-score − F1-score after permutation`

The baseline test F1-score at the selected threshold was **0.6318**.(xgboost final model)

A large positive importance indicates that disrupting the feature substantially reduces predictive performance. An importance near zero indicates that the individual permutation had little measurable effect on F1. A small negative value can occur because of sampling variability and model interactions; it does not establish that the feature is inherently harmful.

### 8.2 Results

| Feature          | Mean importance | Standard deviation |
| ---------------- | --------------: | -----------------: |
| Contract         |          0.1449 |             0.0122 |
| tenure           |          0.0662 |             0.0097 |
| InternetService  |          0.0240 |             0.0065 |
| SeniorCitizen    |          0.0086 |             0.0028 |
| PaperlessBilling |          0.0073 |             0.0070 |
| TotalCharges     |          0.0068 |             0.0047 |
| OnlineSecurity   |          0.0066 |             0.0076 |
| MultipleLines    |          0.0063 |             0.0047 |
| MonthlyCharges   |          0.0057 |             0.0088 |
| PhoneService     |          0.0052 |             0.0010 |
| TechSupport      |          0.0052 |             0.0053 |
| Dependents       |          0.0045 |             0.0037 |
| PaymentMethod    |          0.0044 |             0.0050 |
| StreamingTV      |          0.0031 |             0.0039 |
| StreamingMovies  |          0.0029 |             0.0016 |
| gender           |          0.0016 |             0.0013 |
| OnlineBackup     |          0.0013 |             0.0036 |
| DeviceProtection |         -0.0002 |             0.0005 |
| Partner          |         -0.0004 |             0.0004 |

### 8.3 Interpretation

`Contract` is the most important individual feature by a substantial margin, followed by `tenure`.

`InternetService` is the next most influential individual feature according to this evaluation metric.

Most remaining features have relatively small individual permutation importance values. This does not necessarily mean they have no predictive information. Their information may overlap with other predictors or contribute mainly through interactions.

For example, `tenure` and `TotalCharges` are strongly correlated. When one is permuted individually, the other may retain some of the information relevant to the prediction. A similar issue can occur among the various Internet service and subscription features.

This motivates the group-based permutation analysis described next.

---

## 9. Group Permutation Importance — Correlated and Related Features

### 9.1 Why evaluate groups?

Individual permutation can underestimate the importance of correlated or closely related predictors.

If two features contain overlapping information, disrupting only one may not substantially reduce performance because the model can still use the other. Therefore, low individual permutation importance does not always imply that the underlying information is unimportant.

Group permutation provides a complementary perspective by disrupting several related features together.

In this project, the same row permutation was applied to every feature within a group. This preserves the group's internal combinations across rows while breaking the group's original association with the target and with features outside the group.

Group importance was calculated as the baseline F1-score minus the F1-score after group permutation.

### 9.2 Selected groups and results

| Feature group                    | Mean importance | Standard deviation | Mean F1 after permutation |
| -------------------------------- | --------------: | -----------------: | ------------------------: |
| `tenure` + `TotalCharges`        |          0.1152 |             0.0155 |                    0.5166 |
| `PhoneService` + `MultipleLines` |          0.0058 |             0.0044 |                    0.6260 |
| **Internet-service group**       |          0.1127 |             0.0151 |                    0.5191 |
| `Contract` + `tenure`            |          0.2156 |             0.0196 |                    0.4162 |

The **Internet-service group** contained:

- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

### 9.3 Interpretation

**`Contract` + `tenure`**

This group produced the largest observed performance reduction, with a mean F1 decrease of 0.2156. It indicates that the model relies heavily on the combined information represented by contract type and customer tenure.

**`tenure` + `TotalCharges`**

The group produced a mean importance of 0.1152, considerably greater than the individual importance of `TotalCharges`. This supports the interpretation that the two variables jointly contain useful predictive information.

**Internet-service group**

The combined service group produced a mean importance of 0.1127. Although several service features had relatively small individual permutation importance values, their combined information was substantially more important.

**`PhoneService` + `MultipleLines`**

This group produced a relatively small mean importance of 0.0058. The structural relationship between these variables helps explain why their information may be redundant. However, the result applies to the combined group and the chosen F1 evaluation, not to every possible modeling objective.

### 9.4 Limitations of group permutation

The groups were selected because of known associations and correlations, rather than through an exhaustive search over all possible feature combinations.

The groups also overlap. For example, `tenure` appears in both `tenure` + `TotalCharges` and `Contract` + `tenure`. Consequently, group importance values should not be interpreted as independent contributions or added together.

Although a group permutation preserves the relationships within the permuted group, it can create unrealistic combinations between that group and the remaining features. The resulting importance is therefore a diagnostic measure of model reliance, not a causal effect.

The individual and group analyses are complementary: the first shows the effect of disrupting one feature, while the second investigates the combined information contained in selected related features.

---

## 10. SHAP Analysis

### 10.1 Methodology

SHAP (SHapley Additive exPlanations) was used to explain the contributions of individual model inputs to the XGBoost predictions.

The fitted pipeline contains two steps:

1. A `ColumnTransformer` that preprocesses the original 19 features.
2. An XGBoost classifier that receives the resulting 45 transformed columns.

The SHAP analysis used the fitted XGBoost classifier directly, after applying the already-fitted preprocessing step.

The data used for the global SHAP analysis was a reproducible sample of 500 observations from the training set:

- Source: `X_train`.
- Sample size: 500.
- Random state: 42.
- Transformed feature count: 45.
- SHAP implementation: `shap.TreeExplainer`.
- SHAP version: 0.51.0.

The fitted model was not retrained, and the test set was not used to calculate these SHAP explanations.

**This approach provides a sample-based global interpretation of the trained model without repeatedly evaluating or optimizing the final model against the test set.**

### 10.2 SHAP values and their interpretation

For this binary XGBoost model, the calculated SHAP values explain the model's raw output, which is on the log-odds scale.

- Positive SHAP values push a prediction toward the churn class (`Churn = 1`).
- Negative SHAP values push a prediction away from the churn class and toward non-churn.
- The magnitude of a SHAP value indicates the strength of the contribution on the model's output scale.

These values should not be interpreted directly as percentage-point changes in churn probability.

The global importance measure used was the mean absolute SHAP value across the 500 sampled observations:

`Mean absolute SHAP = mean(abs(SHAP value))`

A larger mean absolute value indicates a larger average contribution to model output, regardless of direction.

### 10.3 Global importance of transformed features

The top transformed features by mean absolute SHAP value were:

| Rank | Transformed feature              | Mean absolute SHAP |
| ---: | -------------------------------- | -----------------: |
|    1 | `Contract_Month-to-month`        |           0.669025 |
|    2 | `tenure`                         |           0.360010 |
|    3 | `OnlineSecurity_No`              |           0.234032 |
|    4 | `MonthlyCharges`                 |           0.220441 |
|    5 | `InternetService_Fiber optic`    |           0.182737 |
|    6 | `TechSupport_No`                 |           0.169626 |
|    7 | `PaymentMethod_Electronic check` |           0.148661 |
|    8 | `Contract_Two year`              |           0.138964 |
|    9 | `TotalCharges`                   |           0.101280 |
|   10 | `PaperlessBilling_No`            |           0.088604 |
|   11 | `MultipleLines_No`               |           0.075608 |
|   12 | `StreamingMovies_Yes`            |           0.065853 |
|   13 | `OnlineBackup_No`                |           0.059871 |
|   14 | `Contract_One year`              |           0.048855 |
|   15 | `SeniorCitizen`                  |           0.034218 |
|   16 | `InternetService_DSL`            |           0.032509 |
|   17 | `Dependents_No`                  |           0.028151 |
|   18 | `StreamingTV_Yes`                |           0.026242 |
|   19 | `PaperlessBilling_Yes`           |           0.016310 |
|   20 | `PhoneService_No`                |           0.015155 |

These results show that the model's predictions are strongly influenced by contract category, customer tenure, online security, monthly charges, Internet service type, and technical support.

**One-Hot Encoding** creates separate columns for different categories of the same original variable. For example, `Contract_Month-to-month`, `Contract_One year`, and `Contract_Two year` are treated as separate model inputs by the classifier and therefore appear separately in this table.

### 10.4 SHAP summary plot: direction of contributions

The SHAP summary plot was examined to understand how feature values relate to the direction of model predictions.

In this plot:

- The horizontal axis represents the SHAP contribution.
- Points to the right of zero push the prediction toward churn.
- Points to the left of zero push the prediction away from churn.
- Color represents the feature value: red indicates a high value and blue indicates a low value.

For numerical features such as `tenure`, red represents higher numerical values and blue represents lower values. For One-Hot Encoded categorical features, red corresponds to the dummy value being 1, while blue corresponds to 0.

plot:
![alt text](../ml/models/pics/image17.png)

The observed patterns were:

**`Contract_Month-to-month`**

**The feature is the most important transformed input**. The plot was interpreted as showing that the presence of this contract category generally pushes predictions toward churn.

**`tenure`**

High tenure values appeared primarily on the negative SHAP side, while low tenure values appeared on the positive side. This indicates that customers with shorter tenure tend to receive higher churn predictions, whereas longer tenure tends to push predictions away from churn.

**`OnlineSecurity_No`**

The presence of this category appeared on the positive SHAP side, indicating that the absence of online security tends to push predictions toward churn.

**`MonthlyCharges`**

The distribution was more mixed. Higher monthly charges appeared mainly on the positive side, but contributions occurred on both sides. The model's learned relationship is therefore not uniform across all observations.

**`InternetService_Fiber optic`**

The presence of the Fiber optic category appeared primarily on the positive side, indicating that this category tends to push predictions toward churn in the fitted model.

These observations describe the model's learned associations. They do not establish that these features independently cause churn.

### 10.5 Aggregating SHAP importance to the original 19 features

To make global importance easier to communicate, **the 45 transformed columns were grouped according to their original input features.**

**For each observation, the signed SHAP contributions of the transformed columns belonging to the same original feature were summed. The mean absolute value of that aggregated contribution was then calculated across the 500 observations.**

This produces a summary at the original-feature level while reusing the already-computed SHAP values.

The aggregation is a practical summary of the contributions from the One-Hot Encoded inputs. It is not equivalent to recalculating SHAP directly on the original 19 input features, because the explanation method and feature representation affect how contributions are assigned.

Signed contributions can also offset each other when different categories of the same feature push predictions in opposite directions.

### 10.6 Aggregated SHAP importance — Original features

| Rank | Original feature   | Mean absolute aggregated SHAP |
| ---: | ------------------ | ----------------------------: |
|    1 | `Contract`         |                      0.762021 |
|    2 | `tenure`           |                      0.360010 |
|    3 | `OnlineSecurity`   |                      0.237192 |
|    4 | `MonthlyCharges`   |                      0.220441 |
|    5 | `InternetService`  |                      0.216638 |
|    6 | `TechSupport`      |                      0.174969 |
|    7 | `PaymentMethod`    |                      0.151212 |
|    8 | `PaperlessBilling` |                      0.104913 |
|    9 | `TotalCharges`     |                      0.101280 |
|   10 | `MultipleLines`    |                      0.085419 |
|   11 | `StreamingMovies`  |                      0.076268 |
|   12 | `OnlineBackup`     |                      0.061783 |
|   13 | `StreamingTV`      |                      0.037915 |
|   14 | `SeniorCitizen`    |                      0.034218 |
|   15 | `Dependents`       |                      0.034110 |
|   16 | `PhoneService`     |                      0.017245 |
|   17 | `gender`           |                      0.014007 |
|   18 | `DeviceProtection` |                      0.002793 |
|   19 | `Partner`          |                      0.001585 |

`Contract` remains the most influential original feature, followed by `tenure`. `OnlineSecurity`, `MonthlyCharges`, and `InternetService` also rank highly.

The aggregated values are useful for a concise global summary, while the transformed-feature summary plot provides more detailed information about individual categories and the direction of their contributions.

The two views serve different purposes: the aggregated chart is easier to read at the original-feature level, whereas the transformed SHAP plot is more informative when investigating category-specific effects.

---

## 11. Comparing Permutation Importance and SHAP

The two interpretation methods provide complementary rather than interchangeable measurements.

### 11.1 Areas of agreement

Both analyses identify `Contract` and `tenure` as important features.

- `Contract` has the largest individual Permutation Importance and the largest aggregated SHAP importance.
- `tenure` ranks second in both analyses.
- `InternetService` is also among the more important predictors in both.

This agreement increases confidence that these features play a substantial role in the fitted model's behavior, although it does not establish causality.

### 11.2 Differences between the rankings

Some features rank relatively low in individual Permutation Importance but higher in SHAP.

For example:

- `OnlineSecurity` has an individual Permutation Importance of 0.0066, but a mean absolute aggregated SHAP value of 0.237192.
- `MonthlyCharges` has an individual Permutation Importance of 0.0057, but a mean absolute SHAP value of 0.220441.
- `TechSupport` has an individual Permutation Importance of 0.0052, but a mean absolute SHAP value of 0.174969.

These differences are not necessarily contradictory.

SHAP measures the contributions assigned to model predictions. Permutation Importance measures the deterioration in F1 when feature information is disrupted. If information overlaps with correlated predictors, shuffling one feature may have little impact on F1 even when that feature receives substantial SHAP contributions.

The distinction is especially relevant for service-related features, which are associated with one another.

### 11.3 Comparing the numerical values

The raw values of SHAP and Permutation Importance must not be compared as though they share a common scale.

- Permutation Importance is expressed as a decrease in F1-score.
- SHAP importance is expressed on the model's raw output scale and summarizes the magnitude of contributions to predictions.

The appropriate comparison is therefore based on qualitative patterns, relative rankings within each method, and the information each method provides.

The group-based permutation results offer additional context for features whose individual permutation importance may be low because related features retain overlapping information.

---

## 12. Final Model Selection and Test-Set Evaluation

Preliminary experiments were conducted separately in the `Experiments/` folder. These exploratory analyses **did not** influence the final selection of hyperparameters or the decision threshold.
**The final hyperparameters were selected using cross-validation procedures applied exclusively to the training set. The decision threshold for the final XGBoost model was selected using out-of-fold (OOF) predictions generated on the training data.**

**Only after** completing model selection and threshold optimization was the test set used to evaluate the final models. This separation ensured that **the final evaluation was not used to guide hyperparameter selection or threshold optimization**.

The test set was subsequently used to calculate the final performance metrics and conduct post-hoc model interpretation analyses, including Permutation Importance. SHAP analysis was performed on a sample of 500 observations from the training set. **No further model training or feature selection was performed based on these interpretability results.**

XGBoost with a tuned decision threshold of **0.35** was selected as the final model.

This choice is consistent with the project objective of identifying potential churners, provided that the company is willing to accept additional false positives in exchange for higher recall.

The model achieved:

- Test accuracy: 0.7800.
- Test precision: 0.5684.
- Test recall: 0.7112.
- Test F1-score: 0.6318.
- Test ROC-AUC: 0.8472.

Compared with the same model at threshold 0.50, the selected threshold reduced false negatives from 176 to 108 and increased the F1-score from 0.5902 to 0.6318.

The model's interpretability results indicate that **contract type and customer tenure are especially important**, while Internet-service and related subscription features also contribute to its predictions.

The model should be considered a decision-support tool for prioritizing retention efforts rather than a definitive predictor of which customers will leave.

---

## 13. Limitations and Future Improvements

### 13.1 Probability Calibration

**Probability calibration** was evaluated for Logistic Regression using out-of-fold (OOF) predictions on the training set. The analysis included the Brier Score, comparison with a prevalence-based baseline, and a calibration curve with 10 quantile-based bins.

This provides an initial assessment of probability reliability. However, calibration was not evaluated for Random Forest or XGBoost, so future work should extend the analysis to all three models.

### 13.2 Threshold tuning (OOF)

Threshold optimization was conducted for XGBoost **but not for Random Forest**. Since the baseline XGBoost model showed stronger ROC-AUC and accuracy than Random Forest, threshold optimization was prioritized for XGBoost.

Consequently, the potential benefits of threshold tuning for Random Forest remain unexplored.

### 13.3 Class-imbalance handling

The following parameters were not tested in this project:

- `class_weight="balanced"` for Random Forest.
- `scale_pos_weight` for XGBoost.

These parameters may affect the models' ability to identify churners and could be investigated in future work.

### 13.4 Correlated predictors and feature importance

Correlated and associated features can complicate feature-importance interpretation.

**Individual Permutation Importance** may underestimate the importance of a feature when related predictors retain overlapping information. **Group permutation** provides a complementary view, but the selected groups are not exhaustive and can overlap.

SHAP also depends on the feature representation and explanation method. The aggregated SHAP values in this report summarize contributions from transformed features and should not be interpreted as a causal or uniquely determined allocation of importance among the original predictors.

Future work could investigate alternative feature-attribution approaches and assess the stability of the explanations across different training samples.

### 13.5 Business impact and calibration

A business-oriented threshold should ideally be selected using the costs of false positives and false negatives, the expected effectiveness of retention interventions, and the value of customers retained.

Future work could include:

- A cost-benefit analysis of candidate decision thresholds.
- Extend probability calibration analysis to Random Forest and XGBoost, and compare calibration and probability-quality metrics across all three models.
- Validation on new or temporally separated data.
- Further investigation of class-imbalance handling.
- Stability analysis of feature importance across different data samples.

---

## 14. Conclusion

This project developed and evaluated machine-learning models for customer churn prediction, selecting a tuned XGBoost pipeline with a decision threshold of 0.35.

The selected operating point achieved a test recall of 71.12% and an F1-score of 0.6318, reducing missed churners relative to the standard threshold of 0.50.

Permutation Importance and SHAP provided complementary interpretations of the model:

- Individual Permutation Importance identified contract type and tenure as the most influential predictors for F1-score at the selected threshold.
- Group Permutation Importance showed that combinations of related features, particularly contract type with tenure and the group of Internet-related services, contain substantial predictive information.
- SHAP identified contract category, tenure, online security, monthly charges, Internet service type, and technical support as important contributors to the model's predictions.
- Aggregated SHAP values provided a more readable summary at the level of the 19 original input features.

The analyses also demonstrate why feature importance must be interpreted carefully when predictors are correlated or associated. Importance rankings depend on the method, evaluation metric, feature representation, and information shared among predictors.

Overall, the project combines predictive modeling with threshold selection and model interpretability, while acknowledging the limitations of the available evaluation and the need for further business validation.
