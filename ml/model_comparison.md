# Model Comparison — Customer Churn

## Final tuned model Evaluation on the Test Set with threshold standard (0.5)

The models were tuned using cross-validation on the training set. The following metrics report the final performance of each selected model on the test set. (threshold = 0.5)

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC | Brier score |
| ------------------- | -------: | --------: | -----: | -------: | ------: | ----------: |
| Logistic Regression | 0.8024 — |  0.6535 — | 0.5438 | 0.5936 — |  0.8392 |      0.1351 |
| Random Forest       | 0.7956 — |  0.6378 — | 0.5321 | 0.5802 — |  0.8392 |           ? |
| XGBoost             | 0.8048 — |  0.6667 — | 0.5294 | 0.5902 — |  0.8472 |           ? |

Confusion matrix:

| Model               | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :------------------ | :-------------------------------------- |
| Logistic Regression | `[[926, 109], [165 209]]`               |
| Random Forest       | `[[922, 113], [175 199]]`               |
| XGBoost             | `[[936, 99],  [176, 1981]]`             |

---

---

# Final tuned model Evaluation on the Test Set with Tuned threshold

Logistic Regression — Threshold Tuning Results\_

| Model / Strategy             | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| :--------------------------- | --------: | -------: | --------: | -----: | -------: | ------: |
| Baseline                     |      0.36 |   0.7665 |    0.5470 | 0.7005 |   0.6143 |  0.8420 |
| Balanced                     |      0.60 |   0.7622 |    0.5399 | 0.7059 |   0.6118 |  0.8416 |
| RandomOverSampler            |      0.61 |   0.7679 |    0.5495 | 0.6979 |   0.6148 |  0.8421 |
| SMOTENC                      |      0.53 |   0.7509 |    0.5230 | 0.6979 |   0.5979 |  0.8210 |
| L2 Regularization (`C=0.01`) |      0.36 |   0.7672 |    0.5485 | 0.6952 |   0.6132 |  0.8378 |

Logistic Regression Confusion Matrices — Threshold Tuning Results:

| Model / Strategy             | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--------------------------- | :-------------------------------------- |
| Baseline                     | `[[818, 217], [112, 262]]`              |
| Balanced                     | `[[810, 225], [110, 264]]`              |
| RandomOverSampler            | `[[821, 214], [113, 261]]`              |
| SMOTENC                      | `[[797, 238], [113, 261]]`              |
| L2 Regularization (`C=0.01`) | `[[821, 214], [114, 260]]`              |

---

XGBoost — Threshold Tuning Results:

| Model / Strategy | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| :--------------- | --------: | -------: | --------: | -----: | -------: | ------: |
| XGBoost          |      0.35 |   0.7800 |    0.5684 | 0.7112 |   0.6318 |  0.8472 |

XGBoost Confusion Matrices — Threshold Tuning Results:

| Model / | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :------ | :-------------------------------------- |
| XGBoost | `[[833, 202], [108, 266]]`              |

---

---

## Evaluation Metrics

- **Accuracy:** overall proportion of correct predictions.
- **Precision:** proportion of predicted churners who actually churn.
- **Recall:** proportion of actual churners correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** ability to distinguish churners from non-churners across different classification thresholds.

## Model Selection

Hyperparameter tuning and model selection were performed using cross-validation on the training set. The test set was reserved for final evaluation.

## Feature Selection

A preliminary feature selection analysis was conducted to investigate the relationships between the input features and their association with the target variable.

The following statistical methods were used:

- **Pearson and Spearman correlation coefficients** to assess linear and monotonic relationships among numerical features.
- **Chi-square test and Cramér's V** to evaluate associations involving categorical features.

Based on these analyses, feature selection experiments were performed to assess whether removing less informative features could improve model performance.

However, the resulting models did not show any meaningful improvement in the evaluation metrics compared with the models trained using the original feature set. Therefore, feature selection did not provide a measurable performance benefit in this project.

## Final Model Selection

After evaluating the models, **XGBoost with a tuned decision threshold of 0.35** was selected as the most suitable model for the customer churn prediction task.

### Is this choice consistent with the project objective?

Yes, **if** the company prioritizes identifying as many potential churners as possible, even at the cost of contacting some additional customers, this trade-off is reasonable.

Using XGBoost with a decision threshold of 0.35, and compared it with the same XGBoost model using the default threshold of 0.50:

- **Meets the minimum recall target of 70%.**
- **Reduces false negatives from 176 to 108**, allowing the model to identify more customers who are likely to churn.
- **Improves the F1-score**.
- **Outperforms the linear models with oversampling and tuned decision thresholds** across the evaluated metrics.

The main trade-off is an increase of **103 false positives**. This means that more customers who would not actually churn may be incorrectly flagged as potential churners, requiring additional retention campaign resources.

However, the actual business value of this strategy depends on the cost of retention campaigns, the effectiveness of the interventions, and the value of the customers retained. A cost-benefit analysis would therefore be necessary to confirm the economic advantage of this decision threshold.

### Limitations and Future Improvements:

**Test set usage**

Some preliminary exploratory experiments, documented in the Experiments/ folder, used the test set to compare hyperparameter configurations. Therefore, the final test results may be influenced by these earlier experiments and should be interpreted with this limitation in mind.

**Threshold tuning**

Threshold tuning was performed for XGBoost but not for Random Forest. Since the baseline XGBoost model showed better performance than the baseline Random Forest model, threshold optimization was prioritized for XGBoost. Consequently, the potential benefits of threshold tuning for Random Forest remain unexplored.

**Class-imbalance handling**

The following class-imbalance parameters were not tested in this project:

class_weight="balanced" for Random Forest.

scale_pos_weight for XGBoost.

These parameters could potentially improve the models' ability to identify customers likely to churn. Testing them, together with further hyperparameter optimization and threshold tuning, is planned for future work.
