# Student Burnout Risk Prediction from Generative AI Usage

## 1. Introduction, Dataset Overview, and Literature Context

### 1.1 Analytical Context

Generative AI has become a normal part of student learning. Students use tools
such as ChatGPT, Copilot, and Gemini for explanation, summarisation, drafting,
coding, brainstorming, and direct answer generation. This creates opportunities
for personalised support and learning efficiency, but it also raises concerns
about dependency, cognitive offloading, academic integrity, skill retention,
anxiety, and burnout.

The purpose of this analysis is to build and evaluate a supervised machine
learning pipeline that predicts whether a student is at **Low**, **Medium**, or
**High** burnout risk. The practical value is early student support: academic
advisors, wellbeing teams, and education-policy decision-makers could use this
type of model to identify students who may need guidance on study habits, AI
literacy, or mental-health support. The model is not intended to punish students
for AI use.

### 1.2 Dataset Source and Scope

The selected dataset is
**[Impact of AI on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students)**,
published by Lavesh Jadon on Kaggle. It is a synthetic educational dataset
designed to reflect current higher-education use of generative AI tools.

The analysis pipeline loaded **50,000 records with 16 original columns**. The
dataset is suitable for supervised learning because it contains a labelled
outcome, a mixture of numerical and categorical predictors, and realistic
modelling challenges such as class imbalance, skewed AI-usage behaviour, proxy
behavioural indicators, possible confounding, and fairness concerns.

| Dataset Property   | Value                                              |
| ------------------ | -------------------------------------------------- |
| Source             | Kaggle                                             |
| Dataset            | `Impact of AI on Students`                         |
| Records            | 50,000                                             |
| Original columns   | 16                                                 |
| Learning task      | Multi-class classification                         |
| Target variable    | `Burnout_Risk_Level`                               |
| Classes            | Low, Medium, High                                  |
| Practical use case | Student-support and early-warning decision support |

### 1.3 Dataset Attributes

The dataset captures academic, behavioural, technological, institutional, and
wellbeing information. The original variables can be grouped as follows:

| Category                        | Variables                                                                                                   | Type                               | Analytical Role                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------- |
| Identifier                      | `Student_ID`                                                                                                | Identifier                         | Removed before modelling because it does not carry meaningful predictive information.     |
| Academic profile                | `Major_Category`, `Year_of_Study`, `Pre_Semester_GPA`, `Post_Semester_GPA`                                  | Categorical and numerical          | Represents discipline, academic stage, baseline performance, and semester outcome.        |
| AI usage behaviour              | `Weekly_GenAI_Hours`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Tool_Diversity`, `Paid_Subscription` | Numerical, categorical, and binary | Measures AI intensity, use purpose, AI skill, paid access, and tool variety.              |
| Study habits                    | `Traditional_Study_Hours`, `Perceived_AI_Dependency`                                                        | Numerical                          | Captures independent study effort and self-reported reliance on AI support.               |
| Institutional context           | `Institutional_Policy`                                                                                      | Categorical                        | Captures whether AI use is encouraged, allowed with citation, or restricted.              |
| Wellbeing and learning outcomes | `Anxiety_Level_During_Exams`, `Skill_Retention_Score`, `Burnout_Risk_Level`                                 | Numerical and categorical          | Connects AI behaviour and study patterns to anxiety, retained learning, and burnout risk. |

### 1.4 Target Variable and Prediction Objective

The target variable is **`Burnout_Risk_Level`**, a three-class categorical
outcome. The observed target distribution is:

| Burnout Risk Class |  Count | Percentage |
| ------------------ | -----: | ---------: |
| Medium             | 21,144 |    42.288% |
| Low                | 16,369 |    32.738% |
| High               | 12,487 |    24.974% |

The prediction objective is to classify each student into the Low, Medium, or
High burnout-risk class using academic profile, AI usage behaviour, traditional
study habits, perceived AI dependency, anxiety, skill retention, tool access,
and institutional policy. Since the High-risk class is the smallest group and is
the most important from a support perspective, model evaluation must look beyond
raw accuracy. This report therefore uses weighted F1-score, macro F1-score,
ROC-AUC, PR-AUC, class-level recall, confusion matrices, threshold tuning, and
subgroup analysis.

The target should be interpreted carefully. The dataset is synthetic and
observational, so model outputs represent predictive associations rather than
causal evidence that AI use directly causes burnout.

### 1.5 Literature Context

Recent research and industry evidence show that AI in education has both
benefits and risks. The literature supports a modelling framework that includes
academic performance, AI behaviour, study habits, institutional policy, and
wellbeing indicators.

| Study / Source                           | Type                           | Focus                                                        | Key Findings                                                                                                                                              | Relevance to This Analysis                                                                             |
| ---------------------------------------- | ------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Dong, Tang, and Wang (2025)              | Research paper / meta-analysis | AI and student academic achievement                          | Reports a positive overall relationship between AI use and academic achievement, while noting that effects depend on learning context and implementation. | Supports including GPA and skill-retention outcomes while recognising that AI effects are not uniform. |
| HEPI Student Generative AI Survey (2026) | Industry / policy survey       | Student generative AI use in higher education                | Reports that student AI use is almost universal and common in assessed work, concept explanation, summarisation, and idea structuring.                    | Justifies treating AI use as a mainstream study behaviour.                                             |
| Schwartz and Diliberti / RAND (2026)     | Industry research report       | Student AI homework use and critical-thinking concerns       | Reports increasing AI use for homework and substantial student concern that AI use may harm critical thinking.                                            | Supports including traditional study time, dependency, and skill-retention variables.                  |
| Klimova and Pikhart (2025)               | Research paper / mini-review   | AI and student academic wellbeing                            | Highlights personalised-learning benefits but also risks including digital fatigue, loneliness, technostress, and over-reliance.                          | Supports modelling burnout, anxiety, and dependency alongside academic outcomes.                       |
| Francis, Jones, and Smith (2025)         | Research paper / review        | Generative AI in higher education, integrity, and equity     | Discusses academic integrity, ethical use, fairness, institutional policy, and AI literacy.                                                               | Supports including institutional policy, paid tool access, and prompt skill.                           |
| Stanford SCALE Initiative (2026)         | Evidence review                | Evidence base for AI in education                            | Finds that AI adoption is moving faster than rigorous causal evaluation.                                                                                  | Supports cautious interpretation and responsible deployment.                                           |
| López-Meneses et al. (2025)              | Research paper / EDM review    | Educational data mining and predictive modelling             | Reviews educational data mining and predictive modelling for learning analytics and intervention support.                                                 | Supports the use of supervised learning and interpretable early-warning systems.                       |
| Gu, He, and Zhang (2026)                 | Research paper                 | AI technology dependence and learning burnout                | Finds that AI technology dependence is positively associated with learning burnout, mediated by technology acceptance and AI self-efficacy.               | Strongly supports modelling AI dependency and burnout together.                                        |
| Yağcı (2022)                             | Research paper                 | Machine learning for student academic-performance prediction | Demonstrates that ML can predict student outcomes and help identify at-risk learners.                                                                     | Provides a foundation for extending student-risk prediction to burnout.                                |

### 1.6 Research Gap and Analytical Contribution

The literature shows several practical gaps. Many studies focus on academic
performance or academic integrity, while fewer combine academic outcomes with
wellbeing indicators such as anxiety, dependency, skill retention, and burnout
risk. Much of the evidence is survey-based or descriptive, while institutions
need interpretable predictive tools that can support timely interventions. AI
use is also often treated as a single behaviour, even though the risk profile of
using AI for explanation is different from using it to replace independent
thinking.

This report addresses those gaps by combining AI usage intensity, AI use
purpose, prompt skill, tool access, traditional study time, dependency, anxiety,
academic outcomes, and institutional policy in one supervised learning pipeline.
It uses `Burnout_Risk_Level` as the primary target, compares several model
families, evaluates class-level performance, applies threshold tuning for
High-risk recall, and uses explainability and subgroup analysis to make the
results more useful for responsible educational decision-making.

## 2. Exploratory Data Analysis and Feature Engineering

### 2.1 Data Inspection and Quality Checks

The raw dataset contained **50,000 rows and 16 columns** before cleaning and
feature engineering.

| Check                                            |      Result |
| ------------------------------------------------ | ----------: |
| Original dataset shape                           | 50,000 x 16 |
| Missing values                                   |           0 |
| Duplicate rows                                   |           0 |
| Shape after dropping `Student_ID` and duplicates | 50,000 x 15 |
| Numerical columns before feature engineering     |           8 |
| Categorical columns before feature engineering   |           6 |

The original numerical columns were:

`Pre_Semester_GPA`, `Weekly_GenAI_Hours`, `Tool_Diversity`,
`Traditional_Study_Hours`, `Perceived_AI_Dependency`,
`Anxiety_Level_During_Exams`, `Post_Semester_GPA`, and `Skill_Retention_Score`.

The original categorical columns were:

`Major_Category`, `Year_of_Study`, `Primary_Use_Case`,
`Prompt_Engineering_Skill`, `Paid_Subscription`, and `Institutional_Policy`.

### 2.2 Numerical Summary

| Feature                    |   Mean |    Std |    Min | Median |     Max |
| -------------------------- | -----: | -----: | -----: | -----: | ------: |
| Pre_Semester_GPA           |  3.146 |  0.479 |  1.183 |  3.210 |   3.998 |
| Weekly_GenAI_Hours         |  8.428 |  8.269 |  0.000 |  5.800 |  40.000 |
| Tool_Diversity             |  2.800 |  1.188 |  1.000 |  3.000 |   5.000 |
| Traditional_Study_Hours    | 11.209 |  5.156 |  1.000 | 11.180 |  35.860 |
| Perceived_AI_Dependency    |  3.505 |  1.821 |  1.000 |  3.000 |  10.000 |
| Anxiety_Level_During_Exams |  4.271 |  2.144 |  1.000 |  4.000 |  10.000 |
| Post_Semester_GPA          |  3.349 |  0.496 |  1.000 |  3.421 |   4.000 |
| Skill_Retention_Score      | 75.798 | 13.282 | 10.780 | 76.000 | 100.000 |

The most notable descriptive pattern is the right-skew in `Weekly_GenAI_Hours`:
the mean is 8.428 hours, but the median is 5.800 hours and the maximum is 40
hours. This indicates that a smaller group of students use GenAI much more
heavily than the typical student.

### 2.3 Grouped Analysis by Burnout Risk

| Feature                         |   High |    Low | Medium |
| ------------------------------- | -----: | -----: | -----: |
| Pre_Semester_GPA mean           |  3.083 |  3.204 |  3.138 |
| Weekly_GenAI_Hours mean         | 15.215 |  4.644 |  7.349 |
| Traditional_Study_Hours mean    | 10.082 | 11.966 | 11.290 |
| Anxiety_Level_During_Exams mean |  4.889 |  3.928 |  4.170 |
| Post_Semester_GPA mean          |  3.278 |  3.405 |  3.348 |
| Skill_Retention_Score mean      | 74.253 | 76.402 | 76.243 |

Students in the **High** burnout class had much heavier GenAI use than students
in the **Low** class, averaging **15.215 weekly GenAI hours** compared with
**4.644**. They also had lower traditional study hours, higher anxiety, slightly
lower GPA, and lower skill retention. This supports the modelling hypothesis
that burnout risk is linked not only to AI usage volume, but also to the balance
between AI reliance, independent study, academic performance, and wellbeing.

### 2.4 Outlier Detection and Treatment

Outliers were detected using the IQR rule and capped using winsorisation.

| Feature                    | Outlier Count |
| -------------------------- | ------------: |
| Weekly_GenAI_Hours         |         2,583 |
| Post_Semester_GPA          |           346 |
| Pre_Semester_GPA           |           328 |
| Skill_Retention_Score      |           216 |
| Perceived_AI_Dependency    |           190 |
| Traditional_Study_Hours    |           161 |
| Tool_Diversity             |             0 |
| Anxiety_Level_During_Exams |             0 |

Capping was more appropriate than deletion because the dataset had no missing
values or duplicate records, and removing outliers would have discarded valid
student records. The capping step reduced the effect of extreme values while
preserving the full sample size.

### 2.5 Feature Engineering

Feature engineering created seven additional variables and produced an enriched
dataset with shape **50,000 rows x 22 columns**.

| Engineered Feature         | Formula / Rule                                         | Rationale                                                |
| -------------------------- | ------------------------------------------------------ | -------------------------------------------------------- |
| `GPA_Change`               | `Post_Semester_GPA - Pre_Semester_GPA`                 | Measures academic movement during the semester.          |
| `GPA_Drop_Flag`            | 1 if `GPA_Change < 0`, else 0                          | Flags students whose GPA declined.                       |
| `AI_to_Study_Ratio`        | `Weekly_GenAI_Hours / (Traditional_Study_Hours + 1)`   | Measures GenAI use relative to traditional study.        |
| `Total_Study_Load`         | `Weekly_GenAI_Hours + Traditional_Study_Hours`         | Captures total weekly learning workload.                 |
| `High_AI_Use_Flag`         | 1 if GenAI hours exceed the 75th percentile, else 0    | Identifies unusually heavy GenAI users.                  |
| `Dependency_Anxiety_Index` | `Perceived_AI_Dependency * Anxiety_Level_During_Exams` | Captures the interaction between dependency and anxiety. |
| `Skill_Efficiency`         | `Skill_Retention_Score / (Total_Study_Load + 1)`       | Measures retained skill relative to overall study load.  |

After feature engineering, the model used **21 predictors**: 15 numerical
features and 6 categorical features.

### 2.6 Preprocessing and Split

The preprocessing pipeline used:

- Median imputation and standardisation for numerical variables.
- Most-frequent imputation and one-hot encoding for categorical variables.
- Unknown-category handling in the one-hot encoder.
- Label encoding for `Burnout_Risk_Level`.
- An 80/20 stratified train-test split.

| Step                     |                        Result |
| ------------------------ | ----------------------------: |
| Processed feature matrix |                   50,000 x 38 |
| Training set             |                   40,000 rows |
| Test set                 |                   10,000 rows |
| Encoded classes          | 0 = High, 1 = Low, 2 = Medium |

### 2.7 Feature Selection

Mutual information and Random Forest feature importance were used to identify
the most informative predictors.

**Top mutual information features**

| Feature                    | Mutual Information |
| -------------------------- | -----------------: |
| Weekly_GenAI_Hours         |           0.130061 |
| AI_to_Study_Ratio          |           0.107259 |
| High_AI_Use_Flag           |           0.098499 |
| Perceived_AI_Dependency    |           0.077343 |
| Total_Study_Load           |           0.067379 |
| Skill_Efficiency           |           0.065985 |
| Dependency_Anxiety_Index   |           0.060888 |
| Anxiety_Level_During_Exams |           0.017533 |
| Traditional_Study_Hours    |           0.012548 |
| Paid_Subscription_True     |           0.009317 |

**Top Random Forest importance features**

| Feature                  | Importance |
| ------------------------ | ---------: |
| Weekly_GenAI_Hours       |   0.103516 |
| AI_to_Study_Ratio        |   0.087388 |
| Total_Study_Load         |   0.072941 |
| Skill_Efficiency         |   0.069757 |
| Pre_Semester_GPA         |   0.064114 |
| GPA_Change               |   0.061729 |
| Skill_Retention_Score    |   0.060419 |
| Traditional_Study_Hours  |   0.059935 |
| Post_Semester_GPA        |   0.058871 |
| Dependency_Anxiety_Index |   0.047098 |

Both methods consistently ranked AI-use intensity and AI-study balance features
near the top. This indicates that the most predictive information comes from how
much students use GenAI, how that compares with traditional study, and how usage
interacts with dependency and anxiety.

## 3. Modelling Methodology and Optimization

### 3.1 Modelling Setup

The modelling stage compared five supervised classification models:

1. Logistic Regression.
2. K-Nearest Neighbours.
3. Random Forest.
4. XGBoost.
5. Artificial Neural Network.

These models were selected because they cover the main algorithm families
required for a balanced supervised-learning comparison.

| Model                     | Model Family              | Rationale                                                                                                           |
| ------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression       | Linear classifier         | Provides an interpretable baseline, supports regularisation, and produces class probabilities for threshold tuning. |
| K-Nearest Neighbours      | Distance-based classifier | Tests whether similar student profiles form useful local neighbourhoods after scaling.                              |
| Random Forest             | Tree-based ensemble       | Captures nonlinear feature interactions and provides feature-importance evidence.                                   |
| XGBoost                   | Boosting ensemble         | Tests whether sequential boosted trees improve performance on structured tabular data.                              |
| Artificial Neural Network | Deep learning model       | Tests whether nonlinear representation learning improves classification beyond classical models.                    |

Model training used **3-fold stratified cross-validation** and hyperparameter
optimisation with `RandomizedSearchCV`. The scoring metric was **weighted
F1-score**, which is appropriate because the target classes are moderately
imbalanced.

Separate preprocessing pipelines were used:

- Logistic Regression, KNN, and ANN used scaled numerical features.
- Random Forest and XGBoost used imputed but unscaled numerical features because
  scaling is not required for tree-based models.

### 3.2 Cross-Validation Results

| Rank | Model                     | Best CV Weighted F1 |
| ---: | ------------------------- | ------------------: |
|    1 | Logistic Regression       |            0.539006 |
|    2 | XGBoost                   |            0.537058 |
|    3 | Artificial Neural Network |            0.522927 |
|    4 | Random Forest             |            0.517075 |
|    5 | KNN                       |            0.488177 |

### 3.3 Best Hyperparameters

| Model                     | Best Hyperparameters                                                                                                                   |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression       | `C=1`, `penalty='l1'`, `class_weight=None`, `solver='saga'`                                                                            |
| KNN                       | `n_neighbors=31`, `weights='distance'`, `metric='euclidean'`                                                                           |
| Random Forest             | `n_estimators=300`, `max_depth=20`, `max_features='log2'`, `min_samples_split=5`, `min_samples_leaf=2`                                 |
| XGBoost                   | `n_estimators=250`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=1.0`, `min_child_weight=3`, `reg_lambda=1` |
| Artificial Neural Network | `hidden_units=32`, `dropout_rate=0.1`, `optimizer='rmsprop'`, `epochs=15`, `batch_size=64`                                             |

Logistic Regression slightly outperformed XGBoost during cross-validation. This
result is useful because Logistic Regression is simpler, faster, easier to
interpret, and provides calibrated probability-style outputs for threshold
tuning.

## 4. Model Performance and Error Analysis

### 4.1 Test Set Results

The optimised models were evaluated on the held-out test set of **10,000
records**.

| Model                     | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 | ROC-AUC | PR-AUC |
| ------------------------- | -------: | -----------------: | --------------: | ----------: | -------: | ------: | -----: |
| Logistic Regression       |   0.5339 |             0.5472 |          0.5339 |      0.5340 |   0.5350 |  0.6934 | 0.5401 |
| Artificial Neural Network |   0.5304 |             0.5470 |          0.5304 |      0.5301 |   0.5299 |  0.6922 | 0.5385 |
| XGBoost                   |   0.5295 |             0.5441 |          0.5295 |      0.5296 |   0.5306 |  0.6925 | 0.5405 |
| Random Forest             |   0.5188 |             0.5182 |          0.5188 |      0.5156 |   0.5267 |  0.6853 | 0.5301 |
| KNN                       |   0.4972 |             0.5106 |          0.4972 |      0.4978 |   0.4989 |  0.6596 | 0.5031 |

The best model on the unseen test set was **Logistic Regression**, with weighted
F1-score of **0.5340**. XGBoost and ANN were close, but did not outperform
Logistic Regression.

### 4.2 Best Model Classification Report

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------: | -----: | -------: | ------: |
| High             |      0.65 |   0.48 |     0.55 |   2,497 |
| Low              |      0.54 |   0.49 |     0.51 |   3,274 |
| Medium           |      0.49 |   0.60 |     0.54 |   4,229 |
| Accuracy         |           |        |     0.53 |  10,000 |
| Macro average    |      0.56 |   0.52 |     0.53 |  10,000 |
| Weighted average |      0.55 |   0.53 |     0.53 |  10,000 |

The model identifies the **High** risk class with relatively high precision but
limited recall. In practical terms, when the model predicts High risk, it is
often correct, but it misses a substantial number of actual High-risk students.

### 4.3 Error Analysis

| Metric                |  Value |
| --------------------- | -----: |
| Total test records    | 10,000 |
| Correct predictions   |  5,339 |
| Misclassified records |  4,661 |
| Error rate            | 0.4661 |

**Misclassification matrix for the best model**

| Actual \ Predicted | High |   Low | Medium |
| ------------------ | ---: | ----: | -----: |
| High               |    0 |   200 |  1,104 |
| Low                |   87 |     0 |  1,591 |
| Medium             |  543 | 1,136 |      0 |

This matrix contains only the misclassified records. The largest error pattern
is that actual **Low** students are often predicted as **Medium**, and actual
**High** students are often predicted as **Medium**.

**Class-level error rates**

| Actual Class | Total Records | Correct | Incorrect | Error Rate |
| ------------ | ------------: | ------: | --------: | ---------: |
| High         |         2,497 |   1,193 |     1,304 |     0.5222 |
| Low          |         3,274 |   1,596 |     1,678 |     0.5125 |
| Medium       |         4,229 |   2,550 |     1,679 |     0.3970 |

The model performs best on the Medium class and has the most concerning error
rate for the High class. Since High-risk recall is important in a wellbeing use
case, threshold tuning was necessary.

### 4.4 High-Risk Threshold Tuning

The probability threshold for the High-risk class was tuned to improve
intervention-oriented performance. The recommended threshold based on High-risk
F1 was **0.30**.

| Threshold | High-Risk Precision | High-Risk Recall | High-Risk F1 | Overall Weighted F1 |
| --------: | ------------------: | ---------------: | -----------: | ------------------: |
|      0.10 |              0.3294 |           0.9083 |       0.4834 |              0.3112 |
|      0.20 |              0.4404 |           0.7401 |       0.5522 |              0.4659 |
|      0.30 |              0.5461 |           0.6187 |       0.5802 |              0.5184 |
|      0.40 |              0.6207 |           0.5190 |       0.5653 |              0.5320 |
|      0.50 |              0.6544 |           0.4778 |       0.5523 |              0.5340 |

After threshold tuning at 0.30, High-risk recall improved from **0.48** to
**0.62**, while High-risk F1 improved from **0.55** to **0.58**. The trade-off
was a drop in overall accuracy and weighted F1. In a wellbeing intervention
setting, this trade-off may be acceptable because missing High-risk students is
more costly than flagging some additional false positives for human review.

### 4.5 Class-Imbalance Sensitivity Check

The notebook was extended with a course-aligned class-imbalance experiment based
on the ML 25 imbalance workflow. It compares the Logistic Regression baseline
with `class_weight='balanced'`, SMOTE, and SMOTETomek, using the existing
train/test split so that synthetic or resampled observations are created only
inside the training process. The comparison reports weighted F1, macro F1,
High-risk precision, High-risk recall, ROC-AUC, and PR-AUC.

This check is important because High-risk recall can be improved either at
training time, through balancing, or at decision time, through threshold tuning.
The practical deployment choice should favour the option that improves
High-risk recall without creating an unacceptable fall in High-risk precision or
overall weighted F1. Since the original tuning already selected
`class_weight=None` for Logistic Regression, the current final recommendation
remains the interpretable Logistic Regression model with High-risk threshold
tuning unless the SMOTE or SMOTETomek experiment produces a clearly better
recall/precision trade-off when the notebook is rerun.

### 4.6 Final Model Selection

The selected model is **Logistic Regression with L1 regularisation**. It was
selected because:

- It achieved the best weighted F1-score on the test set.
- It was also the best model in cross-validation.
- It provides class probabilities needed for threshold tuning.
- It is more interpretable than ANN and ensemble models.
- Its performance was close to more complex models, making the simpler model
  preferable.

## 5. Explainability, Fairness, and Responsible Use

### 5.1 Global Explainability

The final model was explained using permutation importance and SHAP. Because the
selected model was Logistic Regression, SHAP used the model-agnostic
KernelExplainer.

**Permutation importance**

| Feature                  | Importance Mean | Importance Std |
| ------------------------ | --------------: | -------------: |
| Weekly_GenAI_Hours       |        0.127696 |       0.003359 |
| Year_of_Study            |        0.028599 |       0.003603 |
| Pre_Semester_GPA         |        0.009689 |       0.003045 |
| Perceived_AI_Dependency  |        0.008043 |       0.001404 |
| Dependency_Anxiety_Index |        0.005337 |       0.002208 |
| Institutional_Policy     |        0.003047 |       0.001066 |
| Traditional_Study_Hours  |        0.001945 |       0.002370 |
| Post_Semester_GPA        |        0.001443 |       0.000923 |
| Primary_Use_Case         |        0.001018 |       0.000435 |
| Skill_Retention_Score    |        0.000679 |       0.000400 |

**SHAP importance**

| Feature                         | Mean Absolute SHAP |
| ------------------------------- | -----------------: |
| Weekly_GenAI_Hours              |           0.094078 |
| Year_of_Study_Freshman          |           0.019517 |
| Year_of_Study_Graduate          |           0.019386 |
| Pre_Semester_GPA                |           0.016141 |
| Perceived_AI_Dependency         |           0.015450 |
| Traditional_Study_Hours         |           0.009604 |
| Institutional_Policy_Strict_Ban |           0.008945 |
| Dependency_Anxiety_Index        |           0.008606 |
| Year_of_Study_Senior            |           0.005052 |
| Year_of_Study_Sophomore         |           0.004159 |

The explanation results are consistent with the feature-selection stage:
**Weekly_GenAI_Hours** is the dominant predictor, followed by year of study,
previous GPA, perceived AI dependency, traditional study hours, institutional
policy, and the dependency-anxiety interaction.

Because the selected model is Logistic Regression, the notebook also
generates a native coefficient and odds-ratio table for the High-risk class.
Features with positive coefficients have odds ratios above 1 and increase the
odds of a High-risk prediction, while features with negative coefficients have
odds ratios below 1 and reduce those odds after preprocessing. This coefficient
view complements SHAP and permutation importance: SHAP explains prediction
contributions model-agnostically, while odds ratios provide a direct statistical
interpretation of the final Logistic Regression model.

### 5.2 Fairness and Subgroup Analysis

Accuracy, error rate, and High-risk recall were analysed across key student
groups.

| Fairness Check | Main Finding | Implication |
| --- | --- | --- |
| Error rate by major | Arts had the highest error rate, while STEM had the lowest. | Performance should be audited across academic disciplines. |
| High-risk recall by major | Humanities had lower High-risk recall than STEM. | Some groups may be more likely to be missed. |
| High-risk recall by paid subscription | Students without paid subscriptions had lower High-risk recall. | Access to paid tools may create an equity-related performance difference. |
| High-risk recall by policy context | Recall differed across institutional AI-policy categories. | Deployment should monitor policy-related subgroup performance. |

These subgroup differences do not prove unfair treatment, but they show that the
model should be audited before deployment and monitored continuously after
deployment.

### 5.3 Privacy, Ethics, and Responsible Use

Burnout risk prediction is sensitive because it combines academic behaviour,
wellbeing indicators, and technology usage. Responsible deployment should follow
these principles:

- Use the model only as a decision-support tool, not as an automated
  disciplinary system.
- Inform students how data will be used and what support actions may follow.
- Avoid using predictions to penalise students for AI use.
- Prioritise supportive interventions such as academic advising, AI-literacy
  training, study-skills support, or wellbeing check-ins.
- Monitor model performance across subgroups to reduce unequal false negatives.
- Minimise sensitive data collection and protect student privacy.
- Revalidate the model on real institutional data before practical use.

### 5.4 Leakage and Deployment Caveat

The final analytical modelling pipeline used all available engineered
predictors, including `Post_Semester_GPA`, `Skill_Retention_Score`,
`GPA_Change`, and `Skill_Efficiency`. These are useful for retrospective
analysis, but some of them may not be available early enough for a live
early-warning system. A production early-warning version should exclude
post-semester or outcome-like variables and train only on features available
before intervention decisions are made.

Therefore, the reported model is best interpreted as an analytical benchmark. A
deployment-ready version should be rebuilt using a leakage-aware early-warning
feature set.

## 6. Required Visual Evidence

The PDF requires appropriate visualisations for EDA, model comparison, and explainability. The most relevant figures are summarised below without repeating every generated plot.

### 6.1 Exploratory Visualisations

![Figure 1. Distribution of burnout risk level.](question1_report_assets/target_distribution.png)

_Figure 1. Distribution of burnout risk level._

The target distribution confirms a moderate class imbalance: Medium is the largest class, followed by Low and High. This supports using weighted F1, macro F1, PR-AUC, and class-level recall rather than accuracy alone.

![Figure 2. Weekly GenAI hours by burnout risk level.](question1_report_assets/weekly_genai_hours_by_burnout.png)

_Figure 2. Weekly GenAI hours by burnout risk level._

The High-risk group shows substantially higher weekly GenAI use than the Low and Medium groups, supporting the feature-engineering focus on AI intensity and AI-to-study balance.

![Figure 3. Correlation heatmap of numerical variables.](question1_report_assets/correlation_heatmap.png)

_Figure 3. Correlation heatmap of numerical variables._

The heatmap shows strong relationships between related academic and AI-use variables, including the expected persistence between pre- and post-semester GPA and the relationship between GenAI hours and perceived dependency.

### 6.2 Feature Selection, Model Comparison, and Explainability

![Figure 4. Top features by mutual information.](question1_report_assets/mutual_information_top_features.png)

_Figure 4. Top features by mutual information._

Mutual information ranked `Weekly_GenAI_Hours`, `AI_to_Study_Ratio`, `High_AI_Use_Flag`, and `Perceived_AI_Dependency` among the strongest predictors.

![Figure 5. Model comparison by weighted F1-score.](question1_report_assets/test_weighted_f1_model_comparison.png)

_Figure 5. Model comparison by weighted F1-score._

Logistic Regression, ANN, and XGBoost performed similarly, but Logistic Regression was selected because it achieved the best weighted F1 while remaining simpler and more interpretable.

![Figure 6. Threshold tuning for High-risk burnout detection.](question1_report_assets/high_risk_threshold_tuning.png)

_Figure 6. Threshold tuning for High-risk burnout detection._

Threshold tuning improved High-risk recall, which is important because the model is intended for supportive intervention rather than fully automated decision-making.

![Figure 7. Permutation importance for the final model.](question1_report_assets/permutation_importance.png)

_Figure 7. Permutation importance for the final model._

Permutation importance confirms that `Weekly_GenAI_Hours` is the strongest individual feature for the final model.

![Figure 8. High-risk recall by paid subscription.](question1_report_assets/fairness_high_risk_recall_by_paid_subscription.png)

_Figure 8. High-risk recall by paid subscription._

The fairness visual shows a subgroup recall difference by paid subscription status, so deployment should include subgroup monitoring and periodic fairness review.

## 7. Cloud Deployment and MLOps Strategy

### 7.1 Deployment Objective

The model is best suited for a controlled student-support workflow rather than a
fully automated decision system. The deployment goal is to provide risk scores
that help authorised wellbeing or academic-support staff prioritise review,
outreach, and AI-literacy guidance.

The deployed system should support two prediction modes:

| Inference Mode      | Use Case                                                                           | Expected Output                                                                            |
| ------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Batch inference     | Weekly or monthly scoring of enrolled students for student-support dashboards      | Student ID, predicted burnout-risk class, class probabilities, key risk drivers, timestamp |
| Real-time inference | Advisor-facing check during a support session or after a student survey submission | Predicted risk class, probability scores, explanation summary                              |

### 7.2 Proposed GCP Cloud Architecture

A Google Cloud Platform (GCP) architecture is appropriate because the selected
Logistic Regression pipeline is lightweight, interpretable, and can be served
efficiently through a containerised API. The deployment should use managed GCP
services so that the university team can focus on governance, monitoring, and
safe use rather than server administration.

The proposed architecture is:

```text
Student survey / MIS data
        |
        v
Cloud Storage raw-data bucket --> Vertex AI Pipeline / Cloud Run Job
        |                                  |
        v                                  v
Cloud Storage processed-data bucket   trained model artefact
        |                                  |
        v                                  v
BigQuery monitoring tables       Vertex AI Model Registry
                                           |
                                           v
                                   Docker image in Artifact Registry
                                           |
                                           v
                          Cloud Run FastAPI prediction service
                                           |
                                           v
                        Advisor dashboard / support workflow
```

| Layer | GCP service | Role |
| --- | --- | --- |
| Raw and processed data storage | Cloud Storage | Store raw survey extracts, processed feature tables, model artefacts, and batch-scoring files using versioned bucket paths. |
| Analytics and monitoring store | BigQuery | Store prediction logs, drift statistics, evaluation results, and dashboard-ready monitoring tables. |
| Feature processing and batch jobs | Vertex AI Pipelines or Cloud Run Jobs | Run data validation, preprocessing, feature engineering, retraining, and scheduled batch inference. |
| Experiment tracking | Vertex AI Experiments or MLflow backed by Cloud Storage/BigQuery | Track parameters, datasets, metrics, artefacts, and model-selection decisions. |
| Model registry | Vertex AI Model Registry | Store approved model versions, metadata, aliases, evaluation results, and lifecycle status. |
| Container registry | Artifact Registry | Store Docker images for the FastAPI inference service and batch-scoring jobs. |
| Real-time model serving | Cloud Run | Serve a Dockerised FastAPI application with autoscaling, HTTPS, revision management, and controlled access. |
| Batch inference | Cloud Scheduler + Cloud Run Jobs, or Vertex AI batch pipeline | Score weekly/monthly student records and write results to Cloud Storage or BigQuery. |
| Monitoring and logging | Cloud Logging, Cloud Monitoring, Error Reporting, BigQuery dashboards | Track API health, errors, latency, prediction distributions, drift, and performance when labels arrive. |
| Security and governance | IAM, service accounts, Secret Manager, Cloud KMS, VPC controls, audit logs | Enforce least-privilege access, encryption, secret management, network protection, and traceability. |

The architecture should keep personally identifiable student data encrypted at
rest and in transit. Access to predictions should be restricted to authorised
student-support staff through IAM and application-level role controls. Prediction
requests, prediction outputs, model versions, and human follow-up actions should
be logged for auditability.

### 7.3 Model Packaging, Containerisation, and API Design

The final Logistic Regression pipeline should be packaged as a versioned
artefact containing:

- The preprocessing pipeline.
- The trained Logistic Regression model.
- The label encoder.
- The selected High-risk threshold.
- Feature schema and validation rules.
- Model metadata, including training date, dataset version, metrics, and known
  limitations.

The artefact should be saved as a single reproducible package, for example:

```text
model_package/
  model.joblib
  feature_schema.json
  label_mapping.json
  threshold.json
  metrics.json
  model_card.md
```

The model can then be served through a Dockerised FastAPI application. Docker
improves portability because the same application image can run locally, in
testing, and on Cloud Run with the same Python version and dependencies. It also
improves scalability because Cloud Run can start multiple identical container
instances when request volume increases.

A minimal container structure would be:

```text
deployment/
  app.py
  requirements.txt
  Dockerfile
  model_package/
```

A typical API design would include:

| Endpoint              | Purpose                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `GET /health`         | Confirms the service is running.                                                                                    |
| `GET /model-info`     | Returns model version, training date, and approved metric summary.                                                  |
| `POST /predict`       | Accepts one student record and returns class probabilities, predicted class, and threshold-adjusted High-risk flag. |
| `POST /batch-predict` | Accepts a batch of records or a Cloud Storage file reference for scheduled scoring.                                 |

The API response should include probability scores rather than only a hard class
label. This allows advisors to interpret uncertain cases more carefully and
supports threshold tuning for High-risk intervention.

Real-time inference is appropriate when an advisor submits one student record
during a support session. Batch inference is appropriate for weekly or monthly
scoring of many students, where the output is written to BigQuery or Cloud
Storage for review in a dashboard.

### 7.4 MLOps Pipeline

A production workflow should separate experimentation, validation, approval,
deployment, and monitoring. This follows the ML 25 MLOps maturity idea:

- Level 0: the current notebook/report workflow.
- Level 2 target: automated training, testing, packaging, and deployment.
- Level 3 target: governed model promotion, drift monitoring, rollback, audit
  logs, and retraining triggers.

| MLOps stage | GCP implementation |
| --- | --- |
| Data versioning | Store raw and processed datasets in Cloud Storage using immutable date/version paths such as `raw/2026-07-12/` and `processed/v1/`. For stricter reproducibility, use DVC with a GCS remote. |
| Code versioning | Manage notebooks, training scripts, API code, Dockerfile, and pipeline definitions in Git. |
| Experiment tracking | Log model type, hyperparameters, cross-validation scores, test metrics, feature list, dataset version, threshold, and artefacts in Vertex AI Experiments or MLflow. |
| Model versioning | Register candidate and approved models in Vertex AI Model Registry with status labels such as `candidate`, `staging`, `approved`, and `production`. |
| Automated testing | Test feature schema, missing-value handling, invalid categories, class-label mapping, prediction reproducibility, API response format, and fairness metric calculation. |
| Model validation | Require minimum weighted F1, High-risk recall, ROC-AUC, PR-AUC, calibration checks, and subgroup fairness checks before approval. |
| CI/CD | Use Cloud Build or GitHub Actions to run tests, build the Docker image, push it to Artifact Registry, and deploy the approved image to a Cloud Run revision. |
| Deployment control | Use staged deployment or traffic splitting in Cloud Run, then promote the new revision only after health checks and monitoring pass. |
| Rollback | Keep the previous Cloud Run revision and previous model version available for immediate rollback. |

The pipeline should not automatically deploy a model only because a metric
improves slightly. Because burnout prediction is sensitive, deployment approval
should include a human review of performance, fairness, privacy, and leakage
risks.

The automated pipeline should run in three cases:

1. A planned retraining cycle at the end of each academic term.
2. A monitored drift or performance alert.
3. A manually approved model improvement from the data-science team.

### 7.5 Monitoring and Maintenance

Model monitoring should cover technical health, data quality, prediction
behaviour, and real-world effectiveness.

| Monitoring area | Example checks | GCP implementation |
| --- | --- | --- |
| Service health | API latency, error rate, container starts, failed batch jobs. | Cloud Run metrics, Cloud Monitoring dashboards, Cloud Logging, Error Reporting. |
| Data quality | Missing values, invalid categories, out-of-range GPA or hour values, schema mismatch. | Validation job writes failures to BigQuery and triggers Cloud Monitoring alerts. |
| Data drift | Changes in `Weekly_GenAI_Hours`, `Perceived_AI_Dependency`, `Traditional_Study_Hours`, and categorical distributions. | Scheduled drift job compares current input distributions with the training baseline and stores PSI/KS-style statistics. |
| Prediction drift | Changes in Low/Medium/High prediction proportions and High-risk probability distribution. | Prediction logs are aggregated in BigQuery and visualised through a monitoring dashboard. |
| Concept drift | Relationship between features and burnout outcome changes over time. | Recalculate labelled performance once later wellbeing outcomes become available. |
| Performance monitoring | Weighted F1, macro F1, High-risk recall, High-risk precision, ROC-AUC, and PR-AUC on newly labelled data. | Evaluation pipeline compares new metrics with production approval thresholds. |
| Fairness monitoring | High-risk recall and false-negative rates by major, year of study, paid subscription, policy context, and prompt skill. | Scheduled subgroup reports in BigQuery with alerts for large degradation. |
| Human feedback | Advisor feedback on whether flagged students were appropriate for review. | Feedback form or dashboard table linked to prediction ID and model version. |

Retraining should be scheduled at least once per academic term, or sooner if
drift is detected. Retraining should use the most recent labelled data, but the
early-warning deployment model should exclude post-semester or outcome-like
features that are not available before intervention. A retrained model should
move through candidate, staging, approval, and production states rather than
being pushed directly into use.

This deployment strategy is appropriate because the selected model is
interpretable, relatively lightweight, easy to package, and fast enough for both
batch and real-time scoring. Cloud Run is suitable because the API workload is
request-based and does not require continuously managed servers. Cloud Storage
and BigQuery separate data storage from serving, Artifact Registry makes the
container image reproducible, and Vertex AI Model Registry provides model
lifecycle governance. Cloud Monitoring and Cloud Logging support operational
visibility, while IAM, Secret Manager, and encryption controls support privacy
and security. Because the outcome is wellbeing-related, predictions should be
used only for supportive outreach, with human review, audit logs, encryption,
fairness monitoring, and rollback before any production use.

The main limitation is that the current analytical model uses some post-semester
variables; therefore, the production version should be retrained on a
leakage-aware feature set before real use.

## 8. Conclusions and Recommendations

The analysis shows that burnout risk is most strongly associated with GenAI
usage intensity, AI-to-study balance, perceived dependency, anxiety, study
workload, and academic indicators. The High-risk group had substantially higher
weekly GenAI use than the Low-risk group, with lower traditional study time and
higher exam anxiety.

After comparing five models, **Logistic Regression** was selected as the final
model. It achieved **0.5339 accuracy**, **0.5340 weighted F1-score**, **0.6934
ROC-AUC**, and **0.5401 PR-AUC** on the unseen test set. Although the overall
performance is moderate, the model provides interpretable predictions and
probability outputs. Threshold tuning improved High-risk recall from **0.48** to
**0.62**, making the model more suitable for a student-support context where
missing High-risk students is costly.

The final recommendation is to treat the model as a transparent,
human-in-the-loop decision-support system. It can help prioritise student
wellbeing interventions, but it should not replace human judgement or be used
for punitive decisions. Before real deployment, the model should be retrained on
leakage-aware early-warning features and audited for fairness across student
groups.

## References

- Google Cloud. (n.d.).
  [Cloud Run documentation](https://docs.cloud.google.com/run/docs).
- Google Cloud. (n.d.).
  [Cloud Storage documentation](https://docs.cloud.google.com/storage/docs).
- Google Cloud. (n.d.).
  [Artifact Registry documentation](https://docs.cloud.google.com/artifact-registry/docs).
- Google Cloud. (n.d.).
  [Cloud Build documentation](https://docs.cloud.google.com/build/docs).
- Google Cloud. (n.d.).
  [Cloud Monitoring documentation](https://docs.cloud.google.com/monitoring/docs).
- Google Cloud. (n.d.).
  [Vertex AI Model Registry introduction](https://docs.cloud.google.com/vertex-ai/docs/model-registry/introduction).
- Google Cloud. (n.d.).
  [Secret Manager documentation](https://docs.cloud.google.com/secret-manager/docs).
- Jadon, L. (n.d.).
  [Impact of AI on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students).
  Kaggle.
- Dong, L., Tang, X., and Wang, X. (2025).
  [Examining the effect of artificial intelligence in relation to students' academic achievement in classroom: A meta-analysis](https://www.sciencedirect.com/science/article/pii/S2666920X25000402).
  _Computers and Education: Artificial Intelligence_, 8, 100400.
- Freeman, J. (2025).
  [Student Generative AI Survey 2025](https://eric.ed.gov/?id=ED671617). Higher
  Education Policy Institute.
- Higher Education Policy Institute. (2026).
  [Student Generative Artificial Intelligence Survey 2026](https://www.hepi.ac.uk/reports/student-generative-ai-survey-2026/).
- Schwartz, H. L., and Diliberti, M. K. (2026).
  [More Students Use AI for Homework, and More Believe It Harms Critical Thinking](https://www.rand.org/pubs/research_reports/RRA4742-1.html).
  RAND Corporation.
- UNESCO. (2023).
  [Guidance for generative AI in education and research](https://www.unesco.org/en/articles/guidance-generative-ai-education-and-research).
- Klimova, B., and Pikhart, M. (2025).
  [Exploring the effects of artificial intelligence on student and academic well-being in higher education: a mini-review](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1498132/full).
  _Frontiers in Psychology_.
- Francis, N. J., Jones, S., and Smith, D. P. (2025).
  [Generative AI in Higher Education: Balancing Innovation and Integrity](https://www.frontierspartnerships.org/journals/british-journal-of-biomedical-science/articles/10.3389/bjbs.2024.14048/full).
  _British Journal of Biomedical Science_.
- Stanford SCALE Initiative. (2026).
  [Understanding the Evidence Base on AI in K-12 Education](https://scale.stanford.edu/research-in-action/understanding-evidence-base-ai-k12-education).
- López-Meneses, E., et al. (2025).
  [Educational Data Mining and Predictive Modeling in the Age of Artificial Intelligence](https://www.mdpi.com/2073-431X/14/2/68).
  _Computers_.
- Gu, C., He, B., and Zhang, X. (2026).
  [The relationship between college students' AI technology dependence and learning burnout: a chain mediation analysis of technology acceptance and AI self-efficacy](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2026.1843366/full).
  _Frontiers in Psychology_.
- Yağcı, M. (2022).
  [Educational data mining: prediction of students' academic performance using machine learning algorithms](https://link.springer.com/article/10.1186/s40561-022-00192-z).
  _Smart Learning Environments_, 9, Article 11.
