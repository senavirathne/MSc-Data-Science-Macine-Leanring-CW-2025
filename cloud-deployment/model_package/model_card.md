# Student Burnout Risk Classifier

## Model
Logistic Regression multiclass classification model.

## Version
v1

## Target
Burnout_Risk_Level

## Classes
High, Medium, Low

## Intended Use
Decision-support tool for authorised student-support and wellbeing staff.

## Metrics
Accuracy: 0.5339  
ROC-AUC: 0.6934  
PR-AUC: 0.5401

## Limitations
The model is a prototype and is not suitable for automated decision-making.
Some analytical features may create post-semester leakage and should be removed
before real early-warning deployment.

## Ethical Considerations
Predictions require human review. Student identifiers and sensitive features
must not be written to application logs.

## Retraining
Retrain at least once per academic term or when drift, performance decline, or
fairness degradation is detected.
