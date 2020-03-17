# Machine learning-based automated classification of headache disorders using patient-reported questionnaires

Authors: Junmo Kwon, Hyebin Lee, Soohyun Cho, Chin-Sang Chung, Mi Ji Lee, and Hyunjin Park

### Test Environment
- Operating System: Windows 10 Pro 10.0.17134 Build 17134
- Processor: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
- Python 3.7.1

|Package     |Version|
|------------|-------|
|Numpy       |1.15.4 |
|Pandas      |0.25.1 |
|Scikit-learn|0.20.0 |
|XGBoost     |0.82   |

### Instructions
1. Fill out `demographics_template.csv` for both training and test set and name them as `train.csv` and `test.csv`.
2. Perform 10-fold cross validation on the training set and name each training and validation fold as `train_#.csv` and `valid_#.csv` where # is an integer between 1 and 10.
3. Run `lasso.R` to obtain feature appearances for each fold and each model.
4. Run `classification.ipynb` and follow the steps.
