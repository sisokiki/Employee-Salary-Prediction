#Employee Salary Prediction.
This repository contains a machine learning model developed to predict employee salaries based on various demographic and employment-related features. The project utilizes a dataset containing information such as age, workclass, education, marital status, occupation, and hours worked per week to classify income into two categories: <=50K and >50K.

Table of Contents
Project Overview

Dataset

Features

Methodology

Model Performance

Key Findings (Feature Importance)

How to Run the Code

Files in this Repository

Future Enhancements

References

License

Project Overview
The goal of this project is to build a robust and accurate machine learning model that can predict an individual's income bracket. This model can be valuable for:

Fair salary benchmarking.

Understanding factors influencing income.

Assisting in resource allocation and policy making.

Dataset
The model is trained on the adult 3.csv dataset, which is a modified version of the UCI Adult Income dataset. It contains various demographic and employment-related attributes.

Features
The dataset includes the following features:

age: Age of the individual.

workclass: Type of employer (e.g., Private, Self-emp-not-inc, Local-gov).

fnlwgt: Final weight (statistical weight).

education: Highest level of education achieved.

educational-num: Numerical representation of education level.

marital-status: Marital status.

occupation: Type of occupation.

relationship: Relationship status.

race: Individual's race.

gender: Gender of the individual.

capital-gain: Capital gains.

capital-loss: Capital losses.

hours-per-week: Number of hours worked per week.

native-country: Country of origin.

income: Target variable, indicating income <=50K or >50K.

Methodology
The project follows a standard machine learning pipeline:

Data Loading & Initial Inspection: Loading the CSV and examining its structure.

Data Cleaning: Handling missing values (represented as '?') by replacing them with NaN and then imputing them with the mode for categorical columns.

Target Variable Encoding: Converting the 'income' column into numerical labels (0 for <=50K, 1 for >50K).

Feature Preprocessing:

One-Hot Encoding: Transforming categorical features into numerical format.

Feature Scaling: Standardizing numerical features using StandardScaler.

Data Splitting: Dividing the dataset into training (80%) and testing (20%) sets, ensuring stratified sampling for income classes.

Model Selection: Utilizing a RandomForestClassifier, an ensemble learning method known for its high accuracy and robustness.

Hyperparameter Tuning: Employing GridSearchCV to find the optimal set of hyperparameters for the RandomForestClassifier, maximizing model performance.

Model Evaluation: Assessing model performance using:

Accuracy Score

Classification Report (Precision, Recall, F1-score)

Confusion Matrix (with visualization)

ROC Curve and Area Under the Curve (AUC)

Feature Importance Analysis: Identifying the most influential features in predicting income.

Model Persistence: Saving the trained model and the scaler using joblib for future use and deployment.

Example Prediction: Demonstrating how to use the saved model to predict income for new, unseen data.

Model Performance
Based on the evaluation of the tuned model on the test set:

Tuned Model Accuracy: 0.8653

Area Under the Curve (AUC): 0.9181

(For detailed Classification Report and Confusion Matrix, please refer to the Employees_Salary_Prediction.ipynb notebook.)

Key Findings (Feature Importance)
The feature importance analysis revealed that:

marital-status_Married-civ-spouse

capital-gain

age

educational-num

hours-per-week

These features were found to be the most influential factors in determining an individual's income bracket, significantly contributing to the model's predictive power.

How to Run the Code
Clone the Repository:

git clone https://github.com/sisokiki/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction

Open in Google Colab:

Upload Employees_Salary_Prediction.ipynb to your Google Drive.

Open it with Google Colaboratory.

Ensure adult 3.csv is also uploaded to your Colab environment (e.g., in /content/).

Install Libraries: All necessary libraries (pandas, numpy, scikit-learn, matplotlib, seaborn, joblib) are typically pre-installed in Colab. If not, you can install them using !pip install <library_name>.

Run Cells: Execute all cells sequentially in the Jupyter Notebook.

Files in this Repository
Employees_Salary_Prediction.ipynb: The main Jupyter Notebook containing all the code for data preprocessing, model training, evaluation, and analysis.

employees_salary_prediction.py: A Python script version of the model code.

adult 3.csv: The dataset used for training and testing the model.

best_salary_predictor_model.joblib: (Generated after running the notebook) The saved, best-performing machine learning model.

scaler.joblib: (Generated after running the notebook) The saved StandardScaler object used for preprocessing numerical features.

Future Enhancements
Explore other advanced models: Investigate Gradient Boosting Machines (XGBoost, LightGBM) or neural networks.

More sophisticated feature engineering: Create complex interaction features or explore advanced transformations.

Handle class imbalance: Implement techniques like SMOTE if the dataset exhibits significant class imbalance.

Deployment as a web application: Create a simple web interface for interactive predictions.

Interpretability beyond feature importance: Utilize tools like SHAP or LIME for deeper insights into individual predictions.

References
Pandas: McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51–56. https://pandas.pydata.org/docs/

Scikit-learn: Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830. https://scikit-learn.org/stable/documentation.html

Matplotlib: Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95. https://matplotlib.org/stable/contents.html

Seaborn: Waskom, M. L. (2021). Seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021. https://seaborn.pydata.org/

General ML Concepts: Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (2nd ed.). O'Reilly Media.

License
This project is open-sourced under the MIT License. See the LICENSE file for more details. 
