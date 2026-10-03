# 🎓 EduGuard — Student Dropout Risk Analytics

EduGuard is a **Streamlit-based machine learning application** that demonstrates how data thinking and machine learning can be used to estimate a student's **dropout risk score**.

The project uses a **Random Forest Regression model** to predict a continuous risk score based on academic, attendance, socioeconomic, and behavioral factors.

> **Important:** This project uses **synthetic data** generated programmatically for educational and demonstration purposes. It does not use real student records.

---

## 📌 Project Overview

Student dropout can be influenced by multiple factors, including:

* Attendance
* Academic performance
* Study habits
* Family income
* Distance from school
* Previous academic failures
* Disciplinary incidents
* Parental education
* Internet access
* Extracurricular participation

EduGuard combines these factors to generate a **continuous Risk Score** and trains a machine learning regression model to predict that score.

### Machine Learning Approach

```text
Synthetic Student Data
        ↓
Data Preparation
        ↓
Feature Engineering
        ↓
Risk Score Generation
        ↓
Train/Test Split
        ↓
Standardization
        ↓
Random Forest Regression
        ↓
Risk Score Prediction
        ↓
Regression Evaluation
```

---

# 🎯 Objectives

The main objectives of this project are to:

1. Demonstrate a **Data Thinking approach** to an educational problem.
2. Generate and explore synthetic student data.
3. Identify factors associated with dropout risk.
4. Create a continuous student risk score.
5. Train a **Random Forest Regression** model.
6. Evaluate the model using regression metrics.
7. Build an interactive Streamlit dashboard.
8. Demonstrate responsible use of machine learning in education.

---

# 📊 Dataset

The project generates **2,500 synthetic student records** using NumPy and Pandas.

The data is not collected from real students.

## Features

| Feature                       | Description                                 | Type    |
| ----------------------------- | ------------------------------------------- | ------- |
| Age                           | Student age                                 | Numeric |
| Attendance_Percentage         | Percentage of attendance                    | Numeric |
| Average_Grade                 | Average academic grade                      | Numeric |
| Study_Hours_Per_Week          | Weekly study hours                          | Numeric |
| Family_Income                 | Estimated family income                     | Numeric |
| Distance_From_School_KM       | Distance from school                        | Numeric |
| Previous_Failures             | Number of previous academic failures        | Numeric |
| Disciplinary_Incidents        | Number of disciplinary incidents            | Numeric |
| Parental_Education_Years      | Years of parental education                 | Numeric |
| Internet_Access               | Internet availability                       | Binary  |
| Extracurricular_Participation | Participation in extracurricular activities | Binary  |

---

# 🎯 Target Variable

The machine learning target is:

```text
Risk_Score
```

Unlike a classification problem, the target is a **continuous numerical value**.

For example:

```text
-1.25
0.42
1.37
2.81
4.16
```

The model therefore performs **regression**, rather than classification.

---

# 🤖 Machine Learning Model

The project uses:

```text
RandomForestRegressor
```

from Scikit-learn.

### Model Configuration

```python
RandomForestRegressor(
    n_estimators=250,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The numerical features are standardized using:

```text
StandardScaler
```

---

# 📈 Regression Evaluation

The application evaluates the regression model using the following metrics.

## MAE — Mean Absolute Error

Measures the average absolute difference between the actual and predicted risk scores.

```text
Lower MAE = smaller average prediction error
```

---

## MSE — Mean Squared Error

Measures the average squared prediction error.

Large errors have a greater effect on MSE.

```text
Lower MSE = smaller prediction errors
```

---

## RMSE — Root Mean Squared Error

RMSE is the square root of MSE.

It is expressed in the same units as the target variable.

```text
RMSE = √MSE
```

---

## R² Score

R² measures how much variation in the target variable is explained by the regression model.

A value closer to 1 indicates that the model explains more of the variation in the target data.

---

## MAPE — Mean Absolute Percentage Error

MAPE expresses prediction error as a percentage.

The application includes a small denominator safeguard because percentage error can become unstable when actual values are close to zero.

---

# 📊 Visualizations

The application provides several visualizations.

### Risk Score Distribution

Shows the distribution of the generated student risk scores.

### Feature Importance

Shows which input variables contribute most to the Random Forest model's predictions.

### Actual vs Predicted

Compares:

```text
Actual Risk Score
        vs
Predicted Risk Score
```

A reference diagonal line is included to help visualize prediction accuracy.

### Residual Analysis

Shows the difference between actual and predicted values.

```text
Residual = Actual Value - Predicted Value
```

### Prediction Error Distribution

Displays the distribution of the model's prediction errors.

---

# 🖥️ Application Pages

The Streamlit application contains five main sections.

## 🏠 Dashboard

Provides an overview of:

* Number of students
* Average risk score
* RMSE
* R² score
* Risk score distribution
* Feature importance

---

## 🔮 Student Prediction

Allows the user to enter an individual student's information.

The application then generates a predicted continuous risk score.

Example:

```text
Predicted Risk Score: 2.37
```

The interface also provides a simple visual interpretation of the score.

---

## 📊 Data Analysis

Allows users to explore the synthetic dataset.

Available analyses include:

* Feature distributions
* Box plots
* Feature vs. risk score relationships
* Dataset preview
* Descriptive statistics

---

## 📈 Regression Evaluation

Displays:

* MAE
* MSE
* RMSE
* R²
* MAPE
* Actual vs. Predicted plot
* Residual plot
* Prediction error distribution
* Feature importance

---

## ℹ️ About Project

Explains:

* Project objective
* Data Thinking framework
* Machine learning approach
* Dataset features
* Regression target
* Responsible use

---

# 🧠 Data Thinking Framework

The project follows a simple Data Thinking workflow.

### 1. Define

Define student dropout risk as the educational problem.

### 2. Collect

Generate relevant synthetic academic, attendance, socioeconomic, and behavioral data.

### 3. Prepare

Prepare and standardize the data for machine learning.

### 4. Analyze

Explore relationships between student characteristics and risk score.

### 5. Model

Train a Random Forest Regression model.

### 6. Evaluate

Evaluate the model using:

* MAE
* MSE
* RMSE
* R²
* MAPE

### 7. Act

Use the predicted score as a potential signal for appropriate human support.

---

# 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd EduGuard
```

---

## 2. Install dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

Alternatively, create a `requirements.txt` file:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Then install:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Run the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📁 Project Structure

```text
EduGuard/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🔬 Example Workflow

A user can enter:

```text
Age: 16
Attendance: 72%
Average Grade: 58
Study Hours: 7
Family Income: 35000
Distance: 8 KM
Previous Failures: 2
Disciplinary Incidents: 1
Parental Education: 10 years
Internet Access: Yes
Extracurricular: No
```

The Random Forest Regressor processes these features and returns a continuous risk score.

```text
Predicted Risk Score
        ↓
     2.XX
```

The exact prediction depends on the trained model and generated data.

---

# ⚠️ Limitations

This project has several important limitations.

### Synthetic Data

The dataset is artificially generated and does not represent real student populations.

### Synthetic Target

The `Risk_Score` is generated using a predefined mathematical formula. It is therefore not a real-world measurement of dropout probability.

### Demonstration Model

The model is intended to demonstrate machine learning and data-thinking concepts rather than provide a validated educational intervention system.

### Generalization

Performance on synthetic data does not indicate how the model would perform on real student data.

---

# 🔐 Responsible Use

Educational risk prediction can have significant consequences for students.

This project should **not** be used to:

* Automatically label students
* Punish students
* Deny educational opportunities
* Exclude students from programs
* Make disciplinary decisions
* Replace teachers, counselors, or administrators

Any real-world implementation would require appropriate:

* Privacy protections
* Data governance
* Model validation
* Fairness testing
* Human oversight
* Institutional review

The model's output should be treated as a **supporting signal rather than a definitive statement about a student's future**.

---

# 🚀 Future Improvements

Possible future improvements include:

* Replace synthetic data with an appropriate real-world dataset.
* Add cross-validation.
* Perform hyperparameter tuning.
* Add additional regression models.
* Compare Random Forest with Linear Regression and Gradient Boosting.
* Add feature correlation analysis.
* Add model explainability using SHAP.
* Add fairness evaluation.
* Add model monitoring.
* Allow CSV dataset uploads.
* Add downloadable prediction reports.

---

# 👨‍💻 Project Purpose

EduGuard was developed as an educational demonstration of how:

```text
Data Thinking
      +
Data Analysis
      +
Machine Learning
      +
Visualization
      +
Responsible AI
```

can be combined to investigate an educational problem.

---

## 📄 License

This project is intended for educational and demonstration purposes.

If publishing this project publicly, add the license appropriate for your repository, such as MIT License.
