<div align="center">
  <h1>💖 HEART DISEASE PREDICTION</h1>
  <p><i>Machine Learning for Early and Accurate Heart Disease Diagnosis</i></p>
</div>

<br>

<div align="center">
  <a href="https://github.com/brej-29/heart-disease-prediction-ml">
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/brej-29/heart-disease-prediction-ml">
  </a>
  <img alt="Jupyter Notebook" src="https://img.shields.io/badge/Notebook-Jupyter-orange">
  <img alt="Python Language" src="https://img.shields.io/badge/Language-Python-blue">
</div>

<div align="center">
  <br>
  <b>Built with the tools and technologies:</b>
  <br>
  <br>
  <code>Python</code> | <code>NumPy</code> | <code>Pandas</code> | <code>Matplotlib</code> | <code>Seaborn</code> | <code>Scikit-Learn</code> | <code>Jupyter Notebook</code>
</div>

---

## **Table of Contents**
* [Overview](#overview)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Usage](#usage)
* [Data Exploration & Preprocessing](#data-exploration--preprocessing)
* [Modeling & Evaluation](#modeling--evaluation)
* [Model Deployment](#model-deployment)
* [License](#license)
* [Contact](#contact)

---

## **Overview**

This project leverages machine learning to predict the likelihood of heart disease in patients using a dataset of medical attributes (age, sex, blood pressure, cholesterol, etc.). The notebook demonstrates an end-to-end workflow including:
* Data cleaning and preprocessing
* Exploratory data analysis with visualizations
* Feature engineering and correlation analysis
* Model selection and training (Logistic Regression, KNN, Random Forest)
* Hyperparameter tuning with RandomizedSearchCV and GridSearchCV
* Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
* Feature importance analysis
* Model persistence for deployment

<br>

### **Exploratory Data Analysis Highlights**

- **Target Distribution:** The dataset is balanced between patients with and without heart disease.
- **Demographics:** Higher prevalence of heart disease among males; most patients are aged 50-60.
- **Correlation Analysis:** Features like chest pain type (cp), exercise-induced angina (exang), slope, ca, and thal are strong positive correlates with heart disease; thalach (max heart rate) and oldpeak are strong negative correlates.
- **Visualization:** Heatmaps and bar plots provide insights into feature relationships and target distribution.

---

## **Getting Started**

### **Prerequisites**
To run this notebook, you will need the following libraries installed:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `joblib`

### **Installation**
You can install all necessary libraries using `pip`:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### **Usage**
1. Clone the repository:  
   `git clone https://github.com/brej-29/heart-disease-prediction-ml.git`
2. Navigate to the project directory:  
   `cd heart-disease-prediction-ml`
3. Open the Jupyter Notebook:  
   `jupyter notebook`
4. Run the cells in the notebook to reproduce the analysis and model.

---

## **Data Exploration & Preprocessing**

- Loaded the dataset and performed initial inspection (`df.info()`, `df.describe()`, missing values check).
- Visualized target distribution and feature relationships.
- Explored correlations to identify key predictors.
- Split data into training and test sets.

---

## **Modeling & Evaluation**

- Built baseline models: KNN, Logistic Regression, Random Forest.
- Used cross-validation and hyperparameter tuning (`RandomizedSearchCV`, `GridSearchCV`) to optimize models.
- Evaluated models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Visualized confusion matrices and ROC curves for model comparison.
- Random Forest with tuned hyperparameters achieved the best performance (AUC ≈ 0.93).

---

## **Model Deployment**

- The best model (Random Forest) is saved using `joblib` for future predictions.
- Example code for loading and using the persisted model is provided in the notebook.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contact**
If you have any questions or feedback, feel free to reach out via my [LinkedIn Profile](https://www.linkedin.com/in/brejesh-balakrishnan-7855051b9/).
