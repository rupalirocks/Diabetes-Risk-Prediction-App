# Diabetes-Risk-Prediction-App
Diabetes Risk Prediction App An interactive, AI-driven Streamlit dashboard that helps healthcare professionals explore diabetes patterns and estimate an individual’s risk of having diabetes using clinical and lifestyle features.
**Project overview**
Background
Diabetes is one of the fastest-growing chronic diseases worldwide and is associated with complications such as heart disease, kidney failure, nerve damage, and vision impairment. Early detection and targeted prevention are essential to reduce long‑term health and economic burden.
This project uses the Comprehensive Diabetes Clinical Dataset (Kaggle) (110k rows × 16 columns, MIT license) to build an end‑to‑end analytical and predictive pipeline for diabetes risk assessment.
Pain points for healthcare institutions
Many healthcare institutions lack practical tools to:

- Identify individuals at high risk of diabetes early, using routinely collected clinical data
- Understand how diabetes prevalence varies across demographic, lifestyle, and clinical factors
- Support data‑driven prevention strategies, targeted outreach, and resource planning
- Without these insights, high‑risk individuals may be missed, screening may not be prioritised, and preventable complications can accumulate over time.

**Problem statement**
This project addresses two core questions:
**Population‑level analytics**
How can healthcare institutions visualise and analyse diabetes prevalence across demographic, clinical and lifestyle factors to identify high‑risk groups and meaningful patterns for targeted interventions?
**Individual‑level risk prediction**
How can we predict an individual’s likelihood of having diabetes using features such as age, BMI, HbA1c level, blood glucose, hypertension and heart disease with reliable accuracy to support early detection and preventive care?

Proposed target and features

Target variable: diabetes (Yes / No)
Key features used in modelling (depending on final selection and encoding):

- Age
- Gender
- BMI
- HbA1c level
- Blood glucose level
- Hypertension
- Heart disease

**Project aims**
This project aims to develop an AI‑driven analytical dashboard that supports:

- Early detection – estimating diabetes risk for individual patients
- Trend monitoring – understanding patterns in diabetes prevalence across subgroups
- Data‑driven decision‑making – informing outreach, screening and prevention initiatives

Using Streamlit, the app enables clinicians, data teams and public health professionals to:

- Explore diabetes prevalence across demographic (age, gender) and clinical (BMI, HbA1c, blood glucose) factors.
- Identify high‑risk groups that may benefit from targeted screening or wellness programmes
- Predict an individual’s likelihood of having diabetes using a trained machine learning model (e.g. Random Forest with class imbalance handling and tuning)
- View model explainability outputs (e.g. SHAP analysis) to understand which features drive predicted risk

The goal is to empower healthcare institutions with actionable insights that enhance early intervention, improve resource allocation and reduce long‑term healthcare burdens.
Dataset
Source: Comprehensive Diabetes Clinical Dataset – Kaggle
Author: Priyam Choksi
License: MIT
Access method: Direct CSV download from Kaggle
Shape: ~110,000 records × 16 columns
The dataset includes demographic, clinical and lifestyle variables such as age, gender, BMI, HbA1c level, blood glucose level, hypertension, heart disease, and smoking status.

**Features of the app**
*Tab 1 – Diabetes Prediction*

* User inputs for age, BMI, HbA1c level, blood glucose and other relevant factors
* Backend machine learning model (e.g. Random Forest) to estimate the probability of diabetes
* Risk category output (Low / Moderate / High) with clear explanatory text

*Tab 2 – Lifestyle Recommendations*

* Risk‑stratified educational guidance for low, moderate and high predicted risk
* Sections on Nutrition Therapy, Lifestyle Coaching, Monitoring and Education / remission mindset
* Emphasis on general education only and the need for clinician review for diagnosis and treatment decisions

*Tab 3 – Exploratory Data Analysis (EDA)*

* Visualisations of diabetes prevalence and distributions across key variables
* “Diabetes profile” comparisons (average clinical profiles for diabetic vs non‑diabetic groups)
* Class imbalance and summary insights from the dataset

*Tab 4 – Model Performance & Explainability*

* Model training pipeline overview (baseline, SMOTE, tuning, cross‑validation)
* Performance metrics (accuracy, precision, recall, F1‑score)
* Feature importance and SHAP‑based interpretability to explain predictions

Installation
Clone the repository
bash
git clone https://gitlab.com/nyp-sg/cet/shc-c-nyp-sit-sctp/daai-intake-02/sctp-capstone-ii-25.git
cd sctp-capstone-ii-25
Create and activate a virtual environment (optional but recommended)
bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
Install dependencies
bash
pip install -r requirements.txt
Add the dataset
Download the CSV file from Kaggle (Comprehensive Diabetes Clinical Dataset).
Place it in the expected data directory or path used in the app (for example data/diabetes_data.csv), matching the path in your code.
Usage
Run the Streamlit app locally:
bash
streamlit run app_DiabetesPrediction.py
Then open the URL shown in the terminal (usually http://localhost:8501) in your browser.

Typical workflow:

- Go to the Diabetes Prediction tab, enter patient information and generate a predicted diabetes risk and risk category.
- Review Lifestyle Recommendations tailored to the predicted risk level.
- Explore EDA to understand dataset patterns and high‑risk groups.
- Review Model Performance & Explainability to see how the model was trained and what drives its predictions.

Project structure (example)
text

sctp-capstone-ii-25/
├─ app_DiabetesPrediction.py   # Main Streamlit app
├─ models/
│  └─ diabetes_rf_model.pkl    # Trained model (if saved)
├─ data/
│  └─ diabetes_data.csv        # Kaggle dataset (not committed if large)
├─ notebooks/                  # Optional: EDA / modelling notebooks
├─ requirements.txt
└─ README.md

**Roadmap / possible extensions**

- Add additional models (e.g. Gradient Boosting, KNN, Neural Networks) and compare performance
- Enhance risk communication with more visual summaries and patient‑friendly narratives
- Add authentication and role‑based views (clinician vs data analyst)
- Integrate with APIs or databases for live data instead of static CSV
- Containerise the app (Docker) and deploy via cloud services with CI/CD

Authors and acknowledgment
Primary author: Priyam Choksi
Authors and acknowledgment
This project is released under the MIT License.
Please refer to the LICENSE file (or include one) for full terms and conditions.

**App and model development: Rupali Rajesh Desai
Developed as part of SCTP Capstone II (Data Analytics with AI) from Nanyang Polytechnic, Singapore.**
Dataset courtesy of the original Kaggle author; licensing details as per MIT terms.



