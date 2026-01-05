 Teacher Analytics Dashboard with Explainable AI

## Overview

The **Teacher Analytics Dashboard** is a data-driven decision support system designed to help educators and academic administrators objectively analyze student performance, engagement, and risk levels. The system integrates **machine learning models with explainable AI (XAI)** to ensure transparency, interpretability, and actionable insights rather than black-box predictions.

This project focuses on identifying **at-risk students**, understanding **key contributing factors**, and enabling **early pedagogical interventions** through a clean, interactive web dashboard.

---

## Key Objectives

* Analyze academic and behavioral data of students
* Predict student risk levels using machine learning
* Provide **explainable insights** using SHAP and LIME
* Support teachers with data-backed intervention decisions
* Present analytics through an intuitive dashboard interface

---

## Features

* **Student Risk Prediction** using ensemble ML models
* **Explainable AI (XAI)** to justify every prediction
* **Class-level and individual-level analytics**
* **Interactive dashboard** built with Streamlit
* **Production-ready modular architecture**
* **Scalable design** suitable for real educational datasets

---

## Machine Learning Approach

The system uses a **robust ensemble learning strategy** to balance accuracy and interpretability.

**Models Used**

* Random Forest Classifier
* Support Vector Machine (SVM)

**Key Input Features**

* Attendance percentage
* Assignment delay
* Participation index
* Assessment scores
* Academic consistency indicators

**Explainability**

* **SHAP (SHapley Additive exPlanations)** for global and local feature importance
* **LIME (Local Interpretable Model-agnostic Explanations)** for instance-level explanations

This ensures that teachers can clearly understand *why* a student is classified as high or low risk.

---

## System Architecture

```
├── app.py                  # Streamlit application
├── model.py                # ML models and ensemble logic
├── preprocessing.py        # Data validation and cleaning
├── explanations.py         # SHAP and LIME explanation utilities
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

---

## Tech Stack

* **Programming Language:** Python
* **Frontend & Dashboard:** Streamlit
* **Machine Learning:** scikit-learn
* **Explainable AI:** SHAP, LIME
* **Data Processing:** Pandas, NumPy

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/teacher-analytics-dashboard.git
cd teacher-analytics-dashboard
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## Dataset

Due to size constraints, datasets are **not included** in this repository.

Expected dataset columns:

* `attendance_pct`
* `assignment_delay`
* `participation_index`
* `score`
* `max_score`

You may use:

* Institutional academic datasets
* Synthetic or anonymized student data



## Use Cases

* Early identification of academically at-risk students
* Teacher performance analysis based on student outcomes
* Data-driven academic counseling
* Institutional learning analytics
* Research on explainable AI in education



## Why This Project Matters

Traditional academic evaluation methods are often **subjective and reactive**. This project introduces:

* Objective risk prediction
* Transparent AI explanations
* Early intervention capability
* Scalable analytics for educational institutions

It aligns strongly with **AI ethics**, **interpretability**, and **real-world deployment standards**.



## Future Enhancements

* Integration with Learning Management Systems (LMS)
* Time-series performance forecasting
* Teacher recommendation engine
* Role-based dashboards (Admin / Teacher / Counselor)
* Cloud deployment (AWS / GCP)

