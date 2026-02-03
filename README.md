# Student Dropout Risk Prediction & Early Intervention System

This project presents an AI-based decision-support system designed to identify students at risk of dropping out and to suggest targeted early interventions. The system combines an Artificial Neural Network (ANN) with interpretability-driven analysis and what-if simulations.

---

## Problem Statement

High dropout rates in higher education negatively impact students and institutions. Early identification of at-risk students enables timely academic and financial interventions, improving student retention and outcomes.

---

## Key Features

- Binary prediction of student continuation vs dropout risk
- Probability-based risk scoring and confidence indication
- Student risk profiling (academic, financial, combined)
- Personalized academic and financial recommendations
- Goal-based intervention planning
- Multi-parameter what-if simulation
- Contextual comparison with baseline cohort patterns
- Auto-generated intervention summary for decision support

---

## Model Overview

- Artificial Neural Network (ANN)
- Input features include demographic, socio-economic, and first-semester academic indicators
- Output: probability of successful continuation
- Implemented using TensorFlow/Keras
- Preprocessing with scikit-learn (encoding and scaling)

---

## Project Structure

student-dropout-intervention-system/

- app.py
- student_success_binary_model.h5
- preprocessor.pkl
- feature_names.pkl
- requirements.txt
- README.md

dataset_used/

- dataset.csv
- tA1_marital_status.csv
- tA2_nationality.csv
- tA3_application_mode.csv
- tA4_course_names.csv
- tA5_previous_quals.csv
- tA6_parent_previous_quals.csv
- tA7_parent_occupation.csv
- tA8_gender.csv
- tA9_attendance_regime.csv
- tA10_yes_no.csv

notebook/

- training_notebook.ipynb

---

## How to Run the Application

1. Clone or download the repository
2. Install dependencies using:

pip install -r requirements.txt

3. Run the Streamlit application using:

streamlit run app.py

---

## Notes on Baseline Comparisons

Baseline comparisons shown in the application are assumed cohort-level references provided purely for contextual interpretation.  
They are independent of the model’s training data and do not influence predictions.

---

## Limitations

- The model is trained on a single public dataset and may not generalize to all institutions
- Baseline comparisons are indicative rather than empirically derived
- The system provides decision support and does not replace academic counseling

---

## Future Scope

- Incorporation of cohort-derived baseline statistics
- Extension to multi-class outcome prediction (Dropout / Enrolled / Graduate)
- Deployment for institutional-scale use
- Integration with student information systems

---

## License

This project is intended for academic and educational purposes.
