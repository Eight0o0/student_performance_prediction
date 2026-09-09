# 🎓 Student Performance Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning system that predicts student academic performance using the UCI Student Performance dataset. The system identifies at-risk students and provides actionable insights for educational interventions.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Results & Insights](#results--insights)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project leverages machine learning to predict whether a student will pass or fail based on various academic, demographic, and behavioral factors. The system is designed to help educational institutions identify at-risk students early and implement targeted interventions.

### Business Impact
- **Early Warning System**: Identify struggling students before they fail
- **Resource Optimization**: Allocate support resources efficiently
- **Personalized Interventions**: Tailor support based on specific risk factors
- **Data-Driven Decisions**: Enable evidence-based educational policies

---

## ✨ Key Features

### Machine Learning Pipeline
- ✅ **Automated Data Preprocessing**: Handling missing values, encoding, and scaling
- ✅ **Feature Engineering**: Creating intelligent features like effort scores and risk flags
- ✅ **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- ✅ **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- ✅ **Model Interpretability**: SHAP analysis and feature importance visualization

### Web Application
- 📊 **Interactive Dashboard**: Real-time predictions through Streamlit
- 🎯 **Student Profiling**: Input student characteristics for instant predictions
- 📈 **Visual Analytics**: Display prediction probabilities and risk factors
- 💡 **Actionable Insights**: Provide recommendations based on prediction results

---

## 📊 Dataset

**Source**: UCI Machine Learning Repository - Student Performance Dataset

### Dataset Statistics
- **Samples**: 395 student records
- **Features**: 33 attributes (demographic, academic, behavioral)
- **Target**: Final grade (G3) - converted to binary Pass/Fail

### Key Features
| Category | Features |
|----------|----------|
| **Demographic** | Age, Gender, Address, Family Size, Parental Status |
| **Academic** | Past Failures, Study Time, Absences, School Support |
| **Behavioral** | Alcohol Consumption, Going Out, Health, Romantic Status |
| **Family** | Parent Education, Parent Jobs, Family Relationships |
| **Extra-curricular** | Activities, Paid Tutoring, Nursery, Internet Access |

### Feature Engineering Highlights
- **Effort Score**: Combined metric of study time and attendance
- **Risk Flags**: High absence risk, past failure history
- **Support Score**: Aggregated family and school support
- **Absence Rate**: Normalized absence count per class

---

## 🏗️ Technical Architecture
┌─────────────────────────────────────────────────────────────┐
│ Data Collection Layer │
│ (UCI Student Performance Dataset) │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Data Preprocessing Layer │
│ • Data Cleaning • Feature Engineering │
│ • Encoding • Scaling │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Model Training Layer │
│ • Logistic Regression • Random Forest • XGBoost │
│ • Cross-Validation • Hyperparameter Tuning │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Model Evaluation Layer │
│ • Accuracy/F1 Score • ROC-AUC │
│ • Confusion Matrix • Feature Importance │
│ • SHAP Analysis • Error Analysis │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Deployment Layer │
│ (Streamlit Web Application) │
└─────────────────────────────────────────────────────────────┘

