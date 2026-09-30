# Employee Attrition Prediction — ML/MLOps Project

## 📌 Overview

This project develops a machine learning pipeline to predict whether an employee is likely to leave an organization based on employee-related attributes.

The project focuses not only on model training but also on building a reproducible ML workflow using **data validation, preprocessing, model training, evaluation, MLflow experiment tracking, and model registry**.

---

## 🎯 Problem Statement

Employee turnover can create significant costs for organizations through recruitment, hiring, and training of replacement employees.

The objective of this project is to build a machine learning system that predicts employee attrition and provides a structured ML pipeline that can be reproduced and tracked.

**Target variable:** `Attrition`

- `Yes` → Employee left the organization
- `No` → Employee stayed in the organization

---

## 📊 Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

### Dataset characteristics

- **Records:** 1,470 employees
- **Original features:** 35
- **Target:** `Attrition`
- **Missing values:** None
- **Duplicate records:** None

The dataset contains employee information such as:

- Age
- Business Travel
- Department
- Job Role
- Monthly Income
- Job Satisfaction
- Years at Company
- Work-Life Balance
- OverTime
- Stock Option Level
- Total Working Years
- And other employee-related attributes

---

## 🏗️ Project Workflow

The project follows an end-to-end ML lifecycle:

text
Raw Dataset
     ↓
Data Validation
     ↓
Data Preprocessing
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Experiment Tracking with MLflow
     ↓
Model Registry
     ↓
Validation & Reproducibility
