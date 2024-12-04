# Diabetes Prediction Model

This repository contains the implementation of a machine learning project aimed at predicting diabetes using health-related data. The project involves cleaning and preprocessing the dataset, handling class imbalance, feature selection, and evaluating various models to optimize performance.

## Features and Workflow

### **1. Dataset**
- The dataset includes features such as:
  - **HighBP**: Indicator of high blood pressure.
  - **HighChol**: Indicator of high cholesterol.
  - **BMI**: Body Mass Index.
  - **Smoker**: Smoking status.
  - **Stroke**: History of stroke.
  - **HeartDiseaseorAttack**: Presence of heart disease or heart attack.
  - **PhysActivity**: Physical activity levels.
  - **GenHlth**: General health status.
  - **MentHlth**: Mental health status.
  - **PhysHlth**: Physical health status.
  - **DiffWalk**: Difficulty walking.
  - **Sex**, **Age**, **Education**, and **Income**.
- Duplicate rows were detected and inspected to ensure data consistency, with 83.4% of rows being unique.

### **2. Preprocessing**
- Duplicates were identified using `DataFrame.duplicated()` to filter and inspect repeated rows across all columns.
- Rows flagged as duplicates were retained for exploration and may later be addressed based on the specific requirements of the task.

### **3. Feature Engineering**
- **Principal Component Analysis (PCA):**
  - A scree plot suggested retaining **7 components**, but practical constraints led to the removal of 6 features based on explained variance and domain relevance.
  - This ensured simplicity while maintaining predictive power.

### **4. Handling Class Imbalance**
- The dataset was heavily imbalanced between diabetic (minority class) and non-diabetic (majority class) cases.
- Techniques applied:
  - **Random UnderSampling**: Improved recall to 73.2%, prioritizing diabetic case identification. However, precision dropped to 32.4%.
  - **Oversampling with SMOTE**: Generated synthetic samples for the minority class to enhance model performance.
  
### **5. Models and Evaluation**
#### **Decision Tree**
- Initial results showed overfitting:
  - **Training Accuracy**: 99.1%
  - **Testing Accuracy**: 75.8%
- Pruning methods:
  - **Manual Pruning**: Stabilized accuracy at 83.56%, but recall for diabetic cases remained 0 due to bias toward the majority class.
  - **Cost-Complexity Pruning (CCP)**: Used alpha parameter tuning to improve generalization, but recall remained poor.

#### **Random Forest**
- Tested with oversampling and SMOTE to address class imbalance:
  - **Oversampling**:
    - **Recall**: 29.89%
    - **Accuracy**: 76.10%
  - **SMOTE**:
    - **Recall**: 39.3%
    - **Precision**: 42.1%
    - **Accuracy**: 81.9%
    - **ROC-AUC**: 0.5762 (limited discriminatory power).

### **6. Final Model Selection**
- **RandomUnderSampler with Decision Tree**:
  - Selected for its high recall (73.2%) in detecting diabetic cases, essential in healthcare applications.
  - Despite lower precision (32.4%) and accuracy (70.5%), recall was prioritized to minimize undetected diabetic cases.

---

## Key Insights
- Accuracy alone is misleading in imbalanced datasets.
- High recall is critical in healthcare to reduce false negatives, even at the cost of increased false positives.
- Model interpretability and simplicity were balanced with performance metrics to make the solution actionable.

---

## Visualizations
- Confusion Matrices: Highlight the model’s performance in distinguishing diabetic and non-diabetic cases.
- Scree Plot: Shows the variance explained by PCA components.

---

## Requirements
To run the code, ensure the following dependencies are installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn


---


