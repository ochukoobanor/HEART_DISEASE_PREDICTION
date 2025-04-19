 # 🫀 Heart Disease Prediction with Machine Learning

This project aims to predict the presence of heart disease in patients using clinical and demographic data. It features full **Exploratory Data Analysis (EDA)**, machine learning with **Decision Tree** and **Random Forest** classifiers, performance evaluation, and model saving for reuse.

📁 Dataset

The dataset used is `Heart_Disease_Prediction.csv`, which includes patient medical details such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Maximum heart rate achieved
- And more...

🎯 Target:
- `Heart Disease`:  
  - `Presence` → 1  
  - `Absence` → 0  

🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand the structure, distribution, and relationships in the data.

Key steps:
- ✅ Checking for missing values
- 📊 Class distribution visualization
- 🔥 Correlation heatmap of numerical features
- 📈 Histograms of feature distributions
- 📦 Boxplots comparing features against heart disease status

> Libraries used: `matplotlib`, `seaborn`, `pandas`

---

🧠 Machine Learning Models

 🌳 **Decision Tree Classifier**
- Simple, interpretable model.
- Implemented using `sklearn.tree.DecisionTreeClassifier`.

🌲 **Random Forest Classifier**
- Ensemble of decision trees, better accuracy.
- Implemented using `sklearn.ensemble.RandomForestClassifier`.

Both models are trained using an 80/20 train-test split.

Model performance was evaluated using:
- **Classification Report**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- **Confusion Matrix** (visualized using heatmap)

Trained models are saved using `pickle` for reuse:

```python
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)
