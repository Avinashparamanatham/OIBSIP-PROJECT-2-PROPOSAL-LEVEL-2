# Wine Quality Prediction 🍷

A machine learning project focused on predicting wine quality using various chemical characteristics. This project implements multiple classifier models to analyze and predict wine quality based on physicochemical properties.

## 📊 Project Overview

This project aims to predict wine quality scores using machine learning algorithms. By analyzing various chemical properties such as density, acidity, and other characteristics, we can predict the quality rating of wines on a scale.

### 🎯 Features

- Implementation of three distinct classifier models:
  - Random Forest Classifier
  - Stochastic Gradient Descent Classifier
  - Support Vector Classifier (SVC)
- Comprehensive analysis of chemical properties
- Data visualization of wine characteristics
- Model performance comparison

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning Libraries:**
  - scikit-learn
  - NumPy
  - Pandas
- **Data Visualization:**
  - Matplotlib
  - Seaborn

## 📋 Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 🚀 Installation and Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/wine-quality-prediction.git
cd wine-quality-prediction
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook
```bash
jupyter notebook Wine_Quality_Prediction.ipynb
```

## 📈 Dataset

The dataset includes various chemical properties of wines, including:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

## 💻 Usage

```python
# Example code snippet
from src.models import train_model
from src.data_preprocessing import prepare_data

# Load and preprocess data
X_train, X_test, y_train, y_test = prepare_data('wine_data.csv')

# Train model
model = train_model(X_train, y_train, model_type='random_forest')

# Make predictions
predictions = model.predict(X_test)
```

## 📊 Model Performance

The project includes performance metrics for each classifier:
- Accuracy scores
- Confusion matrices
- Classification reports
- ROC curves

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author

Avinashparamanatham

Project Link: [https://github.com/yourusername/wine-quality-prediction](https://github.com/yourusername/wine-quality-prediction)
