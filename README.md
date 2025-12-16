# 🏠 House Price Prediction - End-to-End ML System

An interactive **Streamlit application** for predicting house prices using machine learning. This project demonstrates a complete ML pipeline including data cleaning, exploratory data analysis (EDA), model training, and real-time predictions.

---

## 🚀 Live Demo

[Click here to try the app!](https://house-price-predictions-affkauvvmucwpfugdyaowm.streamlit.app/)

---

## 🎯 Project Overview

This application uses the **California Housing Dataset** to predict house prices based on various features like median income, house age, location, and more. It showcases multiple regression algorithms and provides an intuitive interface for users to explore data and make predictions.

### ✨ Key Features

- **📊 Comprehensive EDA**: Interactive visualizations and statistical analysis
- **🧹 Data Cleaning**: Automated outlier removal and preprocessing
- **🤖 Multiple ML Models**: 6 different regression algorithms
- **📈 Model Comparison**: Side-by-side performance metrics
- **🔮 Interactive Predictions**: Real-time price predictions with custom inputs
- **📉 Visual Analytics**: Feature importance, correlation matrices, and more

---

## 🎓 Problem Statement

### The Challenge

Predicting house prices accurately is crucial for:
- **Real Estate Valuation**: Determining fair market value
- **Investment Decisions**: Identifying profitable opportunities
- **Market Analysis**: Understanding pricing trends
- **Buyer/Seller Guidance**: Setting realistic expectations

### Issues Addressed

1. **Data Quality**: Outliers and inconsistencies in real estate data
2. **Feature Selection**: Identifying which factors most influence price
3. **Model Selection**: Choosing the right algorithm for accuracy
4. **Interpretability**: Understanding why models make certain predictions

---

## ✅ Solution Implemented

### Our Approach

#### 1. Data Cleaning & Preprocessing
- **Outlier Removal**: Using IQR (Interquartile Range) method
- **Missing Value Handling**: Robust data cleaning pipeline
- **Feature Standardization**: Scaling features for optimal model performance

#### 2. Exploratory Data Analysis (EDA)
- **Correlation Analysis**: Understanding feature relationships
- **Distribution Analysis**: Examining data spread and patterns
- **Scatter Plots**: Visualizing feature-target relationships
- **Statistical Summaries**: Comprehensive data insights

#### 3. Multiple Regression Models
We implement and compare 6 different algorithms:

| Model | Type | Best For |
|-------|------|----------|
| Linear Regression | Baseline | Simple relationships |
| Ridge Regression | Regularized | Preventing overfitting |
| Lasso Regression | Regularized | Feature selection |
| Random Forest | Ensemble | Complex patterns |
| Gradient Boosting | Ensemble | High accuracy |
| Support Vector Regressor | Non-linear | Non-linear relationships |

#### 4. Model Evaluation Metrics
- **R² Score**: Explains variance in predictions (higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

#### 5. Interactive Prediction Interface
- User-friendly sliders for input features
- Real-time predictions from all models
- Visual comparison of model predictions

### 🎊 Results Achieved

- **R² Scores**: 0.75 - 0.82 (ensemble models)
- **RMSE**: $40,000 - $50,000 (average error)
- **Key Predictors**: Median Income, Location, House Age

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 📁 Project Structure

```
house-price-prediction/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── (data automatically loaded from sklearn)
└── assets/
    └── live_app.png
```

---

## 💻 How to Use

### 1. Problem & Solution
- Understand the project objectives
- Learn about the approach and methodology

### 2. Dataset Overview
- View sample data and statistics
- Understand feature descriptions
- See data cleaning impact

### 3. EDA (Exploratory Data Analysis)
- Explore correlation matrices
- Analyze feature distributions
- Examine scatter plots and relationships

### 4. Model Training
- Compare performance of 6 different models
- View R², RMSE, and MAE metrics
- Analyze feature importance
- See actual vs predicted plots

### 5. Make Predictions
- Adjust sliders for house features
- Get predictions from all models
- Compare different model outputs
- See average predicted price

---

## 📊 Dataset Information

**California Housing Dataset** (built-in sklearn)

### Features:
1. **MedInc**: Median income in block group (in $10,000s)
2. **HouseAge**: Median house age in block group
3. **AveRooms**: Average number of rooms per household
4. **AveBedrms**: Average number of bedrooms per household
5. **Population**: Block group population
6. **AveOccup**: Average number of household members
7. **Latitude**: Block group latitude
8. **Longitude**: Block group longitude

### Target:
- **Price**: Median house value (in $100,000s)

### Dataset Stats:
- **Total Records**: ~20,640
- **After Cleaning**: ~18,500 (outliers removed)
- **Features**: 8 numeric features
- **Target**: Continuous (regression problem)

---

## 🔍 Model Performance

Typical performance on test set:

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Random Forest | 0.81 | $48,000 | $32,000 |
| Gradient Boosting | 0.79 | $51,000 | $35,000 |
| Ridge Regression | 0.76 | $55,000 | $38,000 |
| Linear Regression | 0.76 | $55,000 | $38,000 |
| Lasso Regression | 0.75 | $56,000 | $39,000 |
| SVR | 0.71 | $62,000 | $42,000 |

---

## 🎨 Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization

---

## 📈 Key Insights

From my analysis, I discovered:

1. **Median Income** is the strongest predictor (correlation: 0.68)
2. **Location** (Lat/Long) significantly impacts prices
3. **House Age** has moderate influence
4. **Ensemble methods** (Random Forest, Gradient Boosting) perform best
5. **Feature interactions** matter more than individual features

---

## 🔮 Future Improvements

- [ ] Add more advanced models (XGBoost, LightGBM)
- [ ] Implement hyperparameter tuning
- [ ] Add cross-validation results
- [ ] Include more datasets (different regions)
- [ ] Add time-series analysis for price trends
- [ ] Implement feature engineering techniques
- [ ] Add model explainability (SHAP values)
- [ ] Create API endpoint for predictions

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Created with ❤️ by Diganta Datta

- GitHub: [@digantadatta45](https://github.com/digantadatta45)

---

## 🙏 Acknowledgments

- California Housing Dataset from sklearn
- Streamlit for the amazing framework
- Scikit-learn for ML algorithms
- The open-source community

---

## 📞 Contact

Have questions or suggestions? Feel free to:
- Open an issue
- Submit a pull request
- Email: digantadatta45@gmail.com

---

**⭐ If you found this project helpful, please give it a star!**
