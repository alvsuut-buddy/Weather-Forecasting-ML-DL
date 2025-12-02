# Weather Forecasting: Machine Learning vs Deep Learning

## Overview
This project aims to predict daily weather conditions (Drizzle, Fog, Rain, Snow, Sun) based on climatological features. The dataset used for this analysis was obtained from Kaggle.

The core value of this project is a comprehensive comparative study between two major Artificial Intelligence approaches:
1. Traditional Machine Learning (Ensemble Methods): Utilizing robust Tree-based algorithms optimized for tabular data.
2. Deep Learning (Neural Networks): Exploring neural network architectures to capture complex non-linear patterns.

## Methodology

1. Data Preprocessing
- Input Features: Precipitation, Temp_Max, Temp_Min, Wind.
- Target Variable: Multi-class weather label (Drizzle, Fog, Rain, Snow, Sun).
- Techniques: Label Encoding for categorical targets, StandardScaler for feature normalization, and Stratified Train-Test Split to maintain class distribution.

2. Models Implemented

Approach A: Machine Learning
We implemented Ensemble Learning algorithms with automated hyperparameter tuning via RandomizedSearchCV:
- Random Forest Classifier
- AdaBoost
- XGBoost (Extreme Gradient Boosting)
- CatBoost (Categorical Boosting)

Approach B: Deep Learning
We constructed Sequential architectures using TensorFlow/Keras with custom tuning loops for units, dropout rates, and learning rates:
- CNN 1D (Convolutional Neural Network for 1D sequential data)
- LSTM (Long Short-Term Memory)
- BiLSTM (Bidirectional LSTM)

## Experimental Results

Based on the experiments conducted, the Machine Learning approach (specifically Tree-based algorithms) significantly outperformed Deep Learning on this specific dataset. This is likely due to the tabular nature of the data and the limited dataset size, where Ensemble methods often excel over complex Neural Networks.

| Model Type | Algorithm | Best Accuracy |
| :--- | :--- | :--- |
| Machine Learning | CatBoost | ~84.74% |
| Machine Learning | Random Forest | ~84.61% |
| Machine Learning | AdaBoost | ~84.16% |
| Machine Learning | XGBoost | ~84.10% |
| Deep Learning | CNN 1D | ~63.82% |
| Deep Learning | BiLSTM | ~59.73% |
| Deep Learning | LSTM | ~55.97% |

Insight: While Deep Learning models like LSTM are powerful for large-scale time-series data, for this specific tabular weather dataset, CatBoost and Random Forest provided the most efficient and accurate solutions.

## Key Features
- Hyperparameter Tuning: Utilized RandomizedSearchCV for ML models to find optimal parameters.
- Robust Validation: Performed iterative retraining (10 runs) to ensure the stability of the ML model accuracy.
- Comprehensive Evaluation: Assessed models using Confusion Matrices, Classification Reports, and Accuracy Scores.
- Visualizations: Generated Heatmaps for error analysis and Barplots for Feature Importance.

## Installation & Usage

1. Clone the repository:
   git clone https://github.com/alvsuut-buddy/Weather-Forecasting-ML-DL.git

2. Install required dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost catboost adaboost

3. Run the notebooks:
   Open the notebooks in Jupyter Lab, Jupyter Notebook, or Google Colab to reproduce the results.

## Credits
Developed by Alvin Oktavian Surya Saputra and Dhia Ahmad Farras as part of an AI Portfolio Project. Data sourced from Kaggle.
