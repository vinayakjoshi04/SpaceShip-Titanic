# Spaceship Titanic - Machine Learning Classification Project

## Overview

This project implements a machine learning solution for the Kaggle Spaceship Titanic competition, where the objective is to predict which passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

## Problem Statement

Predict the binary outcome (`Transported`: True/False) for each passenger based on their demographic information, cabin details, and spending patterns using supervised machine learning techniques.

## Dataset

The dataset contains passenger records with the following key features:
- **PassengerId**: Unique identifier (format: gggg_pp)
- **HomePlanet**: Planet of departure
- **CryoSleep**: Whether passenger was in suspended animation
- **Cabin**: Cabin location (format: deck/num/side)
- **Destination**: Planet destination
- **Age**: Passenger age
- **VIP**: VIP service status
- **Spending Features**: RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
- **Name**: Passenger name
- **Transported**: Target variable (True/False)

## Project Structure

```
Spaceship-Titanic-Project/
├── Spaceship-Titanic.ipynb    # Main analysis and modeling notebook
├── output.csv                 # Final predictions for submission
├── train.csv                  # Training dataset (not included)
├── test.csv                   # Test dataset (not included)
└── README.md                  # Project documentation
```

## Methodology

### Data Preprocessing

1. **Feature Engineering**
   - Extracted `GroupId` and `GroupMember` from `PassengerId`
   - Split `Cabin` into `CabinDeck`, `CabinNum`, and `CabinSide`
   - Created `IsAlone` feature based on group size analysis
   - Aggregated spending features into `TotalSpend`

2. **Missing Value Treatment**
   - Categorical features: Imputed with "Missing" category
   - Numerical features: Applied appropriate statistical imputation
   - `CabinNum`: Filled with -1 for missing values

3. **Feature Encoding**
   - Applied Label Encoding to categorical variables
   - Standardized numerical features using StandardScaler

4. **Feature Selection**
   - Removed non-predictive features (e.g., `Name`)
   - Retained engineered features with high predictive potential

### Model Development

- **Baseline Model**: Logistic Regression
- **Current Performance**: ~58% accuracy

## Technical Implementation

### Dependencies

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Usage

1. Clone the repository
2. Open `Spaceship-Titanic.ipynb` in Jupyter Notebook
3. Run all cells to reproduce the analysis
4. Final predictions will be saved to `output.csv`

## Results

The current model achieves approximately 58% accuracy on the validation set. The output file contains predictions in the required competition format:

```
PassengerId,Transported
0001_01,True
0002_01,False
...
```

## Future Improvements

### Model Enhancement
- Implement ensemble methods (Random Forest, XGBoost, LightGBM)
- Perform hyperparameter optimization using GridSearchCV/RandomizedSearchCV
- Apply cross-validation for robust model evaluation

### Feature Engineering
- Create interaction features between spending categories
- Develop family/group-based features
- Implement polynomial features for numerical variables
- Apply dimensionality reduction techniques (PCA, t-SNE)

### Data Analysis
- Conduct comprehensive exploratory data analysis
- Implement feature importance analysis
- Perform correlation analysis and multicollinearity detection

## Project Status

**Current Status**: Baseline implementation complete  
**Next Steps**: Model optimization and advanced feature engineering  

## Contributing

This is an individual project for the Kaggle competition. However, suggestions and feedback are welcome through issues or pull requests.

## Acknowledgments

- Kaggle for providing the Spaceship Titanic dataset
- The machine learning community for methodological insights
- Scikit-learn documentation and tutorials

## Contact

For questions or collaboration opportunities:
- **Email**: [vinayakjoshi2004@gmail.com]
- **LinkedIn**: [https://www.linkedin.com/in/vinayak-joshi-99521528b/]

---

*Project created as part of the Kaggle Spaceship Titanic competition - a binary classification challenge in the machine learning domain.*
