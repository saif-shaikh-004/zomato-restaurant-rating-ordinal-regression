🍽️ Zomato Restaurant Rating Prediction using Ordinal Logistic Regression


🚀 Project Highlights

📊 Dataset Size: 15000 restaurants

🎯 Problem Type: Ordinal Classification (1–5 ratings)

🧠 Primary Model: Ordinal Logistic Regression

⚙️ Key Techniques

Natural Splines for non-linear relationships\
Multicollinearity testing (VIF)\
Proportional Odds assumption testing\
Partial proportional odds modeling\
Model comparison with Multinomial Logistic Regression\

📈 Best Model Performance

Metric	Value
Accuracy - 58.2%
Within-1 Rating Accuracy - 97.7%
MAE - 0.441
Weighted Kappa - 0.66
McFadden R² - 0.28

🏆 Final Model Selected: Ordinal Logistic Regression

📌 Project Overview

This project analyzes restaurant operational and service variables to predict customer food ratings on Zomato.

Since ratings are ordered (1–5 stars), Ordinal Logistic Regression is used to model the outcome appropriately. The model is also compared with Multinomial Logistic Regression.

📊 Dataset

Contains restaurant-level features such as:

Pricing, delivery time, distance
Packaging, hygiene, and support ratings
Discounts and operational metrics
Target: food_rating (1–5 ordered)

🔄 Workflow
Data → Cleaning → Scaling → Assumption Testing → Modeling → Evaluation → Insights

📈 Models
Ordinal Logistic Regression
Handles ordered outcomes
Uses splines for non-linearity
Adjusts for proportional odds violations
Multinomial Logistic Regression
Treats ratings as unordered
Used for comparison

📊 Results
Metric	 Ordinal	  Multinomial
Accuracy	58.2%	     57.9%
MAE	      0.441      0.443
AIC	      21639.35	 21689.4

🏆 Conclusion

Ordinal Logistic Regression performs better due to:

Lower error
Better model fit
Respect for ordered ratings
