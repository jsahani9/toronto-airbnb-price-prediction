# 🏙️ Toronto Airbnb Price Prediction

Predicting Airbnb listing prices across Toronto using advanced **machine learning** models — primarily **XGBoost** — to uncover how host behavior, property features, and location impact rental pricing.

---

## 📘 Project Overview
This project analyzes Airbnb listing data for Toronto to build a predictive model that estimates listing prices based on both **numerical** and **categorical** features such as:
- Property type, room type, and accommodates
- Host response metrics and verification status
- Listing location (latitude and longitude)
- Review scores and availability patterns

The goal is to create a reliable, data-driven pricing model for hosts and guests to understand market trends in Toronto’s Airbnb ecosystem.
You can find the data here: [InsideAirbnb Toronto Dataset](http://insideairbnb.com/get-the-data/)
---

## 🧠 Machine Learning Approach

### 🔹 Data Preprocessing
- Handled missing values using custom indicators (`*_missing`)
- Applied log transformation to target variable (`price`) to reduce skewness
- Engineered new features like:
  - `beds_per_guest`, `bathrooms_per_guest`, `bedrooms_per_guest`
  - `host_tenure_days`, `days_since_last_review`
- One-hot encoded categorical variables (`property_type`, `neighbourhood_cleansed`, etc.)

### 🔹 Model Training
We experimented with several regression models:
- Linear Regression  
- Ridge & Lasso  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- **XGBoost Regressor** (final model)

After hyperparameter tuning, **XGBoost** achieved the best results.

---

## 📈 Model Performance

| Metric | Score |
|:--|--:|
| **R²** | **0.7947** |
| **RMSE (log)** | 0.0493 |
| **MAE (log)** | 0.0314 |

✅ The model explains nearly **79%** of price variability in Toronto listings.  
✅ Errors are extremely low after the log-scale transformation.

---

## 🔍 Feature Importance (Top 10)
| Rank | Feature | Description |
|:--|:--|:--|
| 1 | `longitude` | Geographic location |
| 2 | `latitude` | Geographic location |
| 3 | `host_tenure_days` | Host experience length |
| 4 | `availability_365` | Days available in a year |
| 5 | `days_since_last_review` | Host engagement recency |
| 6 | `neighbourhood_cleansed` | Neighborhood identifier |
| 7 | `bathrooms_missing` | Missing value indicator |
| 8 | `reviews_per_month` | Activity level |
| 9 | `host_total_listings_count` | Host scale |
| 10 | `minimum_nights` | Booking policy strictness |

---

## 🧩 Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib  
- **Tools:** Jupyter Notebook, GitHub  
- **Environment:** macOS / Conda virtual env  

---

## 📂 Repository Structure


