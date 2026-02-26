# 📊 YouTube Trending Analytics Platform

A machine learning web application built with **Streamlit** to analyze and predict the popularity of YouTube trending videos. The project combines exploratory data analysis, feature engineering, and multi-model ML classification into a single interactive dashboard.

---

## 🚀 Demo

> Upload your `youtube.csv` dataset, train a model in one click, and get real-time popularity predictions.

---

## ✨ Features

- **🏠 Dashboard** — Key metrics overview (total videos, average views, popular rate, engagement)
- **📊 EDA** — Distributions, outlier detection via boxplots, correlation heatmaps, and temporal patterns
- **🔬 Data Processing** — Full preprocessing pipeline with feature engineering details
- **🤖 Machine Learning** — Train and evaluate 6 classifiers with GridSearchCV, confusion matrix, and ROC curve
- **📈 Model Comparison** — Visual comparison of all models across Accuracy, Precision, Recall, F1-Score, and AUC
- **📖 Documentation** — In-app documentation covering the full pipeline and feature descriptions

---

## 🧠 Models

| Model | Accuracy | AUC |
|---|---|---|
| CatBoost | 81.6% | 88.6% |
| XGBoost | 80.1% | 87.3% |
| LightGBM | 79.2% | 86.1% |
| Random Forest | 78.5% | 85.2% |
| Logistic Regression | 72.3% | 78.5% |
| KNN | 68.9% | 72.3% |

---

## 🔧 Feature Engineering

The pipeline engineers the following features from the raw dataset:

| Feature | Type | Description |
|---|---|---|
| `title_length` | Numeric | Number of characters in the title |
| `title_caps_word_count` | Numeric | Number of all-caps words in the title |
| `tag_count` | Numeric | Number of tags (split on `\|`) |
| `category_id` | Categorical | YouTube category (rare categories grouped) |
| `publish_hour` | Numeric | Hour of publication (0–23) |
| `is_week_end` | Binary | Published on Friday/Saturday/Sunday |
| `days_to_trending` | Numeric | Days between publication and trending |
| `Season` | Categorical | Season when the video trended |
| `engagement` | Numeric | likes + dislikes + comments |
| `comments_enabled` | Binary | Whether comments are enabled |
| `ratings_enabled` | Binary | Whether ratings are enabled |
| `country_*` | Binary | One-hot encoded country |

**Target variable:** `popular` — 1 if views > median views, 0 otherwise.

---

## 🛠️ Tech Stack

- **App framework:** [Streamlit](https://streamlit.io/)
- **Data:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/youtube-trending-analytics.git
cd youtube-trending-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset in the project root
# The app expects a file named: youtube.csv

# 4. Run the app
streamlit run app_youtube_trend.py
```

---

## 📁 Project Structure

```
youtube-trending-analytics/
│
├── app_youtube_trend.py                 # Main Streamlit application
├── projet_analyse_youtube_trend.ipynb   # Exploratory notebook
├── youtube.csv                          # Dataset (not included, add your own)
├── requirements.txt
└── README.md
```

---

## 📋 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
lightgbm
catboost
```

---

## 📊 Dataset

The app expects a CSV file named `youtube.csv` with the following columns (at minimum):

`video_id`, `title`, `publish_date`, `trending_date`, `tags`, `views`, `likes`, `dislikes`, `comment_count`, `category_id`, `publish_country`, `comments_disabled`, `ratings_disabled`

A suitable dataset can be found on [Kaggle — YouTube Trending Video Dataset](https://www.kaggle.com/datasets/datasnaek/youtube-new).

---

## 📓 Notebook

The Jupyter notebook `projet_analyse_youtube_trend.ipynb` contains the full exploratory analysis and model benchmarking that preceded the Streamlit app development.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
