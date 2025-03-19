# EY Open Science AI & Data Challenge 2025

## 🌍 About the Challenge

Aligned with the **United Nations Sustainable Development Goals** and the **EY Ripples program**, the **EY Open Science AI & Data Challenge** is an annual competition that provides university students, early-career professionals, and EY employees the opportunity to develop **AI-driven data models** to tackle climate issues and create a **more sustainable future**.

### 🔥 2025 Theme: Urban Heat Island Effect

The **Urban Heat Island (UHI) effect** occurs due to the high density of buildings and the lack of green spaces and water bodies in urban areas. Temperature variations between rural and urban environments can exceed **10°C**, leading to significant **health, social, and energy-related issues**. Vulnerable populations include **young children, older adults, outdoor workers, and low-income communities**.

### 🎯 Goal of the Challenge

Participants must develop a **machine learning model** to:
- **Predict heat island hotspots** in an urban area.
- **Identify key factors** contributing to the formation of these hotspots.

### 📅 Timeline
- **Start Date:** January 20, 2025
- **End Date:** March 20, 2025
- **Participation:** Individual or teams (up to **3 people** per team)

## 📊 Problem Statement

Participants will be provided with **near-surface air temperature data** collected on **July 24, 2021**, using a ground traverse in the **Bronx and Manhattan** regions of **New York City**. The dataset consists of:
- **Traverse points (latitude and longitude).**
- **UHI (Urban Heat Island) Index values.**

The **UHI Index** at any location represents the **relative temperature difference** at that specific point **compared to the city's average temperature**. This metric is crucial for assessing the **intensity of heat** in different urban zones.

The task is to develop a **regression model** that accurately predicts **UHI Index values** for a given set of locations.

---

## 🚀 Repository Structure

```
📂 EY-Open-Science-AI-Data-Challenge-2025
├── 📂 data/                 # Raw and processed datasets
├── 📂 notebooks/            # Jupyter notebooks for analysis & model development
│   ├── vf_clean.ipynb       # Main notebook containing data exploration & model building
├── 📂 models/               # Saved machine learning models
├── 📂 reports/              # Generated reports and visualizations
├── 📜 README.md             # Project overview (this file)
└── 📜 requirements.txt      # Dependencies and packages
```

## 🛠 Setup Instructions

### 1️⃣ Clone the Repository
```
git clone https://github.com/botelhogama/EYdatachallenge/
cd EYdatachallenge
```

### 2️⃣ Install Dependencies
Make sure you have Python 3.8+ installed. Then, install required dependencies:
```
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook
```
jupyter notebook notebooks/vf_clean.ipynb
```

## 📈 Dataset Information

```
| Column       | Description |
|-------------|-------------|
| Latitude    | Latitude coordinate of the measurement point |
| Longitude   | Longitude coordinate of the measurement point |
| UHI Index   | Urban Heat Island Index value |
```

## 🔍 Methodology

1. **Exploratory Data Analysis (EDA)** – Visualizing and understanding the dataset.
2. **Feature Engineering** – Identifying relevant features contributing to UHI.
3. **Model Selection & Training** – Using regression models to predict UHI Index values.
4. **Evaluation** – Assessing model performance using appropriate metrics.
5. **Insights & Recommendations** – Providing actionable insights to mitigate UHI effects.

## 🤝 Contributing
We welcome contributions! Feel free to fork this repo, create a feature branch, and submit a pull request.

## 📜 License
This project is licensed under the **MIT License**.

---

🎯 **Let's use AI to build a sustainable and cooler future for urban communities!** 🚀

