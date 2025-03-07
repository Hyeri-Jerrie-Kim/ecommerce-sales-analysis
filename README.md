# 📊 E-Commerce Sales Analysis

📌 **Author:** Hyeri Kim
📅 **Last Updated:** March 2025  
📂 **Category:** Data Analysis, EDA, Business Intelligence  
🗂 **Dataset:** [E-commerce Sales Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)  

---

## 📖 Project Overview

This project explores **e-commerce sales transactions** to uncover key trends and business insights.  
By analyzing customer behavior, seasonal sales patterns, and product demand, we extract **actionable insights** to drive data-driven decisions.

---

## 📌 Key Features
- 🔍 **Data Cleaning:**  
  Handling missing values, duplicates, incomplete transactions, and outliers to ensure data quality.
- 📊 **Interactive Visualizations:**  
  Dynamic charts and graphs using Matplotlib, Seaborn, and Plotly to explore sales trends and performance.
- 🎯 **Business Insights:**  
  In-depth analysis of customer trends, revenue patterns, and seasonal sales to inform strategic decisions.
- 🕒 **Time Series Analysis:**  
  Detailed examination of yearly, monthly, weekly, daily, and hourly sales trends.

---

## 📊 Dataset
- **📍 Source:** [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **📁 File:** `data.csv`
- **🔢 Total Rows:** ~541,909 transactions  
- **🔑 Key Columns:**
  - `InvoiceNo` - Unique transaction ID  
  - `StockCode` - Product code  
  - `Description` - Product name  
  - `Quantity` - Number of units sold  
  - `InvoiceDate` - Timestamp of purchase  
  - `UnitPrice` - Price per unit  
  - `CustomerID` - Unique customer identifier  
  - `Country` - Customer's country
   
### About the Dataset
This e-commerce dataset, titled **"Online Retail"**, is sourced from the UCI Machine Learning Repository. Unlike many proprietary e-commerce datasets, this dataset is publicly available and contains actual transaction records from **December 2010 to December 2011**. The dataset comprises all transactions from a UK-based and registered non-store online retail company that mainly sells unique all-occasion gifts, with a significant number of customers being wholesalers.

### Acknowledgements
This dataset was made available by Dr. Daqing Chen, Director of the Public Analytics Group at London South Bank University. For any further details or inquiries, please contact: `chend@lsbu.ac.uk`.

---

## 🚀 Technologies Used
- **Python** 🐍 (Pandas, NumPy, Matplotlib, Seaborn, Plotly, WordCloud)
- **Jupyter Notebook** 📒
- **Git** for version control

---

## 🔧 Setup & Installation
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/Hyeri-Jerrie-Kim/ecommerce-sales-analysis.git
   cd ecommerce-sales-analysis
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Launch Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```
4. **Open the Notebook:**
   Navigate to the appropriate folder and open the ecommerce_analysis.ipynb file to run the analysis.
   
---

## 📊 Key Insights & Results

### 1. Regional Performance & Top Products Analysis
- **Regional Sales Dominance:**  
  The **United Kingdom** leads with total sales of approximately **$8.96M**, with notable contributions from the Netherlands, EIRE, Germany, and France.
- **Top Products:**  
  Analysis of the top 10 products by total sales highlights best-selling items, guiding targeted marketing and inventory optimization.

### 2. Time Series & Transaction Patterns
- **Yearly Trend:**  
  Sales surge from **December 2010 (812k)** to **December 2011 (9.77M)**. *(Note: 2010 data only covers December.)*
- **Monthly Trend:**  
  A clear seasonal pattern emerges: December peaks, January dips (post-holiday slowdown), followed by recovery in March.
- **Weekly & Daily Patterns:**  
  - **Weekly:** Sales peaked in the week of December 12, 2010.
  - **Day-of-Week:** Tuesdays and Thursdays register the highest sales, while Sundays are the lowest. The absence of Saturday data suggests either store closures or data gaps.
- **Hourly Trend:**  
  Peak transactions occur between **10 AM and 3 PM**, aligning with typical business hours.
- **Action:**  
  Optimize promotions, schedule campaigns, and adjust operational hours based on these insights.

### 3. Correlation & Revenue Drivers
- **Key Correlation:**  
  A strong positive correlation (0.91) between **Quantity** and **Total Sales** indicates that increasing sales volume is crucial for revenue growth.
- **Unit Price Impact:**  
  **UnitPrice** shows little to no correlation with revenue, suggesting that merely raising prices is unlikely to drive significant gains.
- **Action:**  
  Focus on volume-based promotions such as bundle discounts, loyalty programs, and cross-selling strategies.

### 4. Data Quality Enhancements
- **Cleaning Efforts:**  
  Significant measures were implemented to remove placeholder entries (e.g., “?”, “damages”, “samples”), short descriptions, cancelled transactions, and duplicates. Over 135K transactions were flagged as incomplete.
- **Action:**  
  Ongoing data quality monitoring is essential to maintain reliable insights.

---

## 📜 Future Improvements
- 🏷 **Customer Segmentation:**  
  Apply clustering techniques (e.g., K-Means) to identify distinct customer groups.
- 📉 **Predictive Modeling:**  
  Develop forecasting models to predict future sales trends.
- 📊 **Dashboard Creation:**  
  Build dynamic dashboards using Tableau, Power BI, or Plotly Dash for real-time insights.
- 🔄 **Automation:**  
  Enhance data ingestion and analysis workflows with automated Python scripts. 

---

## 📬 Contact Me
📧 [Hyeri Kim](mailto:hyeri5524@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/hyerikim-ds)

