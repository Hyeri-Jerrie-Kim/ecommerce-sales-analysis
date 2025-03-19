# 📊 E-Commerce Sales Analysis

📌 **Author:** Hyeri Kim </br>
📅 **Last Updated:** March 2025  
📂 **Category:** Data Analysis, EDA, Business Intelligence  
🗂 **Dataset:** [E-commerce Sales Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)  

---

## 📖 Project Overview

Conducted an **e-commerce sales analysis** on **541,909 transactions**, refining the dataset to **519,580 valid records** through data cleaning and preprocessing. By leveraging **time-series visualizations** and **correlation analysis**, I identified **quantity sold** as the key revenue driver—surpassing price adjustments in influencing total sales.

---

## 📌 Key Features
- 🔍 **Data Cleaning:**  
  Handled missing values, duplicates, incomplete transactions, and outliers to ensure data quality.
- 📊 **Interactive Visualizations:**  
  Created dynamic charts using Matplotlib, Seaborn, and Plotly to explore trends and performance.
- 🎯 **Business Insights:**  
  Derived in-depth analysis of customer behavior, seasonal sales patterns, and product demand.
- 🕒 **Time Series Analysis:**  
  Examined yearly, monthly, weekly, daily, and hourly trends to uncover peak seasons and high-traffic hours.

---

## 📊 Dataset
- **📍 Source:** [Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **📁 File:** `ecommerce_data.csv`
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

### 1. Core Findings
- **Quantity Sold as Revenue Driver:**
  Identified **quantity sold**—not price—as the primary factor influencing total sales.</br>
  🔸 **Action:** Implement volume-based promotions and loyalty programs to boost sales.
- **Peak Seasonal Demand:**
  Notable sales surge in **November and December**, aligning with holiday shopping.</br>
  🔸 **Action:** Optimize inventory and marketing strategies for Q4 demand.
- **High-Traffic Hours (10 AM–3 PM):**
  Most sales occur during this window.</br>
  🔸 **Action:** Schedule ads and promotions to capitalize on peak traffic.

### 2. Regional Performance & Products Analysis
- **Regional Sales Dominance:**
  The **United Kingdom** leads with total sales of approximately **$8.96M**, followed by the Netherlands, EIRE, Germany, and France.
- **Top Products:**
  - Analysis of the top 10 products by total sales identifies the best-selling items.
  - These products guide targeted marketing efforts and help optimize inventory management.

### 3. Time Series & Transaction Patterns
- **Yearly & Monthly Trends:**
  Sales surge from December 2010 (812k) to December 2011 (9.77M). December peaks, January dips, and March recovers.
- **Weekly & Daily Patterns:**
  - Peak week: **December 12, 2010**
  - Highest day-of-week sales: **Tuesdays and Thursdays**
  - Lowest day-of-week sales: **Sundays**; no Saturday data suggests closures or gaps.
- **Hourly Trends:**
  - Peak transactions occur between **10 AM and 3 PM**, aligning with standard business hours.
- **Actionable Insight:**
  - Adjust promotional campaigns, scheduling, and staffing to capitalize on these temporal trends.

### 3. Correlation & Revenue Drivers
- **Sales Volume Impact:**
  - A strong positive correlation (0.91) between **Quantity** and **Total Sales** indicates that increasing the number of items sold is key to boosting revenue.
- **Price Impact:**
  - **UnitPrice** shows little to no correlation with total revenue, suggesting that merely raising prices is not an effective strategy.
- **Actionable Insight:**
  - Prioritize volume-based promotions such as bundle discounts, loyalty programs, and cross-selling strategies to drive revenue growth.

### 4. Data Quality Enhancements
- **Cleaning Efforts:**
  Removed placeholder entries (e.g., “?”, “damages”), short descriptions, cancelled transactions, and duplicates. Over **135K** incomplete transactions flagged.
- **Actionable Insight:**
  Ongoing data quality monitoring ensures reliable insights and informed decisions.

---

## 📊 Power BI Dashboard
This project includes an interactive **Power BI dashboard** for analyzing e-commerce sales.

📂 **Files in `/powerbi/` folder**:
- **`ecommerce_dashboard.pbix`** → Full Power BI dashboard.
- **`ecommerce_dashboard.pdf`** → Static version (exported PDF).

### 🛠 How to Open
1. Download **Power BI Desktop** ([Download Here](https://powerbi.microsoft.com/en-us/downloads/)).
2. Open the `.pbix` file in Power BI.

---

## 📜 Future Improvements
- 🏷 **Customer Segmentation:**  
  Apply clustering techniques (e.g., K-Means) to identify distinct customer groups.
- 📉 **Predictive Modeling:**  
  Develop forecasting models to predict future sales trends.
- 🔄 **Automation:**  
  Enhance data ingestion and analysis workflows with automated Python scripts. 

---

## 📬 Contact Me
📧 [Hyeri Kim](mailto:hyeri5524@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/hyerikim-ds)

