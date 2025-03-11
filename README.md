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
  - The **United Kingdom** leads with total sales of approximately **$8.96M**.
  - Other notable contributors include the Netherlands, EIRE, Germany, and France.
- **Top Products:**
  - Analysis of the top 10 products by total sales identifies the best-selling items.
  - These products guide targeted marketing efforts and help optimize inventory management.

### 2. Time Series & Transaction Patterns
- **Yearly Trends:**
  - Sales increased dramatically from **December 2010 (812k)** to **December 2011 (9.77M)**.
  - *Note:* The 2010 data only covers December, emphasizing the need for monthly analysis.
- **Monthly Trends:**
  - A clear seasonal pattern emerges with a December peak, a January dip (post-holiday slowdown), and a recovery in March.
- **Weekly & Daily Patterns:**
  - **Weekly:** Sales peaked during the week of **December 12, 2010**.
  - **Day-of-Week:** 
    - Highest sales on **Tuesdays and Thursdays**.
    - Lowest sales on **Sundays**; absence of Saturday data suggests potential store closures or data gaps.
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
  - Extensive data cleaning was performed by removing placeholder entries (e.g., “?”, “damages”, “samples”), short descriptions, cancelled transactions, and duplicate records.
  - Over **135K transactions** were flagged as incomplete.
- **Actionable Insight:**
  - Continuous monitoring and improvement of data quality are essential to maintain reliable insights and support informed decision-making.

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

