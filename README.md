# 🛒 E-commerce Sales Analysis

## 🏷 Overview 

This project analyzes one year of online retail transactions to uncover sales patterns, customer behavior, and product performance. The goal is to turn raw transactional data into clear business insights that support strategic decision-making in the e-commerce domain.

**Data Range:** Dec 2010 – Dec 2011  •  **Countries:** 38  •  **Cleaned Records:** 397K


## 💼 Business Context

E-commerce companies often collect extensive sales data but struggle to interpret it effectively. This project addresses that gap by identifying what drives **sales growth, repeat purchases, and market concentration.** The findings provide a data-backed foundation for decisions related to marketing, inventory, and customer retention.


## 🔍 Approach & Analysis
Using the [Kaggle – Online Retail Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data), the workflow included:
- Data cleaning and validation (541,909 → 397,884 usable records).  
- Time-based revenue analysis to identify seasonality.  
- Product and regional segmentation to reveal profitability patterns.  
- Customer grouping using RFM metrics to evaluate retention potential.  
- Visual storytelling with Power BI for business presentation.


## 💡 Key Insights
- **Seasonal surge** — November and December revenue increased by 35%, confirming strong holiday-driven demand.  
- **Regional concentration** — The UK generated ~82% of total revenue, highlighting geographic risk and expansion opportunities.  
- **Top 5% customers** — Accounted for nearly half of all sales, underscoring the need for loyalty-focused marketing.  
- **Product imbalance** — A small set of high-turnover items dominated revenue, suggesting SKU optimization potential.  
- **Reorder behavior** — Frequent low-value invoices indicated B2B restocking customers rather than end consumers.  


## 📈 Visual Highlights
- `/images/monthly_sales_trend.png` — Seasonal revenue trend  
- `/images/top_countries_revenue.png` — Regional contribution  
- `/images/top_products.png` — Product performance  
- `/images/customer_segments_pca.png` — RFM-based segmentation 


## 📊 Power BI Dashboard
An interactive **Power BI dashboard** was developed to complement the Python-based analysis, allowing users to explore sales trends, customer behavior, and product performance dynamically.

📂 **Files in `/powerbi/` folder**:
- `ecommerce_dashboard.pbix` — Full interactive Power BI dashboard  
- `ecommerce_dashboard.pdf` — Static exported version for quick preview

### 🛠 How to Open
1. Download **Power BI Desktop** ([Download here](https://powerbi.microsoft.com/en-us/downloads/)).
2. Open the `.pbix` file to interact with the dashboard.


## 🧰 Tools & Libraries
Python | Pandas | Matplotlib | Seaborn | Power BI  


## 🚀 How to Run
Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/Hyeri-Jerrie-Kim/ecommerce-sales-analysis.git
   cd ecommerce-sales-analysis

   pip install -r requirements.txt
   ```
Run the notebook:
   ```bash
   jupyter notebook notebooks/ecommerce_analysis.ipynb
   ```

## 🔗 Author & Links

Hyeri Kim — 📧 [Hyeri Kim](mailto:hyeri5524@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/hyerikim-ds)


