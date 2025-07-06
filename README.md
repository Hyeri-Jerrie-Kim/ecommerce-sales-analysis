# 🛒 E-commerce Sales Analysis

## 🎯 Objective

This project is an end-to-end data science case study analyzing over **541,000 transactions** from a UK-based e-commerce retailer. It explores customer behavior, cleans and standardizes messy retail data, segments customers, and builds machine learning pipelines to predict high-value customers.

Beyond technical analytics, the project translates data insights directly into business strategies, enabling marketing teams to **target the right customers, improve ROI, and drive revenue growth**.

---

## ⚡ Quick Project Highlights

- **Goal:** Identify high-value e-commerce customers and increase ROI
- **Data:** 541,909 transactions from UK retailer
- **Key Result:** LightGBM model achieved 84% recall for high-value customers
- **Business Impact:** Targeted marketing could boost ROI from 87.68% to **194.74%**

---

## 📊 Dataset Overview

- **📍 Source:** [Kaggle - Online Retail Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
- **📁 File:** `ecommerce_data.csv`
- **🔢 Total Rows:** ~541,909 transactions  
- **🕒 Period:** Dec 2010 - Dec 2011  
- **💼 Domain Note:** Includes retail and wholesale transactions for unique all-occasion gifts.

This publicly available dataset was originally provided by the UCI Machine Learning Repository. It captures real transaction records from a UK-based online retailer. This transparency ensures reproducibility and allows anyone to replicate this analysis end-to-end.

---

## 📂 Project Structure
```
ecommerce-analysis/
├── datasets/
│ └── ecommerce_data.csv
├── notebooks/
│ └── ecommerce_analysis.ipynb
├── scripts/
│ └── ecommerce_analysis.py
├── images/
│ └── *.png
├── README.md
└── requirements.txt
```
---

## 🗺️ Data Pipeline Overview

To be inserted
![Data Pipeline](images/data_pipeline.png)

---

## ⚙️ Installation & Usage
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
Or run the Python script version:
   ```sh
   python scripts/ecommerce_analysis.py
   ```

---

## ✅ Key Results & Visualizations

### Revenue Trends

![Monthly Revenue](images/monthly_revenue.png)

- November saw the highest sales, confirming strong seasonal trends.
- Weekday sales peaked on **Tuesdays and Thursdays**.
- High-value purchases were mostly made between 10 AM and 3 PM.

---

### Customer Segmentation

![RFM Segments](images/rfm_segments.png)

- RFM and K-Means revealed high-value customer groups characterized by:
  - Frequent purchases
  - Diverse product baskets
  - Recent engagement

---

### Predictive Modeling

![SHAP Summary](images/shap_summary.png)

- LightGBM achieved:
  - **Recall → 0.84**
  - **F1-score → 0.81**
- SHAP analysis highlighted:
  - `TotalItems`, `Frequency`, and `IsBulk` as top drivers of high-value classification.

---

### Business Impact

![ROI Simulation](images/roi_simulation.png)

ROI simulations showed targeting high-value customers could increase ROI from **87.68%** to **194.74%**, proving the business value of predictive analytics.

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

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## 📄 License

This project is licensed under the MIT License.  

---

## 📬 Contact Me
📧 [Hyeri Kim](mailto:hyeri5524@gmail.com) | 🌐 [LinkedIn](https://linkedin.com/in/hyerikim-ds)

