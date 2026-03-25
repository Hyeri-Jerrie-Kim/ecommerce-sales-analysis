# 🛒 E-Commerce Revenue Strategy Audit
**Operational Audit & Behavioral Segmentation for Strategic Growth**

## 🎯 Strategic Objective
The primary objective of this project is to architect a **Strategic Decision-making Framework** for a UK-based online retailer. By integrating temporal, geographic, and behavioral analytics, the study identifies high-velocity operational windows and segments the customer base into actionable tiers to maximize $ROI$ and enhance **Revenue Recovery**.

**Data Scope:** Dec 2010 – Dec 2011 • **Cleaned Records:** 397K+ Transactions • **Market Reach:** 38 Countries

---

## 🚀 Strategic Insights (Key Findings)
- **The Golden Window**: Identified a high-velocity sales window during **Mid-week (Tue-Thu)** and **Mid-day (10:00 - 15:00)**, providing a data-backed roadmap for precisely timed marketing campaigns.
- **Pareto Concentration**: Validated that **20.94% of SKUs generate 78.93% of total revenue**, justifying a "Hero-first" inventory and promotion strategy.
- **Geographic Risk**: 82% of revenue is centralized in the UK, highlighting a significant geographic dependency and the urgent need for expansion into emerging European markets.
- **RFM Personas**: Segmented customers into 3 actionable tiers:
    - **Champions (17.7%)**: High-value loyalists for retention.
    - **At-Risk (39.1%)**: Former spenders requiring immediate **Win-back** efforts.
    - **Hibernating (43.2%)**: Low-engagement tiers for automated re-activation.

---

## 🛠 Integrated Analytical Framework (The 8-Step Flow)
This project transforms raw transactional data into prescriptive strategies through a structured methodology.

### **Phase 1: Operational Diagnostic Audit**
1. **Temporal Trend Analysis**: Identified peak revenue cycles across monthly and seasonal dimensions.
2. **The "Golden Window" Identification**: Pinpointed maximum sales velocity during **Mid-week** and **Mid-day**.
3. **Geographic Performance Evaluation**: Assessed market dominance (UK) and international expansion potential.
4. **Regional Concentration Analysis**: Quantified geographic revenue dependency to evaluate market risks.

### **Phase 2: Portfolio Resilience & Concentration**
5. **Product Velocity Analysis**: Classified top-performing products as core revenue drivers.
6. **Revenue Concentration (Pareto Principle)**: Confirmed that the top **20.94%** of products generate **78.93%** of revenue.

### **Phase 3: Behavioral Segmentation & Prescription**
7. **Customer Behavioral Clustering**: Applied **K-Means Clustering** ($K=3$ ) on log-normalized **RFM** metrics.
8. **Strategic Prescriptive Analytics**: Synthesized insights into targeted marketing roadmaps for VIP retention and churn recovery.

---

## 📊 Visual Analysis Audit (11 Strategic Assets)
The above 8-step framework is visualized through **11 key assets**, auto-exported via a Python pipeline.

### **Phase 1: Operational Diagnostic Audit**
*Focus: Identifying 'When' and 'Where' revenue velocity is highest.*

| 01. Monthly Trends | 02. Revenue by Day | 03. Revenue by Hour |
| :---: | :---: | :---: |
| ![Monthly](./images/01_monthly_revenue_trends.png) | ![Day](./images/02_revenue_by_day.png) | ![Hour](./images/03_revenue_by_hour.png) |
| **04. Top 15 Products** | **06. Top 10 Countries** | |
| ![Top Prod](./images/04_top_15_products.png) | ![Top Country](./images/06_top_10_countries.png) | |

### **Phase 2: Portfolio & Market Resilience**
*Focus: Quantifying business risk through concentration analysis.*

| 05. Product Pareto Curve | 07. Geographic Concentration |
| :---: | :---: |
| ![Pareto](./images/05_product_pareto_curve.png) | ![Geo](./images/07_geographic_concentration_curve.png) |

### **Phase 3: Behavioral Intelligence & ML Clustering**
*Focus: Data preprocessing and K-Means segmentation for persona building.*

| 08. RFM Raw Distribution | 09. Log Transformation |
| :---: | :---: |
| ![Raw Dist](./images/08_rfm_raw_distribution.png) | ![Log Trans](./images/09_rfm_log_transformation.png) |
| **10. K-Means Elbow Method** | **11. 3D RFM Clusters** |
| ![Elbow](./images/10_kmeans_elbow_method.png) | ![3D Clusters](./images/11_interactive_3d_rfm_clusters.png) |

---

## 📊 Operational Audit Dashboard

_Designed to bridge the gap between high-level KPIs and daily operational execution._

<p align="center">
  <a href="powerbi/Revenue_Strategy_Audit_v1.0.png">
    <img src="powerbi/Revenue_Strategy_Audit_v1.0.png" alt="Dashboard Preview" width="800">
  </a>
  <br>
  <em>Click for high-resolution PNG view</em>
</p>

<p align="center">
  <a href="powerbi/Revenue_Strategy_Audit_v1.0.pbix">
    <img src="https://img.shields.io/badge/Download-PBIX%20File-yellow?style=for-the-badge&logo=powerbi&logoColor=white">
  </a>
  <a href="powerbi/Revenue_Strategy_Audit_v1.0.pdf">
    <img src="https://img.shields.io/badge/Download-PDF%20Report-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white">
  </a>
</p>

#### **Key Functional Modules:**
- **Strategic Market Audit**: Visualized the **82% UK dependency** and identified the **Netherlands** as a high-value B2B hub **(AOV: £3,036)** versus scalable B2C growth corridors in **Germany/France**.
- **Precision Churn Monitor**: Leveraged a data-backed **44-day recency threshold** to categorize the **39.1% At-Risk segment**, enabling targeted win-back automation.    
- **Revenue Velocity Tracker**: Highlighted the **"Golden Window" (Tue-Thu, 10-15h)** to align marketing efforts with peak organic engagement.

#### **Strategic Prescriptions (Actionable Insights)**
The dashboard transcends simple reporting by providing a "Next Steps" framework for stakeholders:
* **Retention Strategy**: Deploy personalized win-back offers for the **At-Risk** segment specifically within the **44-day window** to prevent permanent churn.
* **Market Expansion**: Allocate **15% of the marketing budget** to high-AOV European B2B clusters to diversify geographic risk.
* **Operational Efficiency**: Schedule weekly promotional pushes during the identified **Mid-week/Mid-day peak** to maximize organic engagement and ROI.

---

## 🧰 Tech Stack & Methodology
- **Analytics**: RFM Analysis, Pareto/Concentration, Time-series Decomposition.
- **Machine Learning**: Scikit-learn (K-Means Clustering, StandardScaler, Elbow Metrics).
- **Processing & Visualization**: Python (Pandas, Numpy), Plotly 3D, Seaborn, Power BI.

---

## 🚀 How to Run
1. **Clone the Repo**:
   ```bash
   git clone [https://github.com/Hyeri-Jerrie-Kim/ecommerce-sales-analysis.git](https://github.com/Hyeri-Jerrie-Kim/ecommerce-sales-analysis.git)
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Analysis:
   Execute `notebooks/ecommerce_analysis.ipynb` to regenerate all insights and 11 visual assets.

## 🔗 Author & Links

Hyeri Kim — 🌐 [LinkedIn](https://linkedin.com/in/hyerikim-ds) | 📧 [Hyeri Kim](mailto:hyeri5524@gmail.com) 


