#!/usr/bin/env python
# coding: utf-8

# # 🛒 E-Commerce Sales Analysis
# 
# This Jupyter Notebook demonstrates an **end-to-end analysis** of a real-world e-commerce dataset. It covers **data cleaning, exploratory analysis, and visualizations** using powerful Python libraries such as **Pandas, Matplotlib, Seaborn, and Plotly**.
# 
# In this project, I explore customer behavior, seasonal trends, and regional sales performance to uncover actionable insights. This analysis not only demonstrates my technical proficiency in handling and visualizing data but also illustrates my ability to extract meaningful business insights that can drive decision-making.
# 
# 
# ---
# 
# ## 📥 1. Load Dataset & Initial Exploration
# 
# ### **🔹 Step 1: Import Required Libraries**

# In[2]:


# Install Necessary Modules
get_ipython().system('pip install wordcloud')


# In[3]:


# Import Necessary Libraries
import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tabulate import tabulate
from wordcloud import WordCloud


# ### **🔹 Step 2: Load and Inspect the Raw Data**
# Let's load the dataset and perform an initial exploration to understand its structure and content..**
# 

# In[5]:


# Load Dataset
file_path = '../datasets/ecommerce_data.csv'
raw_data = pd.read_csv(file_path, encoding='ISO-8859-1')


# In[6]:


# Display dataset information and basic statistics
print("Raw Data Information:")
raw_data.info()

print("\nDescriptive Statistics:")
print(raw_data.describe())

print("\nFirst few rows of raw data:")
print(raw_data.head())


# ### **🔹 Step 3: Explore Data**
# - **Check unique values in categorical columns** to understand product distribution.
# - **Detect negative values in Quantity & UnitPrice**, which may indicate returns.
# 

# In[8]:


# Explore categorical columns: Country and Product Descriptions
print("\nNumber of unique countries:", raw_data['Country'].nunique())
print("Number of unique products:", raw_data['Description'].nunique())


# In[9]:


# Summary for numerical columns 'Quantity' and 'UnitPrice'
print("\nSummary of 'Quantity' and 'UnitPrice':")
print(raw_data[['Quantity', 'UnitPrice']].describe())


# #### 📌 Key Findings from the Data Exploration
# 
# - **Dataset Size:** 541,909 transactions with 8 columns.
# - **Key Columns:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.
# - **Unique Values:**
#   - **Countries:** 38
#   - **Products:** 4,223
# 
# **Observations:**
# - **Negative Values:**  
#   - Some negative values appear in **Quantity** and **UnitPrice**.  
#   - *Interpretation:* In retail datasets, negative quantities typically represent **returns/refunds**. Negative unit prices may be tied to such adjustments.
# 
# ---

# ## 🧹 2. Data Cleaning
# 
# ### **🔹 Step 1: Handling Missing Values & Incomplete Records**
# We start by checking for missing values and flagging incomplete rows (e.g., missing descriptions, zero prices, or missing customer IDs).
# 

# In[12]:


# Calculate and display the percentage of missing values
missing_percentage = raw_data.isnull().mean() * 100
print("\nPercentage of missing values in raw data:")
print(missing_percentage)


# In[13]:


# Identify rows with missing 'Description'
missing_description = raw_data[raw_data['Description'].isnull()]
print("\nRows with missing descriptions:")
print(missing_description.head())
print("\nSummary of rows with missing descriptions (for 'Quantity', 'UnitPrice', 'CustomerID'):")
print(missing_description[['Quantity', 'UnitPrice', 'CustomerID']].describe())


# In[14]:


# Check additional patterns for missing descriptions
print("\nNumber of rows with missing 'CustomerID' among missing descriptions:",
      missing_description['CustomerID'].isnull().sum())
print("\nDistribution of 'UnitPrice' for missing descriptions:")
print(missing_description['UnitPrice'].value_counts())


# In[15]:


# Flag incomplete rows (missing Description, UnitPrice == 0, or missing CustomerID)
raw_data['IsIncomplete'] = raw_data['Description'].isnull() | (raw_data['UnitPrice'] == 0.0) | raw_data['CustomerID'].isnull()
incomplete_data = raw_data[raw_data['IsIncomplete']]
print("\nNumber of incomplete rows identified:", len(incomplete_data))
print("Preview of incomplete rows:")
print(incomplete_data.head())


# #### 📌 Key Findings
# - **Missing Descriptions:**  
#   - Approximately 0.27% of rows have missing product descriptions.  
#   - *Interpretation:* These rows often also lack `UnitPrice` and `CustomerID`, suggesting they may be **incomplete transactions**, **promotional entries**, or **non-product records**.
#  
# - **Missing Values:**  
#   - The `CustomerID` is missing in about **24.93%** of transactions, which can occur with guest checkouts in online retail.
# - **Incomplete Transactions:**  
#   - Approximately 135,120 rows are flagged as incomplete due to missing values in key fields.  
#   - *Action:* These rows were marked for further review.

# ### **🔹 Step 2: Normalizing and Inspecting Text Data**
# 
# To process product descriptions consistently, we normalize text and inspect the least common entries using a WordClou.
# 

# In[18]:


# Normalize 'Description' to lowercase (replace missing values with an empty string)
raw_data['NormalizedDescription'] = raw_data['Description'].fillna("").str.lower()

# Count occurrences of each unique description
description_counts = raw_data['NormalizedDescription'].value_counts()


# In[19]:


# Display the least common descriptions in a table
table_data = pd.DataFrame({
    'Description': description_counts.tail(10).index,
    'Frequency': description_counts.tail(10).values
})
print("\nLeast Common Descriptions:")
print(tabulate(table_data, headers='keys', tablefmt='grid'))


# In[20]:


# Generate Word Cloud from least common descriptions for visual inspection
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(description_counts.tail(50).index))
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.title('Least Common Descriptions (Word Cloud)')
plt.show()


# In[21]:


# Analyze description lengths to identify anomalies
raw_data['DescriptionLength'] = raw_data['NormalizedDescription'].str.len()
shortest_descriptions = raw_data.sort_values(by='DescriptionLength').head(20)
longest_descriptions = raw_data.sort_values(by='DescriptionLength', ascending=False).head(20)
print("\nShortest Descriptions (with InvoiceNo & StockCode):")
print(shortest_descriptions[['Description', 'InvoiceNo', 'StockCode']])
print("\nLongest Descriptions (with InvoiceNo & StockCode):")
print(longest_descriptions[['Description', 'InvoiceNo', 'StockCode']])


# ### **🔹 Step 3: Removing Erroneous or Irrelevant Data**
# 
# We now identify and remove placeholder descriptions, very short descriptions, and cancelled transaction.
# 

# #### 📌 Identifying Placeholder Descriptions
# To ensure data quality, we identified descriptions that might be **system-generated placeholders** or **erroneous entries**. Using the **WordCloud visualization**, we detected patterns in rare descriptions, revealing that some descriptions contain generic terms like `test`, `sample`, `unknown`, `barcode`, `damage`, `lost`, and similar words. These words suggest:
# 
# - **Test Entries:** Data created for system testing (`test`, `sample`).
# - **Incomplete or Faulty Records:** Rows where descriptions indicate missing or incorrect data (`unknown`, `barcode`, `?`).
# - **Damaged or Lost Products:** Indications of stock issues (`damage`, `broken`, `lost`, `thrown`).
# - **Mislabeling or Wrong Entries:** Cases where the wrong product might have been logged (`wrong`, `wrongly`).
# 
# To flag these descriptions, we applied the following filtering logic:
# ```python
# placeholder_descriptions = data[
#     data['NormalizedDescription'].str.contains(
#         'test|sample|unknown|placeholder|barcode|\?|damage|wrong|wrongly|lost|broken|thrown', 
#         na=False, regex=True
#     )
# ]
# ```
# This ensures that any potentially unreliable or invalid product descriptions are **identified for further inspection or removal** before performing deeper analysis.
# 

# In[24]:


# Identify placeholder descriptions using specific keywords
placeholder_keywords = 'test|sample|unknown|placeholder|barcode|\?|damage|wrong|wrongly|lost|broken|thrown'
placeholder_descriptions = raw_data[raw_data['NormalizedDescription'].str.contains(placeholder_keywords, na=False, regex=True)]
print("\nIdentified Placeholder Descriptions (for further inspection):")
print(placeholder_descriptions[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'CustomerID']])


# In[25]:


# Remove placeholder descriptions from the dataset
clean_data = raw_data.drop(placeholder_descriptions.index)
print(f"\nRemoved {len(placeholder_descriptions)} placeholder descriptions.")


# In[26]:


# Identify and remove short descriptions (length <= 3)
short_descriptions = clean_data[clean_data['NormalizedDescription'].str.len() <= 3]
print("\nShort Descriptions identified for removal:")
print(short_descriptions[['Description', 'InvoiceNo', 'StockCode']])
clean_data = clean_data.drop(short_descriptions.index)
print(f"Removed {len(short_descriptions)} short descriptions.")


# In[27]:


# Identify and remove cancelled transactions (InvoiceNo starting with 'C')
cancelled_transactions = clean_data[clean_data['InvoiceNo'].str.startswith('C')]
print("\nCancelled Transactions identified for removal:")
print(cancelled_transactions)
clean_data = clean_data.drop(cancelled_transactions.index)
print(f"Removed {len(cancelled_transactions)} cancelled transactions.")


# **📌 Note**: While duplicates were dropped to avoid overcounting, in some contexts (e.g., basket analysis) retaining duplicates might provide additional insights.

# In[29]:


# Save all removed transactions for review
dropped_transactions = pd.concat([placeholder_descriptions, short_descriptions, cancelled_transactions])
dropped_transactions.to_csv('../datasets/dropped_transactions.csv', index=False)
print(f"\n{len(dropped_transactions)} transactions saved to 'dropped_transactions.csv' for further analysis.")


# In[30]:


# Verify the cleanup by checking the number of remaining transactions
print(f"\nRemaining transactions after cleanup: {len(clean_data)}")


# ### **🔹 Step 4: Additional Data Cleaning**
# 
# We now handle duplicate transactions, negative quantities, and zero unit prices. Finally, we convert the date column for time series analysi.
# 

# In[32]:


# Detect potential duplicate transactions based on key columns
duplicate_transactions = clean_data.duplicated(subset=['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice'], keep=False)
print("\nPotential Duplicate Transactions (may include valid multiple-item orders):")
print(clean_data[duplicate_transactions])


# Note: This is not exactly the duplicates that we are looking for since customers might ordered several items in one order.

# In[34]:


# Identify true duplicates by counting (InvoiceNo, StockCode) occurrences
duplicate_counts = clean_data.groupby(['InvoiceNo', 'StockCode']).size()
true_duplicates = duplicate_counts[duplicate_counts > 1].reset_index()
true_duplicates.columns = ['InvoiceNo', 'StockCode', 'Count']
print("\nDuplicate Transactions (Same InvoiceNo & StockCode appearing multiple times):")
print(true_duplicates.head(20))
print(f"Total duplicate (InvoiceNo, StockCode) entries found: {len(true_duplicates)}")


# In[35]:


# Drop duplicates (keeping the first occurrence)
clean_data = clean_data.drop_duplicates(subset=['InvoiceNo', 'StockCode'], keep='first')
print(f"\nRemaining transactions after dropping duplicates: {len(clean_data)}")


# In[36]:


# Handle Negative Quantities
negative_quantity_transactions = clean_data[clean_data['Quantity'] < 0]
print("\nNegative Quantity Transactions:")
print(negative_quantity_transactions.head(20))
print(f"Total negative quantity transactions found: {len(negative_quantity_transactions)}")


# In[37]:


# Handle Zero Unit Price Transactions
zero_price_transactions = clean_data[clean_data['UnitPrice'] == 0.0]
print("\nTransactions with Zero Unit Price:")
print(zero_price_transactions.head(20))
print(f"Total transactions with zero unit price found: {len(zero_price_transactions)}")


# In[38]:


# Remove zero-price transactions from the dataset
clean_data = clean_data[clean_data['UnitPrice'] > 0]
print(f"\nRemaining transactions after removing zero-price transactions: {len(clean_data)}")


# In[39]:


# Check for missing or incorrect country names
print("\nChecking for missing or incorrect country names...")
print("Missing country values count:", clean_data['Country'].isnull().sum())
print("Unique country names:", clean_data['Country'].unique())


# In[40]:


# Convert 'InvoiceDate' to datetime format and set as index for time-series analysis
clean_data['InvoiceDate'] = pd.to_datetime(clean_data['InvoiceDate'], errors='coerce')
clean_data.set_index('InvoiceDate', inplace=True)
print("\nConverted 'InvoiceDate' to datetime format and set as index.")


# In[41]:


# Define the output file path
cleaned_file_path = "../datasets/cleaned_ecommerce_data.csv"

# Save the cleaned data
clean_data.to_csv(cleaned_file_path, index=False, encoding="utf-8")

print(f" Cleaned dataset saved successfully: {cleaned_file_path}")


# ---
# 
# ## 📊 3. Data Analysis & Visualization
# 
# ### **🔹 Step 1: Sales Analysis by Country**
# 
# We calculate the total sales and visualize the top countries by ales.
# 

# In[43]:


# Aggregate Total Sales by Country
clean_data['TotalSales'] = clean_data['Quantity'] * clean_data['UnitPrice']


# In[44]:


# Aggregate Total Sales by Country and sort descending
country_sales = clean_data.groupby('Country')['TotalSales'].sum().reset_index()
country_sales = country_sales.sort_values(by='TotalSales', ascending=False)
print("\nTotal Sales by Country (Top 10):")
print(country_sales.head(10))


# In[45]:


# Create a bar plot for Total Sales by Country
fig_country_sales = px.bar(country_sales.head(10),
                           x='Country', y='TotalSales',
                           title='Total Sales by Country',
                           color='TotalSales',
                           text=country_sales.head(10)['TotalSales'].apply(lambda x: f"${x:,.0f}"),
                           color_continuous_scale='viridis')
fig_country_sales.update_traces(textposition='outside')
fig_country_sales.update_layout(xaxis_title="Country",
                                yaxis_title="Total Sales",
                                bargap=0.2,
                                height=600)
fig_country_sales.show()


# **📌 Country-Level Sales:**  
#   - The **United Kingdom** dominates with total sales of approximately **$8.96M**.
#   - Other top countries include **Netherlands**, **EIRE**, **Germany**, and **France**.

# ### **🔹 Step 2: Top 10 Products by Total Sales**

# In[48]:


# Top 10 Products by Total Sales 
if 'TotalSales' in clean_data.columns and 'Description' in clean_data.columns:
    top_products = clean_data.groupby('Description')['TotalSales'].sum().nlargest(10).reset_index()

    # Fore Seaborn visualizations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='TotalSales', y='Description', data=top_products)
    plt.title('Top 10 Products by Total Sales (Seaborn)')
    plt.xlabel('Total Sales')
    plt.ylabel('Product Name')
    plt.show()

    # For interactive Plotly visualizations
    fig_bar = px.bar(top_products, x='TotalSales', y='Description',
                     title='Top 10 Products by Total Sales (Plotly)', orientation='h')
    fig_bar.update_layout(xaxis_title='Total Sales', yaxis_title='Product Name')
    fig_bar.update_yaxes(autorange='reversed')  # To show them in descending order
    fig_bar.show()


# **📌 Key Insight:**
# - focusing on the top 10 products by **TotalSales** helps in identifying best-selling items for targeted marketing an inventory optimization.

# ### **🔹 Step 3: Time Series Analysis**
# 
# We explore yearly, monthly, and weekly sales trend.
# 

# In[51]:


# Yearly Sales Trend using resampling
yearly_sales = clean_data['TotalSales'].resample('Y').sum().reset_index()
print("\nYearly Sales Trend:")
print(yearly_sales)


# **📌 Yearly Trend:**
#   - Sales grow dramatically from **December 2010 (812k)** to **December 2011 (9.77M)**.  
#   - *Note:* Since 2010 data only covers December, a monthly breakdown is necessary.
#     
# In order to understand trends, we should compare month-to-month growth within 2011.

# In[53]:


# Analyze Monthly Sales Trend
monthly_sales = clean_data['TotalSales'].resample('M').sum().asfreq('M', fill_value=0).reset_index()
print("\nMonthly Sales Trend (first few records):")
print(monthly_sales.head())

fig_monthly_sales = px.line(monthly_sales, x='InvoiceDate', y='TotalSales',
                            title='Monthly Sales Trend', markers=True)
fig_monthly_sales.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)
fig_monthly_sales.show()


# **📌 Monthly Trend:**
# - December 2010 (812k) shows a peak, followed by a dip in January 2011 (686k) (possibly due to a post-holiday slowdown)
# - March 2011 (713k) had a sales recovery, suggesting a potential seasonal trend
# - *Seasonal Insight:* **Holiday demand** significantly boosts December sales.

# In[55]:


# Analyze Weekly Sales Trend
weekly_sales = clean_data['TotalSales'].resample('W').sum().asfreq('W', fill_value=0).reset_index()
print("\nWeekly Sales Trend (first few records):")
print(weekly_sales.head())

fig_weekly_sales = px.line(weekly_sales, x='InvoiceDate', y='TotalSales',
                           title='Weekly Sales Trend', markers=True,
                           color_discrete_sequence=['orange'])
fig_weekly_sales.show()


# **📌 Weekly Trend:** 
# - Sales peaked in the week of **December 12, 2010**.  
# - Zero sales in the first week of January may indicate a data gap or store closure.

# ### **🔹 Step 4: Sales by Day & Hour**
# 
# Next, we analyze sales by the day of the week and by the hour of the da.
# 

# In[58]:


# Sales by Day of the Week
clean_data['DayOfWeek'] = clean_data.index.day_name()
daywise_sales = clean_data.groupby('DayOfWeek')['TotalSales'].sum().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
print("\nSales by Day of the Week:")
print(daywise_sales)

fig_daywise_sales = px.bar(daywise_sales, x='DayOfWeek', y='TotalSales',
                           title='Sales by Day of the Week',
                           color='TotalSales', color_continuous_scale='reds')
fig_daywise_sales.show()


# **📌 Sales by Day of the Week:**
# - **Tuesday** and **Thursday** record the highest sales (over 2.1M each).  
# - **Sunday** shows the lowest sales (~797k).  
# - *Observation:* The absence of Saturday sales (NaN) suggests no transactions—possibly due to store closure or data collection issues.

# In[60]:


# Sales by Hour of the Day
clean_data['Hour'] = clean_data.index.hour
hourly_sales = clean_data.groupby('Hour')['TotalSales'].sum().reset_index()
print("\nSales by Hour:")
print(hourly_sales)

fig_hourly_sales = px.bar(hourly_sales, x='Hour', y='TotalSales',
                          title='Sales by Hour of the Day',
                          color='TotalSales', color_continuous_scale='reds')
fig_hourly_sales.show()


# **📌 Hourly Sales:**
#   - Peak transaction hours are between **10 AM and 3 PM**.  
#   - Sales drop sharply after 5 PM, aligning with standard business hours.

# ### **🔹 Step 5: Distribution & Correlation Analysis**
# 
# Let's inspect the unit price distribution and see how key variables correlat.
# 

# In[63]:


# Unit Price Distribution using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x=clean_data['UnitPrice'])
plt.title('Unit Price Distribution')
plt.show()


# In[64]:


# Correlation Heatmap for key variables
plt.figure(figsize=(8, 5))
sns.heatmap(clean_data[['Quantity', 'UnitPrice', 'TotalSales']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# **📌 Correlation Insights:**  
#   - **Quantity vs. TotalSales:** Strong positive correlation (0.91) indicates that increasing the number of items sold boosts revenue.
#   - **UnitPrice vs. TotalSales:** Little to no correlation (around -0.2 to 0.0) suggests that raising prices does not necessarily increase revenue.
# 
# **📌 Business Recommendations:**  
#   - **Volume-Based Promotions:**  
#     - 📦 **Bundle Discounts:** Offer product bundles at a reduced price.
#     - 🔁 **Subscription & Loyalty Programs:** Reward frequent buyers with exclusive deals.
#     - 🤝 **Cross-Selling Discounts:** Suggest related products when bought together.
# - *Additional Note:* Analyzing the impact of **return transactions (negative quantities)** separately could provide further actionable insights.
# 
# ---

# ## Save & Convert Notebook to Python Script
# To ensure that our Jupyter Notebook (`.ipynb`) is always synchronized with a Python script (`.py`), we will **automatically convert** the notebook to a Python script at the end of execution. This ensures consistency when pushing updates to GitHub.

# In[67]:


# Define notebook and script filenames
notebook_name = "ecommerce_analysis.ipynb"  # Update this if the notebook name changes
script_name = "ecommerce_analysis.py"

# Convert the Jupyter Notebook to a Python script
get_ipython().system(f'jupyter nbconvert --to script {notebook_name}')

# Move the Python script to the 'scripts' folder
destination_path = f"../scripts/{script_name}"
shutil.move(script_name, destination_path)

# Print confirmation message
print(f" Notebook '{notebook_name}' successfully converted and saved as '{destination_path}'")


# ## 🛠️ Restoring `InvoiceDate` as a Column
# 
# ### 📌 Why Reset `InvoiceDate`?
# During the data analysis process, `InvoiceDate` was set as an **index** for easier time-series analysis. However, in Power BI, we need `InvoiceDate` as a **regular column** to create visualizations based on time (e.g., Monthly & Weekly Sales Trend as a column.")
# 

# In[115]:


# Reset the index to restore 'InvoiceDate' as a column
clean_data.reset_index(inplace=True)

# Save the cleaned dataset again
clean_data.to_csv("../datasets/cleaned_ecommerce_data.csv", index=False, encoding="utf-8")

print("Cleaned dataset saved with 'InvoiceDate' as a column.")


# ---
# 
# ## 💡 Summary of Insights
# 
# ### 1. Data Quality & Cleaning
# - **Incomplete Transactions:**  
#   Over **135K transactions** were flagged as incomplete due to missing descriptions, zero unit prices, or missing CustomerID (common in guest checkouts). Negative values generally represent returns/refunds.
# - **Data Cleaning Measures:**  
#   Placeholder entries (e.g., “?”, “damages”, “samples”) and very short descriptions were removed to improve data quality. Cancelled transactions and duplicates were dropped, resulting in a refined dataset for analysis.
# 
# ### 2. Regional Performance & Top Products Analysis
# - **Regional Sales Dominance:**  
#   The **United Kingdom** leads with total sales of approximately **$8.96M**, with other notable contributions from the Netherlands, EIRE, Germany, and France.
# - **Top Products Insight:**  
#   Visualizations of the top 10 products by total sales highlight the best-selling items. This analysis can guide targeted marketing and inventory optimization strategies by focusing on high-performing products.
# 
# ### 3. Time Series & Transaction Patterns
# - **Yearly Trends:**  
#   Sales grow dramatically from **December 2010 (812k)** to **December 2011 (9.77M)**. Note that 2010 data only covers December.
# - **Monthly Trends:**  
#   A seasonal pattern emerges with a December peak, a post-holiday dip in January, and a recovery in March.
# - **Weekly & Daily Patterns:**  
#   Weekly trends show a peak during the week of December 12, 2010, while day-of-week analysis indicates highest sales on Tuesday and Thursday, and the lowest on Sunday. The absence of Saturday sales suggests either store closure or data collection issues.
# - **Hourly Trends:**  
#   Transactions peak between **10 AM and 3 PM**, aligning with standard business hours.
# 
# ### 4. Correlation & Business Strategy
# - **Key Correlations:**  
#   A strong correlation (0.91) between **Quantity** and **Total Sales** indicates that increasing the number of items sold has a direct impact on revenue. In contrast, **Unit Price** shows little to no correlation with Total Sales.
# - **Strategic Implications:**  
#   Focus on volume-based promotions, such as bundle discounts, subscription/loyalty programs, and cross-selling offers, rather than relying solely on price adjustments to drive revenue growth.
#   
# ---
# 
# **Overall, the analysis supports a strategy focused on boosting sales volume and optimizing operational efficiency. The insights across regional performance, top product identification, temporal trends, and correlation analysis provide a solid basis for targeted marketing, staffing, and inventory management decisions.** 😃
# management decisions.** 😃
# 

# In[ ]:




