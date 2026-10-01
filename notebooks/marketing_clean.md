### MARKDOWN
# Marketing Campaign Performance Analysis Using Python

## Project 3: Python Exploratory Data Analysis

This notebook analyzes a marketing campaign performance dataset using Python. The goal is to clean the data, explore campaign performance, identify high-performing channels and segments, and produce business recommendations for marketing budget optimization.

## Business Problem

The marketing team wants to understand which campaigns, channels, customer segments, and target audiences deliver the strongest campaign performance. The analysis focuses on ROI, conversion rate, clicks, impressions, engagement, acquisition cost, and cost efficiency.

### MARKDOWN
## 1. Import Libraries

### CODE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# Use a clean default chart size
plt.rcParams["figure.figsize"] = (10, 5)

### MARKDOWN
## 2. Load Dataset

Place the raw Kaggle CSV file in:

`../data/raw/marketing_campaign_dataset.csv`

### CODE
RAW_DATA_PATH = Path("../data/raw/marketing_campaign_dataset.csv")

df = pd.read_csv(RAW_DATA_PATH)
df.head()

### MARKDOWN
## 3. Initial Dataset Inspection

### CODE
print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
display(df.dtypes)

print("\nMissing values:")
display(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

### MARKDOWN
## 4. Data Cleaning

Cleaning actions:
- Standardize column names
- Convert date column to datetime
- Convert acquisition cost from currency text to numeric
- Extract numeric campaign duration
- Create time-based columns
- Create campaign performance metrics

### CODE
cleaned_df = df.copy()

# Standardize column names
cleaned_df.columns = (
    cleaned_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# Convert date
cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")

# Convert acquisition cost from "$16,174.00" to numeric
cleaned_df["acquisition_cost"] = (
    cleaned_df["acquisition_cost"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# Extract numeric duration
cleaned_df["duration_days"] = cleaned_df["duration"].str.extract(r"(\d+)").astype(int)

# Time features
cleaned_df["year"] = cleaned_df["date"].dt.year
cleaned_df["month_no"] = cleaned_df["date"].dt.month
cleaned_df["month_name"] = cleaned_df["date"].dt.strftime("%b")
cleaned_df["year_month"] = cleaned_df["date"].dt.to_period("M").astype(str)

# Performance metrics
cleaned_df["ctr"] = cleaned_df["clicks"] / cleaned_df["impressions"]
cleaned_df["estimated_conversions"] = cleaned_df["clicks"] * cleaned_df["conversion_rate"]
cleaned_df["cost_per_click"] = cleaned_df["acquisition_cost"] / cleaned_df["clicks"].replace(0, np.nan)
cleaned_df["cost_per_conversion"] = cleaned_df["acquisition_cost"] / cleaned_df["estimated_conversions"].replace(0, np.nan)

# ROI-derived estimates
# Note: Treating ROI as a multiplier/score from the dataset for analytical comparison.
cleaned_df["estimated_revenue_from_roi"] = cleaned_df["acquisition_cost"] * cleaned_df["roi"]
cleaned_df["estimated_profit_from_roi"] = cleaned_df["estimated_revenue_from_roi"] - cleaned_df["acquisition_cost"]

# Bands
cleaned_df["roi_band"] = pd.cut(
    cleaned_df["roi"],
    bins=[-np.inf, 2, 4, np.inf],
    labels=["Low ROI (<2)", "Medium ROI (2-4)", "High ROI (4+)"]
)

cleaned_df["engagement_band"] = pd.cut(
    cleaned_df["engagement_score"],
    bins=[0, 3, 7, 10],
    labels=["Low Engagement", "Medium Engagement", "High Engagement"],
    include_lowest=True
)

cleaned_df.head()

### MARKDOWN
## 5. Data Quality Validation After Cleaning

### CODE
quality_summary = pd.DataFrame({
    "metric": [
        "Rows",
        "Columns",
        "Missing values",
        "Duplicate rows",
        "Invalid dates",
        "Date start",
        "Date end"
    ],
    "value": [
        cleaned_df.shape[0],
        cleaned_df.shape[1],
        cleaned_df.isna().sum().sum(),
        cleaned_df.duplicated().sum(),
        cleaned_df["date"].isna().sum(),
        cleaned_df["date"].min(),
        cleaned_df["date"].max()
    ]
})

quality_summary

### MARKDOWN
## 6. Save Cleaned Dataset

### CODE
CLEANED_DATA_PATH = Path("../data/cleaned/marketing_campaign_cleaned.csv")
CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")

### MARKDOWN
## 7. KPI Summary

### CODE
kpi_summary = pd.DataFrame({
    "KPI": [
        "Total Campaigns",
        "Total Acquisition Cost",
        "Average Acquisition Cost",
        "Average ROI",
        "Average Conversion Rate",
        "Total Clicks",
        "Total Impressions",
        "Overall CTR",
        "Average Engagement Score",
        "Estimated Conversions",
        "Average CPC",
        "Average Cost per Conversion"
    ],
    "Value": [
        cleaned_df["campaign_id"].nunique(),
        cleaned_df["acquisition_cost"].sum(),
        cleaned_df["acquisition_cost"].mean(),
        cleaned_df["roi"].mean(),
        cleaned_df["conversion_rate"].mean(),
        cleaned_df["clicks"].sum(),
        cleaned_df["impressions"].sum(),
        cleaned_df["clicks"].sum() / cleaned_df["impressions"].sum(),
        cleaned_df["engagement_score"].mean(),
        cleaned_df["estimated_conversions"].sum(),
        cleaned_df["acquisition_cost"].sum() / cleaned_df["clicks"].sum(),
        cleaned_df["acquisition_cost"].sum() / cleaned_df["estimated_conversions"].sum()
    ]
})

kpi_summary

### MARKDOWN
## 8. Reusable Summary Function

### CODE
def performance_summary(group_col):
    summary = (
        cleaned_df.groupby(group_col)
        .agg(
            campaigns=("campaign_id", "count"),
            total_cost=("acquisition_cost", "sum"),
            avg_roi=("roi", "mean"),
            avg_conversion_rate=("conversion_rate", "mean"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            avg_engagement=("engagement_score", "mean"),
            estimated_conversions=("estimated_conversions", "sum")
        )
        .reset_index()
    )

    summary["ctr"] = summary["clicks"] / summary["impressions"]
    summary["cost_per_click"] = summary["total_cost"] / summary["clicks"]
    summary["cost_per_conversion"] = summary["total_cost"] / summary["estimated_conversions"]

    return summary.sort_values("avg_roi", ascending=False)

channel_summary = performance_summary("channel_used")
channel_summary

### MARKDOWN
## 9. Channel Performance Analysis

### CODE
channel_summary = performance_summary("channel_used")
display(channel_summary)

plt.figure()
plt.bar(channel_summary["channel_used"], channel_summary["avg_roi"])
plt.title("Average ROI by Marketing Channel")
plt.xlabel("Channel")
plt.ylabel("Average ROI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../visuals/avg_roi_by_channel.png", dpi=150)
plt.show()

### MARKDOWN
## 10. Campaign Type Performance

### CODE
campaign_type_summary = performance_summary("campaign_type")
display(campaign_type_summary)

plt.figure()
plt.bar(campaign_type_summary["campaign_type"], campaign_type_summary["avg_conversion_rate"] * 100)
plt.title("Average Conversion Rate by Campaign Type")
plt.xlabel("Campaign type")
plt.ylabel("Average conversion rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../visuals/conversion_rate_by_campaign_type.png", dpi=150)
plt.show()

### MARKDOWN
## 11. Customer Segment Performance

### CODE
segment_summary = performance_summary("customer_segment")
display(segment_summary)

plt.figure()
plt.barh(segment_summary["customer_segment"], segment_summary["cost_per_conversion"])
plt.title("Cost per Conversion by Customer Segment")
plt.xlabel("Cost per conversion")
plt.ylabel("Customer segment")
plt.tight_layout()
plt.savefig("../visuals/cost_per_conversion_by_segment.png", dpi=150)
plt.show()

### MARKDOWN
## 12. Monthly Trend Analysis

### CODE
monthly_summary = (
    cleaned_df.groupby("year_month")
    .agg(
        campaigns=("campaign_id", "count"),
        total_cost=("acquisition_cost", "sum"),
        avg_roi=("roi", "mean"),
        avg_conversion_rate=("conversion_rate", "mean"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        estimated_conversions=("estimated_conversions", "sum")
    )
    .reset_index()
)

monthly_summary["ctr"] = monthly_summary["clicks"] / monthly_summary["impressions"]
monthly_summary["cost_per_conversion"] = monthly_summary["total_cost"] / monthly_summary["estimated_conversions"]

display(monthly_summary)

plt.figure()
plt.plot(monthly_summary["year_month"], monthly_summary["avg_roi"], marker="o")
plt.title("Monthly Average ROI Trend")
plt.xlabel("Month")
plt.ylabel("Average ROI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../visuals/monthly_avg_roi_trend.png", dpi=150)
plt.show()

### MARKDOWN
## 13. Top 10 Campaigns by ROI

### CODE
top_10_roi = cleaned_df.sort_values("roi", ascending=False).head(10)[[
    "campaign_id",
    "company",
    "campaign_type",
    "channel_used",
    "target_audience",
    "customer_segment",
    "acquisition_cost",
    "roi",
    "conversion_rate",
    "clicks",
    "impressions",
    "engagement_score",
    "date"
]]

top_10_roi

### MARKDOWN
## 14. Correlation Analysis

### CODE
numeric_cols = [
    "conversion_rate",
    "acquisition_cost",
    "roi",
    "clicks",
    "impressions",
    "ctr",
    "estimated_conversions",
    "cost_per_click",
    "cost_per_conversion",
    "engagement_score",
    "duration_days"
]

corr = cleaned_df[numeric_cols].corr()
corr

### CODE
plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Matrix of Campaign Metrics")
plt.tight_layout()
plt.savefig("../visuals/correlation_matrix.png", dpi=150)
plt.show()

### MARKDOWN
## 15. Business Insights

Update this section after reviewing the outputs.

Initial insight directions:
1. Compare ROI by channel.
2. Compare cost per conversion by customer segment.
3. Compare campaign types by conversion rate.
4. Identify monthly ROI patterns.
5. Identify high-performing campaigns.
6. Recommend budget allocation based on ROI and cost efficiency.

### MARKDOWN
## 16. Business Recommendations

Draft recommendations:
1. Increase investment in channels with stronger ROI and lower cost per conversion.
2. Review underperforming campaign types with lower conversion rates.
3. Use customer segment performance to improve audience targeting.
4. Monitor monthly performance trends to adjust campaign timing.
5. Evaluate top campaigns and replicate their channel, audience, and creative strategy.

