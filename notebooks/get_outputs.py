import pandas as pd
import numpy as np
import sqlite3
from scipy import stats
from pathlib import Path
import sys
import io

sys.stdout.reconfigure(encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

RAW_DATA_PATH = Path("../data/raw/marketing_campaign_dataset.csv")
df = pd.read_csv(RAW_DATA_PATH)

cleaned_df = df.copy()
cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
cleaned_df["acquisition_cost"] = cleaned_df["acquisition_cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")
cleaned_df["duration_days"] = cleaned_df["duration"].str.extract(r"(\d+)").astype(int)
cleaned_df["year_month"] = cleaned_df["date"].dt.to_period("M").astype(str)

cleaned_df["ctr"] = cleaned_df["clicks"] / cleaned_df["impressions"]
cleaned_df["estimated_conversions"] = cleaned_df["clicks"] * cleaned_df["conversion_rate"]
cleaned_df["cost_per_click"] = cleaned_df["acquisition_cost"] / cleaned_df["clicks"].replace(0, np.nan)
cleaned_df["cost_per_conversion"] = cleaned_df["acquisition_cost"] / cleaned_df["estimated_conversions"].replace(0, np.nan)
cleaned_df["estimated_revenue"] = cleaned_df["acquisition_cost"] * cleaned_df["roi"]
cleaned_df["estimated_profit"] = cleaned_df["estimated_revenue"] - cleaned_df["acquisition_cost"]

print("=========================================")
print("MAIN EDA NOTEBOOK OUTPUTS")
print("=========================================\n")

def performance_summary(group_col):
    summary = cleaned_df.groupby(group_col).agg(
        campaigns=("campaign_id", "count"),
        total_cost=("acquisition_cost", "sum"),
        avg_roi=("roi", "mean"),
        avg_conversion_rate=("conversion_rate", "mean"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        estimated_conversions=("estimated_conversions", "sum")
    ).reset_index()
    summary["ctr"] = summary["clicks"] / summary["impressions"]
    summary["cost_per_conversion"] = summary["total_cost"] / summary["estimated_conversions"]
    return summary.sort_values("avg_roi", ascending=False)

print("--- Channel Summary ---")
print(performance_summary("channel_used").to_string(index=False))

print("\n--- Campaign Type Summary ---")
print(performance_summary("campaign_type").to_string(index=False))

print("\n--- Customer Segment Summary ---")
print(performance_summary("customer_segment").to_string(index=False))

print("\n--- Monthly Summary ---")
monthly_summary = performance_summary("year_month").sort_values("year_month")
print(monthly_summary.to_string(index=False))

print("\n--- Top 10 Campaigns by ROI ---")
top_10 = cleaned_df.sort_values("roi", ascending=False).head(10)[["campaign_id", "company", "channel_used", "acquisition_cost", "roi", "estimated_profit"]]
print(top_10.to_string(index=False))

print("\n--- Correlation Matrix ---")
numeric_cols = ["conversion_rate", "acquisition_cost", "roi", "clicks", "impressions", "ctr", "cost_per_conversion"]
print(cleaned_df[numeric_cols].corr().to_string())


print("\n\n=========================================")
print("ADVANCEMENTS.PY OUTPUTS")
print("=========================================\n")

conn = sqlite3.connect(":memory:")
cleaned_df.to_sql("campaigns", conn, if_exists="replace", index=False)

print("--- SQL Channel Performance ---")
print(pd.read_sql_query("""
SELECT channel_used, COUNT(campaign_id) AS total_campaigns, ROUND(AVG(roi), 4) AS avg_roi, 
ROUND(AVG(conversion_rate) * 100, 2) AS avg_conversion_rate_pct, ROUND(SUM(acquisition_cost), 2) AS total_cost, 
ROUND(AVG(cost_per_conversion), 2) AS avg_cost_per_conversion, ROUND(SUM(clicks) * 1.0 / SUM(impressions), 4) AS overall_ctr
FROM campaigns GROUP BY channel_used ORDER BY avg_roi DESC;
""", conn).to_string(index=False))

print("\n--- SQL Segment Profitability ---")
print(pd.read_sql_query("""
SELECT customer_segment, COUNT(campaign_id) AS total_campaigns, ROUND(AVG(roi), 4) AS avg_roi, 
ROUND(SUM(estimated_profit), 2) AS total_estimated_profit, ROUND(AVG(cost_per_conversion), 2) AS avg_cac
FROM campaigns GROUP BY customer_segment ORDER BY total_estimated_profit DESC;
""", conn).to_string(index=False))

print("\n--- SQL Monthly ROI Trend ---")
print(pd.read_sql_query("""
SELECT SUBSTR(date, 1, 7) AS year_month, COUNT(campaign_id) AS campaigns, ROUND(AVG(roi), 4) AS avg_roi, 
ROUND(AVG(conversion_rate), 4) AS avg_conversion_rate
FROM campaigns GROUP BY year_month ORDER BY year_month ASC;
""", conn).to_string(index=False))

print("\n--- A/B / Welch T-Test Results ---")
fb = cleaned_df[cleaned_df["channel_used"] == "Facebook"]["roi"]
em = cleaned_df[cleaned_df["channel_used"] == "Email"]["roi"]
wb = cleaned_df[cleaned_df["channel_used"] == "Website"]["roi"]
inf = cleaned_df[cleaned_df["campaign_type"] == "Influencer"]["roi"]
sm = cleaned_df[cleaned_df["campaign_type"] == "Social Media"]["roi"]

print(f"Facebook vs Email ROI   : p-value = {stats.ttest_ind(fb, em, equal_var=False)[1]:.4f}")
print(f"Facebook vs Website ROI : p-value = {stats.ttest_ind(fb, wb, equal_var=False)[1]:.4f}")
print(f"Influencer vs Social    : p-value = {stats.ttest_ind(inf, sm, equal_var=False)[1]:.4f}")

print("\n--- LTV / CAC Ratio by Segment ---")
ltv_cac = cleaned_df.groupby("customer_segment").agg(avg_cac=("acquisition_cost", "mean"), avg_ltv=("estimated_revenue", "mean")).reset_index()
ltv_cac["ltv_cac_ratio"] = ltv_cac["avg_ltv"] / ltv_cac["avg_cac"]
ltv_cac["health"] = ltv_cac["ltv_cac_ratio"].apply(lambda x: "Healthy (>3)" if x > 3 else ("Borderline (1-3)" if x >= 1 else "Unsustainable (<1)"))
print(ltv_cac.sort_values("ltv_cac_ratio", ascending=False).to_string(index=False))
