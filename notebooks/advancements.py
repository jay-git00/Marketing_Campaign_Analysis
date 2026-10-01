"""
=====================================================================
MARKETING CAMPAIGN PROJECT - ADVANCED ADDITIONS
=====================================================================
Run this file AFTER running the main marketing_campaign_eda.ipynb
It adds three advanced capabilities to the existing project:
  1. SQLite Database Layer  (proves you know SQL)
  2. A/B Testing (T-Test)   (proves you know Statistics)
  3. LTV / CAC Analysis     (proves you know Business Finance)
=====================================================================
"""

import sys
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

plt.rcParams["figure.figsize"] = (10, 5)

# ---------------------------------------------------------------
# STEP 0: Load and clean the data (same as the main notebook)
# ---------------------------------------------------------------
RAW_DATA_PATH = Path("../data/raw/marketing_campaign_dataset.csv")
df = pd.read_csv(RAW_DATA_PATH)

cleaned_df = df.copy()
cleaned_df.columns = (
    cleaned_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)
cleaned_df["acquisition_cost"] = (
    cleaned_df["acquisition_cost"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)
cleaned_df["ctr"] = cleaned_df["clicks"] / cleaned_df["impressions"]
cleaned_df["estimated_conversions"] = cleaned_df["clicks"] * cleaned_df["conversion_rate"]
cleaned_df["cost_per_conversion"] = (
    cleaned_df["acquisition_cost"]
    / cleaned_df["estimated_conversions"].replace(0, np.nan)
)
cleaned_df["estimated_revenue"] = cleaned_df["acquisition_cost"] * cleaned_df["roi"]
cleaned_df["estimated_profit"] = cleaned_df["estimated_revenue"] - cleaned_df["acquisition_cost"]


# ===============================================================
# ADVANCEMENT 1: SQLite Database Layer
# ===============================================================
# WHY WE ARE DOING THIS:
# The original project loads data directly from a CSV file.
# In a real company (like Zomato), data NEVER lives in a CSV.
# It lives in a database (PostgreSQL, MySQL, etc.).
# By adding a SQLite layer, we simulate a real-world database workflow.
# We push the cleaned data INTO a database, and then pull it back out
# using SQL queries, EXACTLY like you would do at a real job.

print("=" * 60)
print("ADVANCEMENT 1: SQL Database Layer")
print("=" * 60)

# Create an in-memory SQLite database
conn = sqlite3.connect(":memory:")

# Push the cleaned DataFrame INTO the database as a table
# Think of this like: "CREATE TABLE campaigns AND INSERT all 200,000 rows"
cleaned_df.to_sql("campaigns", conn, if_exists="replace", index=False)

print("[OK] Pushed 200,000 records into SQLite database.")
print("[OK] Table name: 'campaigns'\n")

# --- SQL Query 1: Channel Performance Summary ---
# This is the SQL version of what the notebook does using .groupby()
# In SQL: GROUP BY channel_used, then AVG() the metrics
# This is IDENTICAL to: cleaned_df.groupby("channel_used").agg(...)

sql_channel_query = """
SELECT
    channel_used,
    COUNT(campaign_id)                          AS total_campaigns,
    ROUND(AVG(roi), 4)                          AS avg_roi,
    ROUND(AVG(conversion_rate) * 100, 2)        AS avg_conversion_rate_pct,
    ROUND(SUM(acquisition_cost), 2)             AS total_cost,
    ROUND(AVG(cost_per_conversion), 2)          AS avg_cost_per_conversion,
    ROUND(SUM(clicks) * 1.0 / SUM(impressions), 4) AS overall_ctr
FROM campaigns
GROUP BY channel_used
ORDER BY avg_roi DESC;
"""

channel_sql_result = pd.read_sql_query(sql_channel_query, conn)
print("SQL Result: Channel Performance (sorted by ROI):")
print(channel_sql_result.to_string(index=False))

# --- SQL Query 2: Customer Segment Profitability ---
# This query answers: "Which customer segment generates the most PROFIT?"
# We use SUM(estimated_profit) to rank segments by total business value

sql_segment_query = """
SELECT
    customer_segment,
    COUNT(campaign_id)                      AS total_campaigns,
    ROUND(AVG(roi), 4)                      AS avg_roi,
    ROUND(SUM(estimated_profit), 2)         AS total_estimated_profit,
    ROUND(AVG(cost_per_conversion), 2)      AS avg_cac
FROM campaigns
GROUP BY customer_segment
ORDER BY total_estimated_profit DESC;
"""

segment_sql_result = pd.read_sql_query(sql_segment_query, conn)
print("\nSQL Result: Customer Segment Profitability (sorted by Total Profit):")
print(segment_sql_result.to_string(index=False))

# --- SQL Query 3: Monthly ROI Trend ---
# In a real company, you'd be asked: "Is our marketing getting better over time?"
# This query extracts the year-month from the date and calculates monthly avg ROI

sql_monthly_query = """
SELECT
    SUBSTR(date, 1, 7)              AS year_month,
    COUNT(campaign_id)              AS campaigns,
    ROUND(AVG(roi), 4)              AS avg_roi,
    ROUND(AVG(conversion_rate), 4)  AS avg_conversion_rate
FROM campaigns
GROUP BY year_month
ORDER BY year_month ASC;
"""

monthly_sql_result = pd.read_sql_query(sql_monthly_query, conn)
print("\nSQL Result: Monthly ROI Trend:")
print(monthly_sql_result.to_string(index=False))

conn.close()
print("\n[OK] SQL Analysis complete. Database connection closed.")


# ===============================================================
# ADVANCEMENT 2: A/B Testing with Statistical Significance
# ===============================================================
# WHY WE ARE DOING THIS:
# The main notebook concludes: "Facebook has higher average ROI than Website."
# But here is the critical question: Is that difference REAL, or just luck?
#
# Imagine you flipped a coin 10 times and got 7 heads.
# Is the coin biased? Or did you just get lucky?
# You need a mathematical test to check. That test is called a T-Test.
#
# The T-Test compares two groups and gives you a "p-value."
# p-value < 0.05 means: "The difference is REAL, not luck."
# p-value > 0.05 means: "The difference might just be random."

print("\n" + "=" * 60)
print("ADVANCEMENT 2: A/B Testing - Statistical Significance")
print("=" * 60)

# --- A/B Test 1: Facebook vs Email ROI ---
# We want to test: Is Facebook ROI statistically higher than Email ROI?
# Group A = all Facebook campaign ROI values
# Group B = all Email campaign ROI values

facebook_roi = cleaned_df[cleaned_df["channel_used"] == "Facebook"]["roi"]
email_roi    = cleaned_df[cleaned_df["channel_used"] == "Email"]["roi"]
website_roi  = cleaned_df[cleaned_df["channel_used"] == "Website"]["roi"]

t_stat_fb_email, p_value_fb_email = stats.ttest_ind(facebook_roi, email_roi, equal_var=False)
t_stat_fb_web,   p_value_fb_web   = stats.ttest_ind(facebook_roi, website_roi, equal_var=False)

print(f"\nA/B Test: Facebook ROI vs Email ROI")
print(f"  Facebook avg ROI : {facebook_roi.mean():.4f}")
print(f"  Email avg ROI    : {email_roi.mean():.4f}")
print(f"  p-value          : {p_value_fb_email:.4f}")
print(f"  Conclusion       : {'[SIGNIFICANT] Real difference' if p_value_fb_email < 0.05 else '[NOT SIGNIFICANT] Could be luck'}")

print(f"\nA/B Test: Facebook ROI vs Website ROI")
print(f"  Facebook avg ROI : {facebook_roi.mean():.4f}")
print(f"  Website avg ROI  : {website_roi.mean():.4f}")
print(f"  p-value          : {p_value_fb_web:.4f}")
print(f"  Conclusion       : {'[SIGNIFICANT] Real difference' if p_value_fb_web < 0.05 else '[NOT SIGNIFICANT] Could be luck'}")

# --- A/B Test 2: Influencer vs Social Media (Campaign Type) ---
influencer_roi   = cleaned_df[cleaned_df["campaign_type"] == "Influencer"]["roi"]
social_media_roi = cleaned_df[cleaned_df["campaign_type"] == "Social Media"]["roi"]

t_stat_infl, p_value_infl = stats.ttest_ind(influencer_roi, social_media_roi, equal_var=False)

print(f"\nA/B Test: Influencer Campaigns vs Social Media Campaigns")
print(f"  Influencer avg ROI   : {influencer_roi.mean():.4f}")
print(f"  Social Media avg ROI : {social_media_roi.mean():.4f}")
print(f"  p-value              : {p_value_infl:.4f}")
print(f"  Conclusion           : {'[SIGNIFICANT] Real difference' if p_value_infl < 0.05 else '[NOT SIGNIFICANT] Could be luck'}")

print("\n[OK] A/B Testing complete.")


# ===============================================================
# ADVANCEMENT 3: LTV / CAC Ratio Analysis
# ===============================================================
# WHY WE ARE DOING THIS:
# CAC (Customer Acquisition Cost) = How much it costs to get ONE customer.
# LTV (Lifetime Value) = How much money that customer will bring over their lifetime.
#
# The most important ratio in any consumer business (Zomato, Swiggy, etc.) is:
#   LTV / CAC
#
# If LTV/CAC > 3, the business is HEALTHY. You spend $1 to make $3+.
# If LTV/CAC < 1, the business is DYING. You spend more to acquire than you earn.
#
# In our dataset, we don't have true "lifetime" data, so we ESTIMATE LTV
# by using the estimated_revenue as a proxy for what a customer brings in.
# This is standard practice in academic/portfolio projects.

print("\n" + "=" * 60)
print("ADVANCEMENT 3: LTV / CAC Ratio Analysis by Segment")
print("=" * 60)

# For each customer segment, calculate:
# CAC  = average acquisition cost (what we pay to get a customer)
# LTV  = average estimated revenue per campaign (proxy for lifetime value)
# Ratio = LTV / CAC (health indicator)

ltv_cac = (
    cleaned_df.groupby("customer_segment")
    .agg(
        avg_cac=("acquisition_cost",   "mean"),
        avg_ltv=("estimated_revenue",  "mean"),
        total_campaigns=("campaign_id","count")
    )
    .reset_index()
)

ltv_cac["ltv_cac_ratio"] = ltv_cac["avg_ltv"] / ltv_cac["avg_cac"]
ltv_cac["health"]        = ltv_cac["ltv_cac_ratio"].apply(
    lambda x: "[GREEN] Healthy (>3)" if x > 3 else ("[YELLOW] Borderline (1-3)" if x >= 1 else "[RED] Unsustainable (<1)")
)
ltv_cac_sorted = ltv_cac.sort_values("ltv_cac_ratio", ascending=False)

print("\nLTV / CAC Ratio by Customer Segment:")
print(ltv_cac_sorted[["customer_segment", "avg_cac", "avg_ltv", "ltv_cac_ratio", "health"]].to_string(index=False))

# Plot the LTV/CAC ratios
plt.figure()
colors = ["#2ecc71" if r > 3 else "#f39c12" if r >= 1 else "#e74c3c"
          for r in ltv_cac_sorted["ltv_cac_ratio"]]  # green=healthy, orange=borderline, red=bad
plt.bar(ltv_cac_sorted["customer_segment"], ltv_cac_sorted["ltv_cac_ratio"], color=colors)
plt.axhline(y=3, color="black", linestyle="--", linewidth=1, label="Healthy threshold (3x)")
plt.title("LTV / CAC Ratio by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("LTV / CAC Ratio")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("../visuals/ltv_cac_by_segment.png", dpi=150)
plt.show()

print("\n[OK] LTV/CAC analysis complete. Chart saved to visuals/ltv_cac_by_segment.png")
print("\n" + "=" * 60)
print("ALL ADVANCEMENTS COMPLETE.")
print("=" * 60)
