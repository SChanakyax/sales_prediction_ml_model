#!/usr/bin/env python3
# name=sales_analysis_fixed.py
"""
Robust sales analysis pipeline.
- Auto-detects common column names and renames them to expected names.
- Parses date-like columns into 'transaction_date'.
- Skips analyses that require missing columns and returns clear messages.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from mlxtend.frequent_patterns import apriori, association_rules

sns.set(style="whitegrid")

# candidate names for expected columns (lowercased, normalized)
COLUMN_CANDIDATES = {
    "transaction_id": ["transaction_id","invoice_no","invoiceno","invoice","order_id","orderid","order_no","order"],
    "customer_id": ["customer_id","customerid","customer","cust_id","custid","buyer_id","client_id","CustomerID"],
    "product_id": ["product_id","productid","stockcode","stock_code","product_code","sku","StockCode"],
    "product_category": ["product_category","category","category_name","categoryname","product_category_name","department","segment","group","ProductCategory","Description"],
    "price": ["price","unitprice","unit_price","unit_price_usd","UnitPrice","price_usd"],
    "quantity": ["quantity","qty","Quantity","units","UnitQty"],
    "customer_age": ["customer_age","age","customerage","CustomerAge"],
    "customer_gender": ["customer_gender","gender","sex","CustomerGender"],
    "transaction_date": ["transaction_date","date","order_date","invoice_date","InvoiceDate","timestamp","time","sale_date"]
}

def _normalize(colname):
    return "".join(ch for ch in colname.lower() if ch.isalnum())

def _find_column(df_cols, candidates):
    norm_cols = { _normalize(c): c for c in df_cols }
    for cand in candidates:
        nc = _normalize(cand)
        if nc in norm_cols:
            return norm_cols[nc]
    # try partial contains match (less strict)
    for col in df_cols:
        low = col.lower()
        for cand in candidates:
            if cand.lower() in low:
                return col
    return None

def _auto_map_columns(df):
    mapping = {}
    cols = list(df.columns)
    for expected, cand_list in COLUMN_CANDIDATES.items():
        found = _find_column(cols, cand_list)
        if found:
            mapping[found] = expected
    return mapping

def _detect_and_parse_date(df):
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        return df
    # attempt to find date-like columns using candidates mapping
    mapping = _auto_map_columns(df)
    for orig, mapped in mapping.items():
        if mapped == 'transaction_date':
            df['transaction_date'] = pd.to_datetime(df[orig], errors='coerce')
            return df
    # fallback: find any column with 'date' or 'time' in name
    for c in df.columns:
        if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower():
            df['transaction_date'] = pd.to_datetime(df[c], errors='coerce')
            return df
    # nothing found
    return df

def load_and_prepare(path_or_df):
    # Accept path or dataframe
    if isinstance(path_or_df, str):
        if not os.path.exists(path_or_df):
            raise FileNotFoundError(f"File not found: {path_or_df}")
        df = pd.read_csv(path_or_df)
    else:
        df = path_or_df.copy()

    # auto-rename columns where possible
    mapping = _auto_map_columns(df)
    if mapping:
        df = df.rename(columns=mapping)

    # parse/normalize dates
    df = _detect_and_parse_date(df)
    if 'transaction_date' not in df.columns:
        # couldn't detect a date column
        raise ValueError("No date-like column found. Add 'transaction_date' or a column named like 'date', 'invoice_date', etc.")

    # ensure at least one date can be parsed
    if df['transaction_date'].isna().all():
        raise ValueError("Could not parse any values in the detected date column into datetimes. Check formats.")

    # compute total_amount if possible
    if 'total_amount' not in df.columns:
        if 'price' in df.columns and 'quantity' in df.columns:
            df['total_amount'] = df['price'] * df['quantity']
        elif 'price' in df.columns:
            df['total_amount'] = df['price']
        elif 'quantity' in df.columns:
            df['total_amount'] = df['quantity']  # fallback but not ideal
        else:
            df['total_amount'] = np.nan

    # validate numeric columns where expected
    for col in ('price','quantity'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # demographic cleaning (if present)
    if 'customer_age' in df.columns:
        df['customer_age'] = pd.to_numeric(df['customer_age'], errors='coerce')
        # keep plausible ages only
        df = df[df['customer_age'].between(10,100, inclusive="both")] if 'customer_age' in df.columns else df
    if 'customer_gender' in df.columns:
        df['customer_gender'] = df['customer_gender'].fillna('Unknown').astype(str)

    # sort by date
    df = df.sort_values('transaction_date').reset_index(drop=True)
    return df

# Helper to check availability and return message if missing
def _require(df, cols, name):
    missing = [c for c in cols if c not in df.columns or df[c].isna().all()]
    if missing:
        return False, f"SKIPPED {name}: missing columns or all-NaN columns: {missing}"
    return True, None

# --- analysis functions now check prerequisites and return skip messages when needed ---
def age_gender_influence(df, output_prefix="out"):
    ok, msg = _require(df, ['customer_id','customer_age','customer_gender','total_amount'], "age_gender_influence")
    if not ok:
        return {"skipped": msg}
    # rest same as before (aggregate and plot)
    cust = df.groupby('customer_id').agg(
        total_spend=('total_amount','sum'),
        transactions=('transaction_id','nunique') if 'transaction_id' in df.columns else ('total_amount','count'),
        avg_items_per_tx=('quantity','mean') if 'quantity' in df.columns else ('total_spend','mean'),
        avg_price=('price','mean') if 'price' in df.columns else ('total_spend','mean'),
        age=('customer_age','first'),
        gender=('customer_gender','first')
    ).reset_index()
    plt.figure(figsize=(10,5))
    sns.boxplot(x='gender', y='total_spend', data=cust)
    plt.yscale('log')
    plt.title('Customer total spend by gender (log scale)')
    plt.savefig(f"{output_prefix}_spend_by_gender.png", bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='age', y='total_spend', hue='gender', data=cust, alpha=0.6)
    plt.yscale('log')
    plt.title('Total spend vs age by gender')
    plt.savefig(f"{output_prefix}_spend_vs_age.png", bbox_inches='tight')
    plt.close()
    cust['log_spend'] = np.log1p(cust['total_spend'])
    X = pd.get_dummies(cust[['age','gender']], drop_first=True)
    y = cust['log_spend']
    model = LinearRegression().fit(X.fillna(0), y.fillna(0))
    preds = model.predict(X.fillna(0))
    r2 = r2_score(y.fillna(0), preds)
    return {"cust_summary": cust, "r2": float(r2)}

def time_series_analysis(df, output_prefix="out"):
    ok, msg = _require(df, ['transaction_date','total_amount'], "time_series_analysis")
    if not ok:
        return {"skipped": msg}
    ts = df.set_index('transaction_date').resample('D').agg({'total_amount':'sum'})['total_amount'].fillna(0)
    period = 7 if len(ts) >= 14 else None
    res = seasonal_decompose(ts, model='additive', period=period) if period else None
    if res is not None:
        res.trend.plot(title='Trend'); plt.savefig(f"{output_prefix}_trend.png"); plt.close()
        res.seasonal.plot(title='Seasonal'); plt.savefig(f"{output_prefix}_seasonal.png"); plt.close()
        res.resid.plot(title='Residual'); plt.savefig(f"{output_prefix}_resid.png"); plt.close()
    if 'transaction_date' in df.columns:
        df['dow'] = df['transaction_date'].dt.day_name()
        df['hour'] = df['transaction_date'].dt.hour
        if 'product_category' in df.columns:
            pivot = df.pivot_table(index='dow', columns='hour', values='total_amount', aggfunc='sum').fillna(0)
            days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            pivot = pivot.reindex([d for d in days if d in pivot.index])
            plt.figure(figsize=(12,5))
            sns.heatmap(np.log1p(pivot), cmap='viridis')
            plt.title('Log total_amount heatmap (day-of-week x hour)')
            plt.savefig(f"{output_prefix}_dow_hour_heatmap.png")
            plt.close()
    return {"time_series": ts, "decompose": res}

def category_popularity(df, top_n=20):
    ok, msg = _require(df, ['product_category','total_amount'], "category_popularity")
    if not ok:
        return {"skipped": msg}
    cat = df.groupby('product_category').agg(
        total_revenue=('total_amount','sum'),
        total_units=('quantity','sum') if 'quantity' in df.columns else ('total_amount','count'),
        transactions=('transaction_id','nunique') if 'transaction_id' in df.columns else ('total_amount','count'),
        unique_customers=('customer_id','nunique') if 'customer_id' in df.columns else ('total_revenue','count')
    ).sort_values('total_revenue', ascending=False)
    return cat.head(top_n)

def age_spending_preferences(df, output_prefix="out"):
    ok, msg = _require(df, ['customer_age','product_category','total_amount'], "age_spending_preferences")
    if not ok:
        return {"skipped": msg}
    df['age_group'] = pd.cut(df['customer_age'].fillna(-1), bins=[0,24,34,44,54,64,100], labels=['<25','25-34','35-44','45-54','55-64','65+'])
    pref = df.groupby(['age_group','product_category']).agg(total_revenue=('total_amount','sum')).reset_index()
    pivot = pref.pivot(index='product_category', columns='age_group', values='total_revenue').fillna(0)
    plt.figure(figsize=(10,12))
    sns.heatmap(np.log1p(pivot), cmap='magma')
    plt.title('Log revenue by product_category vs age_group')
    plt.savefig(f"{output_prefix}_category_age_heatmap.png")
    plt.close()
    per_customer = df.groupby('customer_id').agg(age=('customer_age','first'), total_spend=('total_amount','sum')) if 'customer_id' in df.columns else None
    corr = per_customer[['age','total_spend']].corr().iloc[0,1] if per_customer is not None else None
    return {"pivot": pivot, "age_spend_correlation": corr}

def seasonal_behavior(df, output_prefix="out"):
    ok, msg = _require(df, ['transaction_date','total_amount'], "seasonal_behavior")
    if not ok:
        return {"skipped": msg}
    df['month'] = df['transaction_date'].dt.month
    month_summary = df.groupby('month').agg(total_revenue=('total_amount','sum'), avg_basket=('total_amount','mean'), avg_items=('quantity','mean') if 'quantity' in df.columns else ('total_amount','mean'))
    plt.figure(figsize=(8,4))
    month_summary['total_revenue'].plot(kind='bar')
    plt.title('Revenue by month')
    plt.savefig(f"{output_prefix}_revenue_by_month.png")
    plt.close()
    cust = df.groupby('customer_id').agg(total_spend=('total_amount','sum'), tx_count=('transaction_id','nunique')) if 'customer_id' in df.columns else None
    if cust is not None and len(cust) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=0).fit(cust.fillna(0))
        cust['segment'] = kmeans.labels_
        df = df.merge(cust['segment'], left_on='customer_id', right_index=True)
        seg_month = df.groupby(['segment','month']).agg(total_revenue=('total_amount','sum')).reset_index()
    else:
        seg_month = pd.DataFrame()
    return {"month_summary": month_summary, "segment_month": seg_month}

def basket_size_analysis(df, output_prefix="out"):
    ok, msg = _require(df, ['transaction_id','total_amount','quantity'], "basket_size_analysis")
    if not ok:
        return {"skipped": msg}
    tx = df.groupby('transaction_id').agg(total_items=('quantity','sum'), total_amount=('total_amount','sum'))
    tx['size_bin'] = pd.cut(tx['total_items'], bins=[0,1,2,4,7,100], labels=['1','2','3-4','5-7','8+'])
    summary = tx.groupby('size_bin').agg(transactions=('total_amount','count'), avg_value=('total_amount','mean'))
    plt.figure(figsize=(6,4))
    summary['avg_value'].plot(kind='bar')
    plt.title('Avg transaction value by basket size')
    plt.savefig(f"{output_prefix}_avg_value_basket_size.png")
    plt.close()
    return {"tx_summary": summary, "tx_data": tx}

def price_distribution_per_category(df, output_prefix="out"):
    ok, msg = _require(df, ['product_category','price'], "price_distribution_per_category")
    if not ok:
        return {"skipped": msg}
    plt.figure(figsize=(10,8))
    top_cats = df['product_category'].value_counts().nlargest(12).index
    sns.boxplot(x='product_category', y='price', data=df[df['product_category'].isin(top_cats)])
    plt.xticks(rotation=45)
    plt.title('Price distribution (boxplot) for top categories')
    plt.savefig(f"{output_prefix}_price_dist_boxplot.png")
    plt.close()
    price_summary = df.groupby('product_category')['price'].describe()
    return price_summary

def association_rules_analysis(df, min_support=0.01, min_confidence=0.3):
    ok, msg = _require(df, ['transaction_id','product_category'], "association_rules_analysis")
    if not ok:
        return {"skipped": msg}
    basket = df.groupby(['transaction_id','product_category']).size().unstack(fill_value=0)
    basket = basket.applymap(lambda x: 1 if x>0 else 0)
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(['lift','confidence'], ascending=False)
    return {"frequent": frequent, "rules": rules}

def run_all(path_or_df, output_prefix="out"):
    df = load_and_prepare(path_or_df)
    results = {}
    results['age_gender'] = age_gender_influence(df, output_prefix)
    results['time_series'] = time_series_analysis(df, output_prefix)
    results['cat_pop'] = category_popularity(df)
    results['age_pref'] = age_spending_preferences(df, output_prefix)
    results['seasonal'] = seasonal_behavior(df, output_prefix)
    results['basket'] = basket_size_analysis(df, output_prefix)
    results['price_summary'] = price_distribution_per_category(df, output_prefix)
    try:
        results['assoc'] = association_rules_analysis(df)
    except Exception as e:
        results['assoc'] = {"error": str(e)}
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sales analyses.")
    parser.add_argument("path", nargs="?", default="dataset/retail_sales_dataset.csv", help="Path to transactions CSV file")
    parser.add_argument("-o","--output", default="analysis", help="Output prefix")
    parser.add_argument("--demo", action="store_true", help="Run demo data instead of file")
    args = parser.parse_args()

    if args.demo:
        rng = pd.date_range("2025-01-01", periods=200, freq="H")
        demo_df = pd.DataFrame({
            "transaction_id": [f"t{i}" for i in range(len(rng))],
            "customer_id": np.random.choice([f"c{i}" for i in range(50)], size=len(rng)),
            "transaction_date": rng,
            "product_id": np.random.choice([f"p{i}" for i in range(30)], size=len(rng)),
            "product_category": np.random.choice(["A","B","C","D"], size=len(rng)),
            "price": np.round(np.random.uniform(5,200,size=len(rng)),2),
            "quantity": np.random.randint(1,5,size=len(rng)),
            "customer_age": np.random.randint(18,80,size=len(rng)),
            "customer_gender": np.random.choice(["M","F","Other"], size=len(rng))
        })
        input_data = demo_df
    else:
        input_data = args.path

    try:
        results = run_all(input_data, output_prefix=args.output)
        print("Analyses complete. Results keys:", list(results.keys()))
        for k,v in results.items():
            if isinstance(v, dict) and v.get("skipped"):
                print(f"{k}: {v['skipped']}")
    except Exception as e:
        print("Error running analyses:", e)
        raise