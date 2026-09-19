"""
Enterprise Data Quality Suite Execution Runner
Target: E-Commerce Transaction Ledger (541K+ Records)
Standard: ANSI SQL / SQLite In-Memory Execution
Output: Executive DQ Audit Summary Report (OSFI E-13 / BCBS 239)
"""

import os
import sqlite3
import pandas as pd


def resolve_data_path() -> str:
    """Find the raw dataset path across common directory layouts."""
    candidates = [
        os.path.join("data", "raw", "ecommerce_data.csv"),
        os.path.join("..", "data", "raw", "ecommerce_data.csv"),
        "ecommerce_data.csv",
        os.path.join("data", "ecommerce_data.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Raw data file not found. Checked locations: {candidates}"
    )


def resolve_sql_path() -> str:
    """Find the SQL validation suite path."""
    candidates = [
        os.path.join("sql", "data_quality_suite.sql"),
        "data_quality_suite.sql",
        os.path.join("..", "sql", "data_quality_suite.sql"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"SQL script not found. Checked locations: {candidates}"
    )


def run_dq_pipeline():
    # 1. Ingest Raw Dataset
    data_path = resolve_data_path()
    print(f"\n[1/4] Ingesting raw data from: {data_path} ...")
    df_raw = pd.read_csv(data_path, encoding="ISO-8859-1")
    print(f"      Total records loaded: {len(df_raw):,}")

    # 2. Schema Standardization (Mapping PascalCase to Enterprise snake_case)
    print("\n[2/4] Standardizing schema to Enterprise Physical Asset format...")
    df_normalized = df_raw.rename(
        columns={
            "InvoiceNo": "invoice_no",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "UnitPrice": "unit_price",
            "CustomerID": "customer_id",
            "Country": "country",
        }
    )

    # Calculate derived metric for financial consistency validation
    df_normalized["line_revenue"] = (
        df_normalized["quantity"] * df_normalized["unit_price"]
    ).round(2)

    # 3. Load into In-Memory Database
    print(
        "\n[3/4] Initializing In-Memory Relational Engine (tbl_transaction_ledger)..."
    )
    conn = sqlite3.connect(":memory:")
    df_normalized.to_sql("tbl_transaction_ledger", conn, index=False)

    # 4. Execute SQL Data Quality Suite
    sql_path = resolve_sql_path()
    print(f"\n[4/4] Executing SQL DQ Suite: {sql_path} ...\n")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    df_results = pd.read_sql_query(sql_script, conn)

    # 5. Display Executive Monitoring Output
    print("=" * 115)
    print(
        "ENTERPRISE DATA QUALITY AUDIT & MONITORING REPORT (OSFI E-13 / BCBS 239)"
    )
    print("=" * 115)
    print(
        df_results.to_string(
            index=False,
            col_space={
                "dq_dimension": 14,
                "rule_name": 35,
                "sla_threshold": 32,
                "total_records": 14,
                "failure_count": 14,
                "failure_rate_pct": 16,
                "sla_status": 10,
            },
        )
    )
    print("=" * 115)

    # Export Audit Summary Report
    report_dir = "governance"
    os.makedirs(report_dir, exist_ok=True)
    output_report_path = os.path.join(report_dir, "dq_audit_summary_report.csv")
    df_results.to_csv(output_report_path, index=False)
    print(f"\n>>> Audit summary report exported: {output_report_path}\n")


if __name__ == "__main__":
    run_dq_pipeline()