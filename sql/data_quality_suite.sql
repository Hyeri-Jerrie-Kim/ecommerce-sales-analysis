-- =============================================================================
-- Enterprise Data Quality Validation Suite: Core Transaction Ledger
-- Target Schema: tbl_transaction_ledger
-- Regulatory Scope: OSFI E-13, BCBS 239 Data Integrity Baseline
-- DQ Dimensions: Completeness, Uniqueness, Domain Validity, Financial Consistency
-- =============================================================================

WITH dq_metrics_summary AS (
    SELECT
        COUNT(*) AS total_records,

        -- 1. Completeness Dimension (Customer ID 100% SLA)
        COUNT(CASE WHEN customer_id IS NULL OR TRIM(CAST(customer_id AS TEXT)) = '' THEN 1 END) AS null_customer_id_count,

        -- 2. Validity Dimension (Unit Price > 0.00)
        COUNT(CASE WHEN unit_price <= 0.00 OR unit_price IS NULL THEN 1 END) AS invalid_price_count,

        -- 3. Validity Dimension (Cancellation alignment: Negative quantity strictly with 'C' invoice)
        COUNT(CASE 
            WHEN quantity < 0 AND invoice_no NOT LIKE 'C%' THEN 1
            WHEN quantity > 0 AND invoice_no LIKE 'C%' THEN 1
            ELSE NULL 
        END) AS unreconciled_cancellation_count,

        -- 4. Validity Dimension (Quantity non-zero)
        COUNT(CASE WHEN quantity = 0 THEN 1 END) AS zero_quantity_count,

        -- 5. Consistency Dimension (line_revenue = quantity * unit_price)
        COUNT(CASE 
            WHEN ROUND(line_revenue, 2) != ROUND(quantity * unit_price, 2) THEN 1 
            ELSE NULL 
        END) AS revenue_discrepancy_count

    FROM tbl_transaction_ledger
),

duplicate_key_summary AS (
    -- 6. Uniqueness Dimension (Composite PK: invoice_no + stock_code)
    SELECT COUNT(*) AS duplicate_composite_key_count
    FROM (
        SELECT invoice_no, stock_code
        FROM tbl_transaction_ledger
        GROUP BY invoice_no, stock_code
        HAVING COUNT(*) > 1
    )
)

-- =============================================================================
-- Final Unified Executive Monitoring View (SLA Compliance Metrics)
-- =============================================================================
SELECT
    'Completeness' AS dq_dimension,
    'Customer ID Non-Null' AS rule_name,
    '100% Populated (0% Null SLA)' AS sla_threshold,
    s.total_records,
    s.null_customer_id_count AS failure_count,
    ROUND((s.null_customer_id_count * 100.0 / s.total_records), 2) AS failure_rate_pct,
    CASE WHEN s.null_customer_id_count = 0 THEN 'PASS' ELSE 'BREACH' END AS sla_status
FROM dq_metrics_summary s

UNION ALL

SELECT
    'Uniqueness',
    'Composite PK (InvoiceNo + StockCode)',
    '100% Distinct Records',
    s.total_records,
    d.duplicate_composite_key_count,
    ROUND((d.duplicate_composite_key_count * 100.0 / s.total_records), 2),
    CASE WHEN d.duplicate_composite_key_count = 0 THEN 'PASS' ELSE 'BREACH' END
FROM dq_metrics_summary s, duplicate_key_summary d

UNION ALL

SELECT
    'Validity',
    'Unit Price Positive Float',
    'Unit Price > 0.00',
    s.total_records,
    s.invalid_price_count,
    ROUND((s.invalid_price_count * 100.0 / s.total_records), 2),
    CASE WHEN s.invalid_price_count = 0 THEN 'PASS' ELSE 'BREACH' END
FROM dq_metrics_summary s

UNION ALL

SELECT
    'Validity',
    'Cancellation Alignment',
    'Negative Qty strictly with "C" Prefix',
    s.total_records,
    s.unreconciled_cancellation_count,
    ROUND((s.unreconciled_cancellation_count * 100.0 / s.total_records), 2),
    CASE WHEN s.unreconciled_cancellation_count = 0 THEN 'PASS' ELSE 'BREACH' END
FROM dq_metrics_summary s

UNION ALL

SELECT
    'Validity',
    'Quantity Non-Zero',
    'Quantity != 0',
    s.total_records,
    s.zero_quantity_count,
    ROUND((s.zero_quantity_count * 100.0 / s.total_records), 2),
    CASE WHEN s.zero_quantity_count = 0 THEN 'PASS' ELSE 'BREACH' END
FROM dq_metrics_summary s

UNION ALL

SELECT
    'Consistency',
    'Line Revenue Ledger Match',
    'line_revenue = quantity * unit_price',
    s.total_records,
    s.revenue_discrepancy_count,
    ROUND((s.revenue_discrepancy_count * 100.0 / s.total_records), 2),
    CASE WHEN s.revenue_discrepancy_count = 0 THEN 'PASS' ELSE 'BREACH' END
FROM dq_metrics_summary s;