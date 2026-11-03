# SRT distribution and missing-open source audit — 3 November 2026

This audit cross-checks the nine ex-date records in `data/processed/srt_dividends_yahoo.csv` against issuer announcements or an issuer annual report hosted by SGX. Amounts are **gross SGD per SRT unit**. These records are evidence for a future distribution ledger; the current model label and saved feature snapshot remain the raw close-price series. No distribution-inclusive backtest has been implemented yet.

| Ex-date | Yahoo SGD | Issuer SGD | Payment date | Primary record | Status |
| --- | ---: | ---: | --- | --- | --- |
| 2022-07-19 | 0.026383 | 0.021062 + 0.005321 = 0.026383 | 2022-08-22 | [1H2022](https://links.sgx.com/FileOpen/Announcement%20Dividend%20Distribution%201H2022%2020220712.ashx?App=Announcement&FileID=723614), [2H2021](https://links.sgx.com/1.0.0/corporate-announcements/UUF19XSTD8B9NM33/723625_Announcement%20Dividend%20Distribution%202H2021%2020220713.pdf) | Two issuer distributions share one ex-date; Yahoo combines them |
| 2023-01-19 | 0.016500 | 0.016500 | 2023-02-23 | [CSOP 2022 annual report, subsequent events](https://links.sgx.com/1.0.0/corporate-announcements/Q4VI1BE3GXBCOOXG/751715_CSOP%20SG%20ETF%20Series%20I%20-%20Annual%20Report%202022.pdf) | Match |
| 2023-07-19 | 0.029200 | 0.029200 | 2023-08-23 | [2H2023 announcement](https://links.sgx.com/FileOpen/Announcement%20Dividend%20Distribution%202H2023%20230710.ashx?App=Announcement&FileID=764954) | Match |
| 2024-01-19 | 0.025000 | 0.025000 | 2024-02-22 | [1H2024 announcement](https://links.sgx.com/FileOpen/Announcement%20Dividend%20Distribution%201H2024_240112_2.ashx?App=Announcement&FileID=782825) | Match |
| 2024-07-19 | 0.022000 | 0.022000 | 2024-08-22 | [2H2024 announcement](https://links.sgx.com/1.0.0/corporate-announcements/NOYZQ20FW51IVXSN/809400_Announcement%20Dividend%20Distribution%202H2024%20240711.pdf) | Match |
| 2025-01-20 | 0.022000 | 0.022000 | 2025-02-24 | [1H2025 announcement](https://links.sgx.com/1.0.0/corporate-announcements/R6TFLC0RN6VTENW7/830182_Announcement%20Dividend%20Distribution%201H2025%20250110.pdf) | Match |
| 2025-07-21 | 0.022000 | 0.022000 | 2025-08-21 | [2H2025 announcement](https://links.sgx.com/1.0.0/corporate-announcements/NSYA6FGDMPV1FXCJ/851495_Announcement%20Dividend%20Distribution%202H2025%20250710.pdf) | Match |
| 2026-01-19 | 0.022000 | 0.022000 | 2026-02-23 | [1H2026 announcement](https://links.sgx.com/FileOpen/Announcement%20Dividend%20Distribution%201H2026%20260109.ashx?App=Announcement&FileID=871675) | Match |
| 2026-07-20 | 0.021000 | 0.021000 | 2026-08-21 | [2H2026 announcement](https://links.sgx.com/FileOpen/Announcement%20Dividend%20Distribution%202H2026%20260710.ashx?App=Announcement&FileID=896004) | Match |

**2025-10-24 open remains unresolved.** The saved snapshot has a repaired close of SGD 0.801 and no open. A targeted search did not establish an independently verifiable SRT open for that date. Do not infer it from close, high or low. The causal no-trade-and-carry rule in protocol v1.2 applies unless a dated source supplies a traceable opening print; any repair requires a new source hash, data audit and protocol version before final wealth reporting.

For a distribution-inclusive ledger, book entitlement from units held immediately before each ex-date, a receivable on ex-date and cash only on payment date. Declare gross-individual-investor and price-only assumptions separately; actual after-tax receipts vary by investor category. The 2022-07-19 row requires one combined ex-date entry with two underlying issuer announcements, not two independently traded ex-date events.
