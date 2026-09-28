# U21–U80 training-row distribution: seed 137

Descriptive only; row shares do not identify a cause of the test gap. R3c test evidence is exploratory, not pre-registered.

| Dimension | Category | R3c share | R4 fixed share |
|---|---|---:|---:|
| fault_share | constraint_violation | 0.3458333333333333 | 0.375 |
| fault_share | missing_dependency | 0.3375 | 0.325 |
| fault_share | stale_version | 0.31666666666666665 | 0.3 |
| family_share | auth_service | 0.15416666666666667 | 0.15416666666666667 |
| family_share | cache_service | 0.175 | 0.175 |
| family_share | logging_service | 0.15833333333333333 | 0.15833333333333333 |
| family_share | network_service | 0.14166666666666666 | 0.14583333333333334 |
| family_share | queue_service | 0.21666666666666667 | 0.2125 |
| family_share | storage_service | 0.15416666666666667 | 0.15416666666666667 |

Source: `row_distribution.json`, including R3c dump-observed group order and R4 selection order; original R3c within-batch dataloader order is not independently established. Each run has 240 unique instances; overlap is 233. R4 reuses 7 warm-up instances and R3c reuses none. R4 stratifies each stage by fault type, orders unseen instances before seen instances inside each cell, then permutes the stage rows. Seed noise cannot be ruled out offline.
