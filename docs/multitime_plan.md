# Multi-Time Comparison Plan

Multi-time comparison sets are planned for p-refinement-style analysis across several physical times. Each set must be aligned by the physical time stored in the Nek5000 field file header, not by assuming identical file indices represent identical times across cases.

The planned target times are approximately `5.0`, `10.0`, `15.0`, and `19.5`. `N11` remains the reference case. Future comparisons are expected to cover concentration, velocity magnitude, and pressure fluctuation.

This step only adds comparison-set configuration and documentation. It does not change comparison, interpolation, metrics, plotting, or extraction logic, and it does not add mean error metrics.

## Configured Comparison Sets

The file indices below were selected as the closest available stored physical times found in the raw Nek5000 file headers.

| comparison_set | target_time | N5 index | N7 index | N9 index | N11 index |
| --- | ---: | ---: | ---: | ---: | ---: |
| t05 | 5.0 | 21 | 21 | 21 | 11 |
| t10 | 10.0 | 41 | 41 | 41 | 21 |
| t15 | 15.0 | 61 | 61 | 61 | 31 |
| t19p5 | 19.5 | 79 | 79 | 79 | 40 |

## Header Times Used

| comparison_set | case | index | stored_time |
| --- | --- | ---: | ---: |
| t05 | N5 | 21 | 5.00236067446 |
| t05 | N7 | 21 | 5.00129086714 |
| t05 | N9 | 21 | 5.00108809531 |
| t05 | N11 | 11 | 5.00073324795 |
| t10 | N5 | 41 | 10.0004069593 |
| t10 | N7 | 41 | 10.0019487417 |
| t10 | N9 | 41 | 10.0013061779 |
| t10 | N11 | 21 | 10.0003840477 |
| t15 | N5 | 61 | 15.0011826847 |
| t15 | N7 | 61 | 15.000902342 |
| t15 | N9 | 61 | 15.0008597114 |
| t15 | N11 | 31 | 15.0016310673 |
| t19p5 | N5 | 79 | 19.5026151658 |
| t19p5 | N7 | 79 | 19.5012282796 |
| t19p5 | N9 | 79 | 19.5000965834 |
| t19p5 | N11 | 40 | 19.50099856145 |
