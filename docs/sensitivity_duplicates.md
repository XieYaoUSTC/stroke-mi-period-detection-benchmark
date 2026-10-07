# 重复记录处理的敏感性(code/sensitivity_keep_one.py;2026-10-07)

口径:完整版(不剔除)/ 主口径(每对两份都剔 + sub-02 随访剔)/ 保留其一(剔编号较大者 + sub-02 随访剔)。均为患者均值。


## ll_raw.json

| 指标 | 完整版 | 主口径(两份都剔) | 保留其一 |
|---|---|---|---|
| csp_all | 0.744 | 0.7432 | 0.7436 |
| fixed_follow | 0.5917 | 0.6009 | 0.5961 |
| fixed_post | 0.5867 | 0.595 | 0.5907 |
| gain10_follow | 0.034 | 0.0275 | 0.033 |
| gain10_post | 0.0265 | 0.0265 | 0.03 |
| lda_c34 | 0.6259 | 0.6411 | 0.6332 |
| lda_far | 0.674 | 0.6845 | 0.6791 |
| lda_far_c34 | 0.7048 | 0.7203 | 0.7123 |
| n_dep_follow | 22 | 20 | 21 |
| n_dep_post | 23 | 21 | 22 |
| n_pre | 27 | 25 | 26 |
| riem_ts_lr | 0.8523 | 0.8568 | 0.8545 |
| riem_within_minus_fixed_follow | 0.0522 | 0.0234 | 0.0497 |
| riem_within_minus_fixed_post | 0.0701 | 0.0557 | 0.0731 |
| rule_c34 | 0.6677 | 0.6766 | 0.672 |
| rule_far | 0.71 | 0.7217 | 0.7156 |
| rule_within_minus_fixed_follow | -0.0059 | -0.0067 | -0.0059 |
| rule_within_minus_fixed_post | 0.0021 | 0.0027 | 0.0022 |

## sf_restlate.json

| 指标 | 完整版 | 主口径(两份都剔) | 保留其一 |
|---|---|---|---|
| csp_all | 0.8183 | 0.8314 | 0.8245 |
| fixed_follow | 0.6348 | 0.6294 | 0.6294 |
| fixed_post | 0.6478 | 0.6527 | 0.6518 |
| gain10_follow | 0.0184 | 0.0211 | 0.0211 |
| gain10_post | 0.0138 | 0.0148 | 0.0137 |
| lda_c34 | 0.7708 | 0.7738 | 0.7722 |
| lda_far | 0.7461 | 0.746 | 0.746 |
| lda_far_c34 | 0.7904 | 0.7932 | 0.7917 |
| n_dep_follow | 19 | 17 | 17 |
| n_dep_post | 24 | 22 | 23 |
| n_pre | 24 | 22 | 23 |
| riem_ts_lr | 0.8956 | 0.9017 | 0.8985 |
| riem_within_minus_fixed_follow | 0.0566 | 0.0481 | 0.0481 |
| riem_within_minus_fixed_post | 0.0413 | 0.0461 | 0.0387 |
| rule_c34 | 0.7714 | 0.7733 | 0.7723 |
| rule_far | 0.7579 | 0.7583 | 0.7581 |
| rule_within_minus_fixed_follow | 0.004 | 0.0003 | 0.0003 |
| rule_within_minus_fixed_post | 0.0069 | 0.0066 | 0.006 |
