# Cleanup Manifest

本清单记录标准目录产物二次整理到 handoff 目录后的映射结果。

- ref_min files: 31
- archive_not_for_handoff files: 114
- delete_now files: 531

## Missing ref_min Files

- none

## Notes

- 整理逻辑为复制，不直接把实验脚本的输出目录改写成 handoff 目录。
- `archive_not_for_handoff` 收纳 `plan/reports/results/statistic_results` 中除 ref_min 与 delete_now 以外的其余文件。
- `delete_now` 收纳 `logs/`、`__pycache__/`、`_smoke*` 和 `temp_weather_011.sh`。
