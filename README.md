# event-delta

目标：验证低频 RGB 语义锚点与连续 Event 状态增量能否以较低成本改善流式 VLM 在任意查询时刻的状态与回答；当前阶段为 R1 SEVD 数据审计与因果时间轴确认；运行方式：`python3 tools/audit_sevd.py --root <SEVD_ROOT> --recording fixed-3 --recording fixed-4 --output artifacts/r1/audit.json`。
