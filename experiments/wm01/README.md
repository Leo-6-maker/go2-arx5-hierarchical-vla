# WM01 · Task-space world-model-assisted policy learning

**状态：设计已发布；训练／原生适配器尚未实现。** 目标不是重训WBC，也不是先替换GR00T，而是用受控实验判断小WM是否帮助学出任务策略。

先读[详细协议](../../docs/protocols/WM01_TASKSPACE_WORLD_MODEL_POLICY_V1.md)，再读[Codex交接](../../docs/handover/2026-11-03/WM01_CODEX_EXECUTE.md)。`plan.json`中的数值是首版设计；未绑定环境、任务scorer、数据/资产和实际站点，因此不能直接作为训练命令。

本次真正可执行、无模型／物理副作用的命令：

```bash
python experiments/wm01/check_plan.py
python -m unittest discover -s experiments/wm01/tests -p 'test_*.py' -v
```

这两个命令不调度作业、不加载Torch/MuJoCo、不写数据，也不证明模型／任务成功。后续训练命令必须由Codex实现CLI和原生验收后记录，禁止把计划中的文件名当成已有程序。

原始数据、权重、私有资产与主机路径留在现有授权存储。本目录只保留设计、配置模板和轻量检查；结果写`reports/WM01/<run_id>/`的脱敏摘要与授权artifact引用。
