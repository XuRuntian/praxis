# Praxis

基于人工范例研究物理切分方法，并以冻结规则批量执行。第一版仅支持当前 LeRobot Parquet 数据布局；不提供通用数据转换、描述生成或 Agent 调度框架。

Agent 只需交付 `rule.py`，必要时附参数和模板。接口见 [docs/RULE_SPEC.md](docs/RULE_SPEC.md)。执行器按任务目录加载方法，不需修改核心代码；不固定范例数量，批量执行不要求目标有人工标注。

## 安装

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
source .venv/bin/activate
```

Python >=3.11。首次复现可使用 `requirements-replay.txt` 的实验依赖版本。包安装名为 praxis-segmentation，命令名为 praxis。

## 批量切分

```bash
praxis run --data-root /path/to/training_set \
  --tasks examples/split_20_tasks.json --rules-root task_rules/r2 \
  --output outputs/r2_replay
```

源数据不复制、不转换。任务目录中需要 meta/info.json 和 data 下的 Parquet。读取器只选择规则声明的物理列和帧/时间/episode 索引，不读取任务标注。数据不可用时保存 error，不丢弃 episode。

每条输出包含 dataset、episode_id、frame_count、cuts、has_prediction、status、review_reasons、error。`cuts=[30,70]` 表示新阶段从原始第 30、70 帧（零基）开始；隐含端点为 0 和 frame_count。空 cuts 且 has_prediction=true 表示整条一个片段；has_prediction=false 表示没有预测，两者不能混淆。兼容当前评分的半开区间由 boundaries.py 集中转换。

输出目录必须不存在。freeze.json 保存代码和规则哈希、依赖版本；manifest.json 保存预测及物理文件哈希。有效输出可能为 review，不等于标注正确。

## 独立评分与复现核验

仅预测完成后执行评分。当前评分只测切分，失败按空预测计入分母。IoU 最大基数一对一匹配与原基线一致。

```bash
praxis score --run outputs/r2_replay --data-root /path/to/training_set \
  --output outputs/r2_score.json
praxis compare --run outputs/r2_replay \
  --reference-root /path/to/physical-rule-benchmark \
  --reference-manifest /path/to/physical-rule-benchmark/predictions/task_research_r2_20261106/manifest.json \
  --output outputs/r2_comparison.json
```

compare 不读取真值，核对每条切分点、状态、帧数、原因、错误和物理文件哈希。只在这个迁移核验步骤需要历史实验产物；正常 run 不依赖旧仓库。

## 结构与范围

- `src/praxis/data.py`：当前数据路径解析和输入适配。
- `src/praxis/rules.py`：最小规则接口与 R2 适配器。
- `src/praxis/methods/r2/`：保留数值行为的第二轮兼容实现。
- `task_rules/r2/`：20 个任务参数、两范例模板及可用的研究证据。
- `src/praxis/runner.py`：批量执行与版本记录。
- `src/praxis/evaluation.py`：评分和迁移核验。

R2 模块保留了原实验的任务条件分支和输入检查，这不是要求新方法服从统一算法。后续按实际复用需要抽取组件。模板来自两条人工范例，含旧语义标签；当前接口不输出描述。

本次迁移不是新实验，也不修复 106 条已知输入失败。原实验已暴露，不能宣称盲测。来源和逐文件原始哈希见 MIGRATION.json；评分匹配实现来源为原基线的 iou/match_segments。数据、完整预测、图表留在仓库外或被 Git 忽略的 outputs 下。

## 新任务接入

将方法放在 `task_rules/<task_id>/rule.py`，实现 `segment(episode)` 返回切分点或 None。若有模板，提供 `load(directory)` 返回实现同一方法的对象。运行清单只需 task_id、dataset、target_ids；exemplar_ids 可选。`--tasks` 是清单参数，旧 `--split` 保持兼容。

内置 20 个 R2 任务各有薄入口，数值实现不变。执行器不再根据任务名或算法类型分支。新方法目录的全部资源会被记录哈希，运行期间改变或增删资源将导致完整性检查失败。模板初始化每任务一次。
