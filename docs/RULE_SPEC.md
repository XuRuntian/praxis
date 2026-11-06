# Agent 方法交付：最小约定

当前只做切分。Agent 根据已有人工标注探索方法，交付一个任务目录；Praxis 加载它并批量执行。范例数量不固定，后续数据不需要人工 GT，评分是独立可选步骤。

## 必需：rule.py

```python
required_columns = ["observation.state"]  # 可省略，默认即此列

def segment(episode):
    signals = episode["arrays"]
    # 使用物理信号计算切分点，此处仅展示接口。
    return []
```

- `episode` 由当前数据读取器提供：arrays（列名到数值数组）、channels（通道定义和 fps）、eid、n（帧数）、timestamps、input_notes。
- 返回严格递增的 Python 整数列表，数字表示新阶段开始的原始零基帧位置，不含 0 和 n。
- `[]` 表示整条一个阶段；`None` 表示无法切分。异常由执行器记录。
- 方法内部可以归一化、平滑、对齐，但结果必须映射回原始帧。半开区间转换由 Praxis 集中处理。
- 当前可请求 observation.state、其子列和 action。未来数据能力另行扩展，不在算法里猜通道含义。

## 需要模板或初始化时

可以增加必要的参数、模板和辅助文件，不规定格式。用可选入口一次加载：

```python
from pathlib import Path

def load(directory: Path):
    return MyMethod(directory)  # 自己实现，加载本目录必要资源

class MyMethod:
    required_columns = ["observation.state", "action"]

    def __init__(self, directory):
        ...

    def segment(self, episode):
        ...  # 返回切分点或 None
```

不写 load 时直接使用模块的 segment；写了 load 则使用它返回的对象。两种方式都不需要修改 Praxis 核心代码。默认从当前 meta/info.json 读取通道信息；对象或模块可提供 channels 以使用已核对的通道定义。

## 探索交接

在代码注释或简短 README 中记录人工范例 ID/来源、方法依据、实际验证结果及未解决问题。没有固定两范例要求；别把范例验证当作独立测试。参数可以写在代码里，模板只在需要时附带。不要求额外 manifest、验证 JSON 或另一套协议。

方法运行所需资源放在任务目录或已安装依赖中，不依赖历史实验路径。代码是受信任的本地程序，不在沙箱中运行；不得主动读取测试标签、历史答案或修改自己与模板。随机方法应固定种子。Praxis 自动记录任务清单、代码与资源哈希、环境和预测；无需 Agent 手写冻结清单。

## 批量使用

```json
{"tasks": [{"task_id": "my_task", "dataset": "原始任务目录名", "target_ids": [2, 3]}]}
```

```bash
praxis run --data-root /path/to/data --tasks tasks.json \
  --rules-root task_rules --output outputs/run1
```

执行器加载 task_rules/my_task/rule.py。清单可选提供 exemplar_ids 用于检查与目标不重合；这只是额外检查，不决定方法能否执行。旧 --split 参数继续有效。

普通返回值会保存为 review 或 abstain，review 不表示人工确认正确。R2 兼容适配器还会保留原有详细原因；新增方法不需要构造这套结果字典。GT 只在另外调用 praxis score 时读取。
