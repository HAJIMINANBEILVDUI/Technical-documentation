from pathlib import Path

# ---------------------- 项目根目录（不用改，自动识别当前仓库） ----------------------
project_root = Path("./")

# ---------------------- 完全对齐你UML架构的目录结构 ----------------------
dirs_to_create = [
    # 存储层（对应UML中的Storage）
    "data/raw/ct",               # FileStorage: 原始CT影像
    "data/processed/train",      # FileStorage: 预处理训练数据
    "data/processed/val",       # FileStorage: 预处理验证数据
    "data/processed/test",      # FileStorage: 预处理测试数据
    "models/trained/nnunet",     # ModelStorage: 训练好的nnU-Net模型
    "config/train",              # ConfigStorage: 训练配置文件
    "config/inference",          # ConfigStorage: 推理配置文件
    "config/app",                # ConfigStorage: 应用配置文件

    # 核心处理引擎（对应UML中的Engine）
    "src/engine/data_manager",   # DataManager模块
    "src/engine/preprocessor",   # Preprocessor模块
    "src/engine/trainer",         # Trainer模块（nnU-Net训练）
    "src/engine/inference",       # Inference模块（nnU-Net推理）
    "src/engine/visualizer",     # Visualizer模块
    "src/engine/evaluator",       # Evaluator模块

    # 图形用户界面（对应UML中的GUI）
    "src/gui",

    # 公共工具模块
    "src/utils",

    # 测试目录（对应每个核心模块）
    "tests/unit/data_manager",
    "tests/unit/preprocessor",
    "tests/unit/trainer",
    "tests/unit/inference",
    "tests/unit/visualizer",
    "tests/unit/evaluator",
    "tests/unit/gui",

    # 文档和脚本目录
    "docs",
    "scripts/train",
    "scripts/inference",
    "scripts/preprocess"
]

# ---------------------- 自动创建目录和__init__.py ----------------------
for dir_path_str in dirs_to_create:
    dir_path = project_root / dir_path_str
    # 创建目录（自动创建父目录，避免重复创建报错）
    dir_path.mkdir(parents=True, exist_ok=True)

    # 给Python包目录添加带模块说明的__init__.py
    if dir_path_str.startswith(("src/", "tests/")):
        init_file = dir_path / "__init__.py"
        # 根据目录名自动生成模块说明
        if "data_manager" in dir_path_str:
            doc = '"""数据管理模块：负责CT影像的导入、索引与读取。"""\n'
        elif "preprocessor" in dir_path_str:
            doc = '"""预处理模块：负责CT影像的重采样、归一化与数据增强。"""\n'
        elif "trainer" in dir_path_str:
            doc = '"""模型训练模块：封装nnU-Net的训练流程与日志管理。"""\n'
        elif "inference" in dir_path_str:
            doc = '"""模型推理模块：实现nnU-Net的肺叶分割推理与后处理。"""\n'
        elif "visualizer" in dir_path_str:
            doc = '"""可视化模块：实现CT影像与分割结果的2D/3D可视化。"""\n'
        elif "evaluator" in dir_path_str:
            doc = '"""评估模块：计算Dice、HD95等分割指标并生成报告。"""\n'
        elif "gui" in dir_path_str:
            doc = '"""图形用户界面模块：实现用户交互与流程控制。"""\n'
        elif "utils" in dir_path_str:
            doc = '"""公共工具模块：提供日志、文件操作等通用功能。"""\n'
        else:
            doc = f'"""测试用例：{dir_path_str.split("/")[-1]}模块的单元测试。"""\n'
        init_file.write_text(doc)

print(f"项目目录结构已创建完成！根目录：{project_root.resolve()}")