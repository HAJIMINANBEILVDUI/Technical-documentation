"""nnU-Net模型训练脚本"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.trainer.trainer import NNUNetTrainer, TrainingState


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_environment():
    """设置nnU-Net环境变量"""
    os.environ['nnUNet_raw'] = 'data/nnUNet_raw_data_base/nnUNet_raw_data'
    os.environ['nnUNet_preprocessed'] = 'data/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = 'data/nnUNet_trained_models'


def train_model(
    dataset_id: int,
    config: str,
    fold: int,
    epochs: Optional[int],
    resume: bool,
    device: Optional[str]
) -> Dict[str, Any]:
    """
    训练模型

    Args:
        dataset_id: 数据集ID
        config: nnU-Net配置
        fold: 交叉验证折数
        epochs: 训练轮数
        resume: 是否恢复训练
        device: 计算设备

    Returns:
        训练结果
    """
    print("=" * 60)
    print("nnU-Net Lung Lobe Segmentation Training")
    print("=" * 60)
    print(f"Dataset ID: {dataset_id}")
    print(f"Configuration: {config}")
    print(f"Fold: {fold}")
    print(f"Epochs: {epochs or 'default'}")
    print(f"Resume: {resume}")
    print("=" * 60)

    setup_environment()

    trainer = NNUNetTrainer(
        dataset_id=dataset_id,
        task_name="LungLobeSegmentation",
        config=config
    )

    if device:
        trainer.device = torch.device(device)

    device_info = trainer.get_device_info()
    print(f"\nDevice Info:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    dataset_path = os.path.join(
        os.environ['nnUNet_raw'],
        f"Task{dataset_id:03d}_LungLobeSegmentation"
    )

    print(f"\n正在准备数据集: {dataset_path}")
    prepare_result = trainer.prepare_data(dataset_path)
    print(f"数据集准备完成:")
    print(f"  - 训练样本数: {prepare_result.get('num_training', 'N/A')}")
    print(f"  - 标签: {prepare_result.get('labels', {})}")

    print(f"\n开始训练...")
    result = trainer.train(
        fold=fold,
        epochs=epochs,
        resume=resume
    )

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Status: {result.get('status')}")
    print(f"Best Dice: {result.get('best_dice', 0):.4f}")
    print(f"Total Time: {result.get('elapsed_time', 0) / 3600:.2f} hours")
    print("=" * 60)

    return result


def plan_experiments(
    dataset_id: int,
    gpu_memory_gb: float = 8.0
) -> Dict[str, Any]:
    """
    规划实验

    Args:
        dataset_id: 数据集ID
        gpu_memory_gb: GPU内存（GB）

    Returns:
        规划结果
    """
    print("=" * 60)
    print("nnU-Net Experiment Planning")
    print("=" * 60)

    setup_environment()

    trainer = NNUNetTrainer(dataset_id=dataset_id)

    dataset_path = os.path.join(
        os.environ['nnUNet_raw'],
        f"Task{dataset_id:03d}_LungLobeSegmentation"
    )

    result = trainer.plan_training(
        dataset_path=dataset_path,
        gpu_memory_target_gb=gpu_memory_gb
    )

    print("规划完成:")
    print(f"  实验数量: {result.get('num_experiments', 0)}")

    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='nnU-Net Lung Lobe Segmentation Training'
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=101,
        help='nnU-Net数据集ID'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='3d_fullres',
        choices=['2d', '3d_lowres', '3d_fullres', '3d_cascade'],
        help='nnU-Net配置'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='交叉验证折数'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从检查点恢复训练'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='计算设备 (cuda:0, cpu)'
    )
    parser.add_argument(
        '--plan_only',
        action='store_true',
        help='仅规划实验，不训练'
    )
    parser.add_argument(
        '--gpu_memory',
        type=float,
        default=8.0,
        help='GPU内存（GB）'
    )

    args = parser.parse_args()

    if args.plan_only:
        plan_experiments(
            dataset_id=args.dataset_id,
            gpu_memory_gb=args.gpu_memory
        )
    else:
        train_model(
            dataset_id=args.dataset_id,
            config=args.config,
            fold=args.fold,
            epochs=args.epochs,
            resume=args.resume,
            device=args.device
        )


if __name__ == '__main__':
    main()
