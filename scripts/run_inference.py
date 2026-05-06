"""nnU-Net肺叶分割推理脚本"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.inference.predictor import Predictor, PredictionResult
from src.engine.inference.postprocessor import Postprocessor
from src.engine.data_manager.data_loader import DataLoader
from src.engine.visualizer.visualizer import Visualizer2D, Visualizer3D
from src.engine.evaluator.evaluator import Evaluator


def run_inference(
    model_path: str,
    input_path: str,
    output_dir: str,
    dataset_id: int,
    config: str,
    use_postprocess: bool = True,
    save_overlay: bool = True,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    运行推理

    Args:
        model_path: 模型路径
        input_path: 输入影像路径
        output_dir: 输出目录
        dataset_id: 数据集ID
        config: 模型配置
        use_postprocess: 是否使用后处理
        save_overlay: 是否保存叠加图
        device: 计算设备

    Returns:
        推理结果
    """
    print("=" * 60)
    print("nnU-Net Lung Lobe Segmentation Inference")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    predictor = Predictor(
        model_path=model_path,
        dataset_id=dataset_id,
        config=config,
        device=device
    )

    postprocessor = Postprocessor() if use_postprocess else None

    visualizer_2d = Visualizer2D()
    visualizer_3d = Visualizer3D()

    input_path_obj = Path(input_path)

    if input_path_obj.is_file():
        results = run_single_inference(
            predictor, postprocessor,
            input_path, output_dir,
            visualizer_2d, visualizer_3d,
            save_overlay
        )
    else:
        results = run_batch_inference(
            predictor, postprocessor,
            input_path, output_dir,
            visualizer_2d, visualizer_3d,
            save_overlay
        )

    print("\n" + "=" * 60)
    print("Inference Complete")
    print("=" * 60)
    print(f"Total: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)

    return results


def run_single_inference(
    predictor: Predictor,
    postprocessor: Optional[Postprocessor],
    input_path: str,
    output_dir: str,
    visualizer_2d: Visualizer2D,
    visualizer_3d: Visualizer3D,
    save_overlay: bool
) -> Dict[str, Any]:
    """单张影像推理"""
    print(f"\n处理影像: {input_path}")

    start_time = time.time()

    try:
        image_data, metadata = DataLoader.load_image(input_path)
        spacing = metadata.get('spacing', (1.0, 1.0, 1.0))

        result = predictor.predict(input_path)

        if postprocessor:
            postprocess_result = postprocessor.postprocess(result.segmentation)
            result.segmentation = postprocess_result['segmentation']

        processing_time = time.time() - start_time

        case_name = Path(input_path).stem.replace('_0000', '')
        output_prefix = Path(output_dir) / case_name

        result.save(str(output_prefix) + '_segmentation.nii.gz', spacing)

        print(f"\n统计信息:")
        lobe_stats = predictor.get_lobe_statistics(result)
        for lobe_name, stats in lobe_stats.items():
            volume_mm3 = stats['volume_mm3'] * spacing[0] * spacing[1] * spacing[2]
            print(f"  {lobe_name}: {volume_mm3:.2f} mm³")

        if save_overlay:
            try:
                slice_idx = image_data.shape[0] // 2
                visualizer_2d.save_slice(
                    image_data,
                    result.segmentation,
                    str(output_prefix) + '_overlay.png',
                    slice_idx=slice_idx
                )
                print(f"叠加图已保存")
            except Exception as e:
                print(f"保存叠加图失败: {e}")

        try:
            visualizer_3d.save_3d_view(
                result.segmentation,
                str(output_prefix) + '_3d.png'
            )
            print(f"3D视图已保存")
        except Exception as e:
            print(f"保存3D视图失败: {e}")

        return {
            'total': 1,
            'success': 1,
            'failed': 0,
            'processing_time': processing_time,
            'output_prefix': str(output_prefix)
        }

    except Exception as e:
        print(f"推理失败: {e}")
        return {
            'total': 1,
            'success': 0,
            'failed': 1,
            'errors': [str(e)]
        }


def run_batch_inference(
    predictor: Predictor,
    postprocessor: Optional[Postprocessor],
    input_dir: str,
    output_dir: str,
    visualizer_2d: Visualizer2D,
    visualizer_3d: Visualizer3D,
    save_overlay: bool
) -> Dict[str, Any]:
    """批量推理"""
    input_path_obj = Path(input_dir)
    image_files = list(input_path_obj.glob('*.nii.gz')) + list(input_path_obj.glob('*.nii'))

    if not image_files:
        print(f"未找到NIfTI文件: {input_dir}")
        return {'total': 0, 'success': 0, 'failed': 0}

    print(f"\n找到 {len(image_files)} 个影像文件")

    results = []
    for i, image_file in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 处理: {image_file.name}")

        result = run_single_inference(
            predictor, postprocessor,
            str(image_file), output_dir,
            visualizer_2d, visualizer_3d,
            save_overlay
        )
        results.append(result)

    success = sum(1 for r in results if r['success'] > 0)
    failed = sum(1 for r in results if r['failed'] > 0)
    total_time = sum(r.get('processing_time', 0) for r in results)

    return {
        'total': len(image_files),
        'success': success,
        'failed': failed,
        'total_processing_time': total_time,
        'average_processing_time': total_time / len(image_files) if image_files else 0
    }


def evaluate_results(
    prediction_dir: str,
    ground_truth_dir: str,
    output_report: str
) -> Dict[str, Any]:
    """
    评估推理结果

    Args:
        prediction_dir: 预测结果目录
        ground_truth_dir: 金标准目录
        output_report: 输出报告路径

    Returns:
        评估结果
    """
    print("=" * 60)
    print("Evaluating Results")
    print("=" * 60)

    evaluator = Evaluator()

    pred_files = sorted(Path(prediction_dir).glob('*.nii.gz'))
    gt_files = sorted(Path(ground_truth_dir).glob('*.nii.gz'))

    predictions = []
    ground_truths = []
    case_ids = []

    for pred_file, gt_file in zip(pred_files, gt_files):
        try:
            pred_data, _ = DataLoader.load_label(str(pred_file))
            gt_data, _ = DataLoader.load_label(str(gt_file))

            predictions.append(pred_data)
            ground_truths.append(gt_data)
            case_ids.append(pred_file.stem)

        except Exception as e:
            print(f"加载文件失败: {pred_file} - {e}")

    if not predictions:
        print("没有可评估的结果")
        return {}

    eval_results = evaluator.evaluate_dataset(
        predictions,
        ground_truths,
        case_ids
    )

    summary = evaluator.create_evaluation_summary(
        [evaluator.calculate_metrics(predictions[i], ground_truths[i], case_ids[i])
         for i in range(len(predictions))]
    )
    print(summary)

    evaluator.save_report(eval_results, output_report)
    print(f"\n评估报告已保存: {output_report}")

    return eval_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='nnU-Net Lung Lobe Segmentation Inference'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入影像或目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='输出目录'
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=101,
        help='数据集ID'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='3d_fullres',
        help='模型配置'
    )
    parser.add_argument(
        '--no_postprocess',
        action='store_true',
        help='禁用后处理'
    )
    parser.add_argument(
        '--save_overlay',
        action='store_true',
        default=True,
        help='保存叠加图'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='计算设备'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='评估模式'
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        default=None,
        help='金标准目录（评估模式）'
    )
    parser.add_argument(
        '--report',
        type=str,
        default='evaluation_report.json',
        help='评估报告输出路径'
    )

    args = parser.parse_args()

    if args.evaluate:
        if not args.ground_truth:
            print("错误: 评估模式需要指定 --ground_truth")
            return

        evaluate_results(
            args.input,
            args.ground_truth,
            args.report
        )
    else:
        run_inference(
            model_path=args.model_path,
            input_path=args.input,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            config=args.config,
            use_postprocess=not args.no_postprocess,
            save_overlay=args.save_overlay,
            device=args.device
        )


if __name__ == '__main__':
    main()
