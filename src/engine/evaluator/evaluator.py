"""评估器：计算分割性能指标，包括Dice、IoU、Hausdorff距离等"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json

import numpy as np
from scipy.spatial.distance import directed_hausdorff


@dataclass
class EvaluationResult:
    """评估结果"""
    case_id: str
    metrics: Dict[str, float]
    per_class_metrics: Dict[int, Dict[str, float]]
    processing_time: float = 0.0

    def get_dice(self, class_id: Optional[int] = None) -> float:
        """获取Dice系数"""
        if class_id is None:
            return self.metrics.get('dice', 0.0)
        return self.per_class_metrics.get(class_id, {}).get('dice', 0.0)

    def get_iou(self, class_id: Optional[int] = None) -> float:
        """获取IoU"""
        if class_id is None:
            return self.metrics.get('iou', 0.0)
        return self.per_class_metrics.get(class_id, {}).get('iou', 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'case_id': self.case_id,
            'metrics': self.metrics,
            'per_class_metrics': {
                str(k): v for k, v in self.per_class_metrics.items()
            },
            'processing_time': self.processing_time
        }


class EvaluatorError(Exception):
    """评估异常类"""
    pass


class Evaluator:
    """分割结果评估器"""

    LOBE_NAMES = {
        0: 'background',
        1: 'left_upper_lobe',
        2: 'left_lower_lobe',
        3: 'right_upper_lobe',
        4: 'right_middle_lobe',
        5: 'right_lower_lobe'
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化评估器

        Args:
            config: 评估配置字典
        """
        self.config = config or {}
        self.results_cache: Dict[str, EvaluationResult] = {}

    @staticmethod
    def compute_dice(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        class_id: Optional[int] = None
    ) -> float:
        """
        计算Dice系数

        Args:
            prediction: 预测结果
            ground_truth: 金标准
            class_id: 类别ID，如果为None则计算所有类别的平均

        Returns:
            Dice系数
        """
        if class_id is not None:
            pred_mask = (prediction == class_id)
            gt_mask = (ground_truth == class_id)
        else:
            pred_mask = prediction > 0
            gt_mask = ground_truth > 0

        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)

        if union == 0:
            return 0.0

        dice = (2.0 * intersection) / union

        return float(dice)

    @staticmethod
    def compute_iou(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        class_id: Optional[int] = None
    ) -> float:
        """
        计算IoU (Intersection over Union)

        Args:
            prediction: 预测结果
            ground_truth: 金标准
            class_id: 类别ID

        Returns:
            IoU值
        """
        if class_id is not None:
            pred_mask = (prediction == class_id)
            gt_mask = (ground_truth == class_id)
        else:
            pred_mask = prediction > 0
            gt_mask = ground_truth > 0

        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)

        if union == 0:
            return 0.0

        iou = intersection / union

        return float(iou)

    @staticmethod
    def compute_precision(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        class_id: Optional[int] = None
    ) -> float:
        """
        计算精确率

        Args:
            prediction: 预测结果
            ground_truth: 金标准
            class_id: 类别ID

        Returns:
            精确率
        """
        if class_id is not None:
            pred_mask = (prediction == class_id)
            gt_mask = (ground_truth == class_id)
        else:
            pred_mask = prediction > 0
            gt_mask = ground_truth > 0

        true_positive = np.sum(pred_mask & gt_mask)
        predicted_positive = np.sum(pred_mask)

        if predicted_positive == 0:
            return 0.0

        precision = true_positive / predicted_positive

        return float(precision)

    @staticmethod
    def compute_recall(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        class_id: Optional[int] = None
    ) -> float:
        """
        计算召回率

        Args:
            prediction: 预测结果
            ground_truth: 金标准
            class_id: 类别ID

        Returns:
            召回率
        """
        if class_id is not None:
            pred_mask = (prediction == class_id)
            gt_mask = (ground_truth == class_id)
        else:
            pred_mask = prediction > 0
            gt_mask = ground_truth > 0

        true_positive = np.sum(pred_mask & gt_mask)
        actual_positive = np.sum(gt_mask)

        if actual_positive == 0:
            return 0.0

        recall = true_positive / actual_positive

        return float(recall)

    @staticmethod
    def compute_f1_score(precision: float, recall: float) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def compute_hausdorff_distance(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        class_id: Optional[int] = None,
        percentile: float = 95
    ) -> float:
        """
        计算Hausdorff距离

        Args:
            prediction: 预测结果
            ground_truth: 金标准
            class_id: 类别ID
            percentile: 百分位数（用于95% Hausdorff距离）

        Returns:
            Hausdorff距离（单位：体素）
        """
        if class_id is not None:
            pred_mask = (prediction == class_id)
            gt_mask = (ground_truth == class_id)
        else:
            pred_mask = prediction > 0
            gt_mask = ground_truth > 0

        pred_points = np.array(np.where(pred_mask)).T
        gt_points = np.array(np.where(gt_mask)).T

        if len(pred_points) == 0 or len(gt_points) == 0:
            return float('inf')

        try:
            forward = directed_hausdorff(pred_points, gt_points)[0]
            backward = directed_hausdorff(gt_points, pred_points)[0]

            hd = max(forward, backward)

            if percentile < 100:
                distances_pred_to_gt = np.linalg.norm(
                    pred_points[:, np.newaxis] - gt_points[np.newaxis],
                    axis=2
                )
                distances_pred_to_gt = np.min(distances_pred_to_gt, axis=1)
                hd = np.percentile(distances_pred_to_gt, percentile)

            return float(hd)

        except Exception:
            return float('inf')

    @staticmethod
    def compute_average_surface_distance(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        class_id: Optional[int] = None
    ) -> float:
        """
        计算平均表面距离 (ASD)

        Args:
            prediction: 预测结果
            ground_truth: 金标准
            class_id: 类别ID

        Returns:
            平均表面距离
        """
        if class_id is not None:
            pred_mask = (prediction == class_id)
            gt_mask = (ground_truth == class_id)
        else:
            pred_mask = prediction > 0
            gt_mask = ground_truth > 0

        pred_surface = Evaluator._get_surface_points(pred_mask)
        gt_surface = Evaluator._get_surface_points(gt_mask)

        if len(pred_surface) == 0 or len(gt_surface) == 0:
            return float('inf')

        try:
            from scipy.spatial.distance import cdist

            dist_pred_to_gt = cdist(pred_surface, gt_surface).min(axis=1)
            dist_gt_to_pred = cdist(gt_surface, pred_surface).min(axis=1)

            asd = (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)) / 2

            return float(asd)

        except Exception:
            return float('inf')

    @staticmethod
    def _get_surface_points(mask: np.ndarray) -> np.ndarray:
        """获取表面的体素点"""
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(mask)
        surface = mask & ~eroded

        return np.array(np.where(surface)).T

    def calculate_metrics(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        case_id: str = "unknown",
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> EvaluationResult:
        """
        计算所有性能指标

        Args:
            prediction: 预测的分割结果
            ground_truth: 金标准标签
            case_id: 案例ID
            spacing: 体素间距（用于计算物理距离）

        Returns:
            EvaluationResult对象
        """
        if prediction.shape != ground_truth.shape:
            raise EvaluatorError(
                f"形状不匹配: prediction={prediction.shape}, ground_truth={ground_truth.shape}"
            )

        per_class_metrics = {}

        for label_id in range(1, 6):
            pred_mask = (prediction == label_id)
            gt_mask = (ground_truth == label_id)

            if not np.any(gt_mask) and not np.any(pred_mask):
                continue

            dice = self.compute_dice(prediction, ground_truth, label_id)
            iou = self.compute_iou(prediction, ground_truth, label_id)
            precision = self.compute_precision(prediction, ground_truth, label_id)
            recall = self.compute_recall(prediction, ground_truth, label_id)
            f1 = self.compute_f1_score(precision, recall)
            hd = self.compute_hausdorff_distance(prediction, ground_truth, label_id)
            asd = self.compute_average_surface_distance(prediction, ground_truth, label_id)

            voxel_volume = spacing[0] * spacing[1] * spacing[2]

            per_class_metrics[label_id] = {
                'dice': dice,
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'hausdorff_distance': hd * voxel_volume,
                'hausdorff_distance_voxel': hd,
                'average_surface_distance': asd * voxel_volume,
                'average_surface_distance_voxel': asd,
                'volume_predicted': int(np.sum(pred_mask)) * voxel_volume,
                'volume_ground_truth': int(np.sum(gt_mask)) * voxel_volume,
                'volume_difference': (int(np.sum(pred_mask)) - int(np.sum(gt_mask))) * voxel_volume
            }

        overall_dice = self.compute_dice(prediction, ground_truth)
        overall_iou = self.compute_iou(prediction, ground_truth)
        overall_precision = self.compute_precision(prediction, ground_truth)
        overall_recall = self.compute_recall(prediction, ground_truth)
        overall_f1 = self.compute_f1_score(overall_precision, overall_recall)
        overall_hd = self.compute_hausdorff_distance(prediction, ground_truth)
        overall_asd = self.compute_average_surface_distance(prediction, ground_truth)

        metrics = {
            'dice': overall_dice,
            'iou': overall_iou,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'hausdorff_distance': overall_hd * (spacing[0] * spacing[1] * spacing[2]) ** (1/3),
            'average_surface_distance': overall_asd * (spacing[0] * spacing[1] * spacing[2]) ** (1/3)
        }

        if len(per_class_metrics) > 0:
            mean_dice = np.mean([m['dice'] for m in per_class_metrics.values()])
            mean_iou = np.mean([m['iou'] for m in per_class_metrics.values()])
            metrics['mean_dice'] = mean_dice
            metrics['mean_iou'] = mean_iou

        result = EvaluationResult(
            case_id=case_id,
            metrics=metrics,
            per_class_metrics=per_class_metrics
        )

        self.results_cache[case_id] = result

        return result

    def evaluate_dataset(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        case_ids: Optional[List[str]] = None,
        spacings: Optional[List[Tuple[float, float, float]]] = None
    ) -> Dict[str, Any]:
        """
        评估整个数据集

        Args:
            predictions: 预测结果列表
            ground_truths: 金标准列表
            case_ids: 案例ID列表
            spacings: 体素间距列表

        Returns:
            评估报告
        """
        if len(predictions) != len(ground_truths):
            raise EvaluatorError("预测结果和金标准数量不匹配")

        if case_ids is None:
            case_ids = [f"case_{i:04d}" for i in range(len(predictions))]

        if spacings is None:
            spacings = [(1.0, 1.0, 1.0)] * len(predictions)

        results = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            try:
                result = self.calculate_metrics(
                    pred, gt,
                    case_id=case_ids[i],
                    spacing=spacings[i]
                )
                results.append(result)
            except Exception as e:
                print(f"评估案例 {case_ids[i]} 失败: {e}")

        summary = self._compute_summary(results)

        return {
            'individual_results': [r.to_dict() for r in results],
            'summary': summary,
            'num_evaluated': len(results)
        }

    def _compute_summary(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """计算汇总统计"""
        if not results:
            return {}

        all_dice = [r.metrics.get('dice', 0) for r in results]
        all_iou = [r.metrics.get('iou', 0) for r in results]
        all_hd = [r.metrics.get('hausdorff_distance', 0) for r in results]

        per_class_summary = {}
        for label_id in range(1, 6):
            class_dice = [
                r.per_class_metrics.get(label_id, {}).get('dice', 0)
                for r in results
            ]
            if any(d > 0 for d in class_dice):
                per_class_summary[self.LOBE_NAMES[label_id]] = {
                    'mean_dice': float(np.mean(class_dice)),
                    'std_dice': float(np.std(class_dice)),
                    'min_dice': float(np.min(class_dice)),
                    'max_dice': float(np.max(class_dice))
                }

        return {
            'mean_dice': float(np.mean(all_dice)),
            'std_dice': float(np.std(all_dice)),
            'mean_iou': float(np.mean(all_iou)),
            'std_iou': float(np.std(all_iou)),
            'mean_hausdorff_distance': float(np.mean(all_hd)),
            'std_hausdorff_distance': float(np.std(all_hd)),
            'per_class_summary': per_class_summary
        }

    def compare_models(
        self,
        model1_results: List[Dict[str, Any]],
        model2_results: List[Dict[str, Any]],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """
        对比两个模型的性能

        Args:
            model1_results: 模型1的评估结果
            model2_results: 模型2的评估结果
            model1_name: 模型1名称
            model2_name: 模型2名称

        Returns:
            对比报告
        """
        def extract_metrics(results):
            return {
                'dice': [r['metrics']['dice'] for r in results],
                'iou': [r['metrics']['iou'] for r in results],
                'hd': [r['metrics']['hausdorff_distance'] for r in results]
            }

        m1_metrics = extract_metrics(model1_results)
        m2_metrics = extract_metrics(model2_results)

        comparison = {
            'dice': {
                model1_name: {
                    'mean': float(np.mean(m1_metrics['dice'])),
                    'std': float(np.std(m1_metrics['dice']))
                },
                model2_name: {
                    'mean': float(np.mean(m2_metrics['dice'])),
                    'std': float(np.std(m2_metrics['dice']))
                },
                'difference': float(np.mean(m1_metrics['dice']) - np.mean(m2_metrics['dice'])),
                'winner': model1_name if np.mean(m1_metrics['dice']) > np.mean(m2_metrics['dice']) else model2_name
            },
            'iou': {
                model1_name: {
                    'mean': float(np.mean(m1_metrics['iou'])),
                    'std': float(np.std(m1_metrics['iou']))
                },
                model2_name: {
                    'mean': float(np.mean(m2_metrics['iou'])),
                    'std': float(np.std(m2_metrics['iou']))
                },
                'difference': float(np.mean(m1_metrics['iou']) - np.mean(m2_metrics['iou'])),
                'winner': model1_name if np.mean(m1_metrics['iou']) > np.mean(m2_metrics['iou']) else model2_name
            },
            'hausdorff_distance': {
                model1_name: {
                    'mean': float(np.mean(m1_metrics['hd'])),
                    'std': float(np.std(m1_metrics['hd']))
                },
                model2_name: {
                    'mean': float(np.mean(m2_metrics['hd'])),
                    'std': float(np.std(m2_metrics['hd']))
                },
                'difference': float(np.mean(m1_metrics['hd']) - np.mean(m2_metrics['hd'])),
                'winner': model1_name if np.mean(m1_metrics['hd']) < np.mean(m2_metrics['hd']) else model2_name
            }
        }

        return comparison

    def analyze_errors(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        case_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        分析分割错误

        Args:
            prediction: 预测结果
            ground_truth: 金标准
            case_id: 案例ID

        Returns:
            错误分析报告
        """
        false_positive = (prediction > 0) & (ground_truth == 0)
        false_negative = (prediction == 0) & (ground_truth > 0)
        true_positive = (prediction > 0) & (ground_truth > 0)

        total_fp_voxels = int(np.sum(false_positive))
        total_fn_voxels = int(np.sum(false_negative))

        error_by_lobe = {}
        for label_id in range(1, 6):
            fp = np.sum((prediction == label_id) & (ground_truth != label_id))
            fn = np.sum((prediction != label_id) & (ground_truth == label_id))
            tp = np.sum((prediction == label_id) & (ground_truth == label_id))

            total = np.sum(ground_truth == label_id)

            error_by_lobe[self.LOBE_NAMES[label_id]] = {
                'false_positive_voxels': int(fp),
                'false_negative_voxels': int(fn),
                'true_positive_voxels': int(tp),
                'total_voxels': int(total),
                'fp_percentage': (fp / total * 100) if total > 0 else 0,
                'fn_percentage': (fn / total * 100) if total > 0 else 0
            }

        connected_fp, num_fp = Evaluator._get_connected_components(false_positive)
        connected_fn, num_fn = Evaluator._get_connected_components(false_negative)

        return {
            'case_id': case_id,
            'total_false_positive': total_fp_voxels,
            'total_false_negative': total_fn_voxels,
            'total_true_positive': int(np.sum(true_positive)),
            'error_by_lobe': error_by_lobe,
            'fp_regions': num_fp,
            'fn_regions': num_fn,
            'error_rate': (total_fp_voxels + total_fn_voxels) / max(
                np.sum(prediction > 0) + np.sum(ground_truth > 0), 1
            )
        }

    @staticmethod
    def _get_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """获取连通区域"""
        from scipy.ndimage import label
        labeled, num = label(mask)
        return labeled, num

    def save_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: str
    ):
        """
        保存评估报告

        Args:
            evaluation_results: 评估结果
            output_path: 输出路径
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=4, ensure_ascii=False)

    def create_evaluation_summary(
        self,
        results: List[EvaluationResult]
    ) -> str:
        """
        创建文本格式的评估摘要

        Args:
            results: 评估结果列表

        Returns:
            格式化的摘要字符串
        """
        if not results:
            return "No results to summarize."

        summary = self._compute_summary(results)

        lines = []
        lines.append("=" * 60)
        lines.append("LUNG LOBE SEGMENTATION EVALUATION SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        lines.append("Overall Metrics:")
        lines.append(f"  Mean Dice Score: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
        lines.append(f"  Mean IoU:        {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
        lines.append(f"  Mean HD:         {summary['mean_hausdorff_distance']:.2f} mm")
        lines.append("")

        lines.append("Per-Class Dice Scores:")
        for lobe_name, stats in summary.get('per_class_summary', {}).items():
            lines.append(
                f"  {lobe_name:20s}: {stats['mean_dice']:.4f} ± {stats['std_dice']:.4f}"
            )

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
