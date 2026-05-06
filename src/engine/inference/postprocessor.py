"""分割结果后处理器：去除噪声、平滑边界、确保连通性"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import morphological_gradient, distance_transform_edt
from typing import Dict, Any, List, Optional, Tuple
from skimage import morphology


class PostprocessingError(Exception):
    """后处理异常类"""
    pass


class Postprocessor:
    """分割结果后处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化后处理器

        Args:
            config: 后处理配置字典
        """
        self.config = config or {}
        postprocess_config = self.config.get('postprocess', {})

        self.enable = postprocess_config.get('enable', True)
        self.remove_small_objects = postprocess_config.get('remove_small_objects', True)
        self.min_object_size = postprocess_config.get('min_object_size', 100)
        self.smooth_boundary = postprocess_config.get('smooth_boundary', True)
        self.smooth_iterations = postprocess_config.get('smooth_iterations', 2)
        self.ensure_connectivity = postprocess_config.get('ensure_connectivity', True)

    def remove_small_regions(
        self,
        segmentation: np.ndarray,
        min_size: Optional[int] = None
    ) -> np.ndarray:
        """
        去除小的孤立区域（噪声）

        Args:
            segmentation: 分割结果 (D, H, W)
            min_size: 最小区域大小（体素数）

        Returns:
            去噪后的分割结果
        """
        if min_size is None:
            min_size = self.min_object_size

        result = segmentation.copy()

        for label_id in range(1, 6):
            mask = (segmentation == label_id)
            if not np.any(mask):
                continue

            labeled, num_features = ndimage.label(mask)

            for region_id in range(1, num_features + 1):
                region_size = np.sum(labeled == region_id)
                if region_size < min_size:
                    result[labeled == region_id] = 0

        return result

    def smooth_boundaries(
        self,
        segmentation: np.ndarray,
        iterations: Optional[int] = None
    ) -> np.ndarray:
        """
        平滑分割边界

        Args:
            segmentation: 分割结果
            iterations: 平滑迭代次数

        Returns:
            平滑后的分割结果
        """
        if iterations is None:
            iterations = self.smooth_iterations

        result = segmentation.copy()

        binary_mask = segmentation > 0
        binary_smoothed = morphology.binary_closing(
            binary_mask,
            morphology.ball(1)
        )
        binary_smoothed = morphology.binary_opening(
            binary_smoothed,
            morphology.ball(1)
        )

        boundary = binary_mask & ~binary_smoothed
        result[boundary] = 0

        for i in range(iterations):
            for label_id in range(1, 6):
                mask = (segmentation == label_id)

                eroded = ndimage.binary_erosion(mask, iterations=2)
                dilated = ndimage.binary_dilation(eroded, iterations=2)

                boundary_region = mask & ~dilated
                result[boundary_region] = 0

        for label_id in range(1, 6):
            current_mask = (segmentation == label_id)
            if not np.any(current_mask):
                continue

            for _ in range(iterations):
                neighbor_sum = np.zeros_like(current_mask, dtype=np.float32)

                for offset in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                    shifted = np.roll(np.roll(np.roll(
                        current_mask.astype(np.float32),
                        offset[0], axis=0
                    ), offset[1], axis=1), offset[2], axis=2)
                    neighbor_sum += shifted

                current_mask = current_mask | (neighbor_sum >= 3)

            result[current_mask] = label_id

        return result

    def ensure_lobe_connectivity(
        self,
        segmentation: np.ndarray
    ) -> np.ndarray:
        """
        确保每个肺叶区域的连通性

        Args:
            segmentation: 分割结果

        Returns:
            连通性修正后的分割结果
        """
        result = segmentation.copy()

        for label_id in range(1, 6):
            mask = (segmentation == label_id)
            if not np.any(mask):
                continue

            labeled, num_components = ndimage.label(mask)

            if num_components > 1:
                sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
                max_size_idx = np.argmax(sizes) + 1

                result[labeled == max_size_idx] = label_id

                for component_id in range(1, num_components + 1):
                    if component_id != max_size_idx:
                        component_mask = (labeled == component_id)
                        component_size = np.sum(component_mask)

                        if component_size < self.min_object_size:
                            result[component_mask] = 0
                        else:
                            distances = distance_transform_edt(~mask)
                            nearest_label = self._find_nearest_label(
                                component_mask, mask, result, distances
                            )
                            result[component_mask] = nearest_label

        return result

    def _find_nearest_label(
        self,
        component_mask: np.ndarray,
        all_mask: np.ndarray,
        current_seg: np.ndarray,
        distances: np.ndarray
    ) -> int:
        """找到距离最近的肺叶标签"""
        if not np.any(component_mask):
            return 0

        coords = np.array(np.where(component_mask)).T

        nearest_label = 0
        min_distance = float('inf')

        for label_id in range(1, 6):
            label_mask = (current_seg == label_id)
            if not np.any(label_mask):
                continue

            for coord in coords[:100]:
                coord_tuple = tuple(coord)
                if distances[coord_tuple] < min_distance:
                    min_distance = distances[coord_tuple]
                    nearest_label = label_id

        return nearest_label

    def fill_holes(
        self,
        segmentation: np.ndarray,
        max_hole_size: int = 1000
    ) -> np.ndarray:
        """
        填充肺叶内部的空洞

        Args:
            segmentation: 分割结果
            max_hole_size: 最大填充空洞大小

        Returns:
            填充空洞后的分割结果
        """
        result = segmentation.copy()

        lung_mask = segmentation > 0
        binary_filled = ndimage.binary_fill_holes(lung_mask)

        holes = binary_filled & ~lung_mask

        labeled_holes, num_holes = ndimage.label(holes)

        for hole_id in range(1, num_holes + 1):
            hole_size = np.sum(labeled_holes == hole_id)
            if hole_size <= max_hole_size:
                hole_mask = (labeled_holes == hole_id)

                nearby_label = self._get_nearby_majority_label(
                    hole_mask, result
                )
                result[hole_mask] = nearby_label

        return result

    def _get_nearby_majority_label(
        self,
        hole_mask: np.ndarray,
        segmentation: np.ndarray
    ) -> int:
        """获取空洞附近的主要标签"""
        struct = ndimage.iterate_structure(
            ndimage.generate_binary_structure(3, 1),
            3
        )

        dilated = ndimage.binary_dilation(hole_mask, structure=struct)
        neighborhood = dilated & ~hole_mask

        if not np.any(neighborhood):
            return 0

        labels, counts = np.unique(
            segmentation[neighborhood],
            return_counts=True
        )

        labels = labels[labels > 0]
        counts = counts[labels > 0]

        if len(labels) == 0:
            return 0

        return labels[np.argmax(counts)]

    def enforce_anatomical_constraints(
        self,
        segmentation: np.ndarray
    ) -> np.ndarray:
        """
        强制执行解剖学约束

        Args:
            segmentation: 分割结果

        Returns:
            约束后的分割结果
        """
        result = segmentation.copy()

        z_size, y_size, x_size = segmentation.shape

        right_side = segmentation.copy()
        right_side[:, :, :x_size // 2] = 0

        left_side = segmentation.copy()
        left_side[:, :, x_size // 2:] = 0

        right_labels = [3, 4, 5]
        left_labels = [1, 2]

        for label_id in right_labels:
            right_mask = (right_side == label_id)
            left_mask = (left_side == label_id)

            result[left_mask & (segmentation == label_id)] = 0

        for label_id in left_labels:
            left_mask = (left_side == label_id)
            right_mask = (right_side == label_id)

            result[right_mask & (segmentation == label_id)] = 0

        return result

    def postprocess(
        self,
        segmentation: np.ndarray,
        apply_all: bool = True
    ) -> Dict[str, Any]:
        """
        完整的后处理流程

        Args:
            segmentation: 原始分割结果
            apply_all: 是否应用所有后处理步骤

        Returns:
            包含后处理结果和信息的字典
        """
        if not self.enable:
            return {
                'segmentation': segmentation,
                'applied_steps': [],
                'changes': {}
            }

        result = segmentation.copy()
        applied_steps = []
        changes = {}

        original_stats = self._compute_statistics(result)

        if self.remove_small_objects and apply_all:
            result = self.remove_small_regions(result)
            applied_steps.append('remove_small_regions')
            changes['remove_small_regions'] = {
                'before': original_stats,
                'after': self._compute_statistics(result)
            }
            original_stats = self._compute_statistics(result)

        if self.smooth_boundary and apply_all:
            result = self.smooth_boundaries(result)
            applied_steps.append('smooth_boundaries')
            changes['smooth_boundaries'] = {
                'before': original_stats,
                'after': self._compute_statistics(result)
            }
            original_stats = self._compute_statistics(result)

        if self.ensure_connectivity and apply_all:
            result = self.ensure_lobe_connectivity(result)
            applied_steps.append('ensure_connectivity')
            changes['ensure_connectivity'] = {
                'before': original_stats,
                'after': self._compute_statistics(result)
            }
            original_stats = self._compute_statistics(result)

        result = self.fill_holes(result)
        applied_steps.append('fill_holes')

        if apply_all:
            result = self.enforce_anatomical_constraints(result)
            applied_steps.append('enforce_anatomical_constraints')

        final_stats = self._compute_statistics(result)

        return {
            'segmentation': result,
            'applied_steps': applied_steps,
            'changes': changes,
            'statistics': {
                'before': original_stats,
                'after': final_stats
            }
        }

    def _compute_statistics(
        self,
        segmentation: np.ndarray
    ) -> Dict[str, Any]:
        """计算分割结果的统计信息"""
        lobe_names = {
            1: 'left_upper_lobe',
            2: 'left_lower_lobe',
            3: 'right_upper_lobe',
            4: 'right_middle_lobe',
            5: 'right_lower_lobe'
        }

        stats = {}
        total_lung = 0

        for label_id in range(1, 6):
            mask = (segmentation == label_id)
            voxel_count = int(np.sum(mask))
            stats[lobe_names[label_id]] = voxel_count
            total_lung += voxel_count

        stats['total_lung'] = total_lung
        stats['num_labels'] = len([v for v in stats.values() if v > 0])

        return stats

    def get_quality_metrics(
        self,
        segmentation: np.ndarray
    ) -> Dict[str, Any]:
        """
        获取分割质量指标

        Args:
            segmentation: 分割结果

        Returns:
            质量指标字典
        """
        metrics = {}

        lung_mask = segmentation > 0

        metrics['total_lung_voxels'] = int(np.sum(lung_mask))
        metrics['lobe_counts'] = {}
        metrics['lobe_percentages'] = {}

        total = np.sum(lung_mask)
        if total == 0:
            return metrics

        lobe_names = {
            1: 'left_upper_lobe',
            2: 'left_lower_lobe',
            3: 'right_upper_lobe',
            4: 'right_middle_lobe',
            5: 'right_lower_lobe'
        }

        for label_id, lobe_name in lobe_names.items():
            count = int(np.sum(segmentation == label_id))
            metrics['lobe_counts'][lobe_name] = count
            metrics['lobe_percentages'][lobe_name] = (count / total) * 100

        boundary = self._compute_boundary_pixels(segmentation)
        metrics['boundary_voxels'] = int(np.sum(boundary))

        metrics['expected_lobes_present'] = sum(
            1 for count in metrics['lobe_counts'].values() if count > 0
        )

        return metrics

    def _compute_boundary_pixels(self, segmentation: np.ndarray) -> np.ndarray:
        """计算边界像素"""
        boundary = np.zeros_like(segmentation, dtype=bool)

        for label_id in range(1, 6):
            mask = (segmentation == label_id)
            if not np.any(mask):
                continue

            eroded = ndimage.binary_erosion(mask)
            boundary_pixels = mask & ~eroded
            boundary |= boundary_pixels

        return boundary
