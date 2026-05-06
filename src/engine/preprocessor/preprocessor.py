"""CT影像预处理器：负责重采样、归一化、裁剪和数据增强"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from typing import Tuple, Dict, Any, Optional
import random


class PreprocessorError(Exception):
    """预处理异常类"""
    pass


class Preprocessor:
    """CT影像预处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化预处理器

        Args:
            config: 预处理配置字典
        """
        self.config = config or {}
        self.resample_config = self.config.get('resample', {})
        self.normalize_config = self.config.get('normalize', {})
        self.crop_config = self.config.get('crop', {})
        self.augment_config = self.config.get('augmentation', {})

        self.target_spacing = tuple(self.resample_config.get(
            'target_spacing', [1.0, 1.0, 1.0]
        ))
        self.interpolate_method = self.resample_config.get(
            'interpolate', 'bspline'
        )

        self.hu_clip_min = self.normalize_config.get('hu_clip_range', [-1000, 400])[0]
        self.hu_clip_max = self.normalize_config.get('hu_clip_range', [-1000, 400])[1]
        self.normalize_range = self.normalize_config.get(
            'normalize_range', [0, 1]
        )

        self.auto_crop = self.crop_config.get('auto_crop_background', True)
        self.crop_margin = self.crop_config.get('margin', 10)

        self.augment_enabled = self.augment_config.get('enable', True)
        self.rotation_range = self.augment_config.get('rotation_range', [-15, 15])
        self.flip_prob = self.augment_config.get('flip_prob', 0.5)
        self.noise_std = self.augment_config.get('noise_std', 0.01)

    def resample(
        self,
        image: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Optional[Tuple[float, float, float]] = None,
        order: int = 3
    ) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[int, int, int]]:
        """
        重采样影像到目标体素间距

        Args:
            image: 原始影像 (D, H, W)
            original_spacing: 原始体素间距 (z, y, x)
            target_spacing: 目标体素间距，默认为配置中的值
            order: 插值阶数 (0: 最近邻, 1: 双线性, 3: 三次样条)

        Returns:
            Tuple[重采样后的影像, 新的体素间距, 新的形状]
        """
        if target_spacing is None:
            target_spacing = self.target_spacing

        target_spacing = tuple(target_spacing)
        original_spacing = tuple(original_spacing)

        zoom_factors = np.array(original_spacing) / np.array(target_spacing)
        new_shape = np.round(np.array(image.shape) * zoom_factors).astype(int)

        resampled = ndimage.zoom(
            image,
            zoom=zoom_factors,
            order=order,
            mode='constant',
            cval=0,
            prefilter=True
        )

        return resampled, target_spacing, resampled.shape

    def resample_sitk(
        self,
        image: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Optional[Tuple[float, float, float]] = None,
        interpolator: Optional[str] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        使用SimpleITK进行高质量重采样

        Args:
            image: 原始影像
            original_spacing: 原始体素间距
            target_spacing: 目标体素间距
            interpolator: 插值方法 ('nearest', 'linear', 'bspline')

        Returns:
            Tuple[重采样后的影像, 新的体素间距]
        """
        if target_spacing is None:
            target_spacing = self.target_spacing
        if interpolator is None:
            interpolator = self.interpolate_method

        sitk_interpolators = {
            'nearest': sitk.sitkNearestNeighbor,
            'linear': sitk.sitkLinear,
            'bspline': sitk.sitkBSpline
        }
        sitk_interp = sitk_interpolators.get(interpolator, sitk.sitkBSpline)

        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing(original_spacing)

        original_size = sitk_image.GetSize()
        original_spacing_arr = np.array(original_spacing)
        target_spacing_arr = np.array(target_spacing)
        new_size = (original_size * original_spacing_arr / target_spacing_arr).astype(int)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size.tolist())
        resampler.SetInterpolator(sitk_interp)
        resampler.SetDefaultPixelValue(0)

        resampled_sitk = resampler.Execute(sitk_image)
        resampled = sitk.GetArrayFromImage(resampled_sitk)

        return resampled, target_spacing

    def normalize(
        self,
        image: np.ndarray,
        clip_range: Optional[Tuple[float, float]] = None,
        target_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        归一化HU值到指定范围

        Args:
            image: CT影像
            clip_range: HU值裁剪范围，默认为配置中的值
            target_range: 目标归一化范围，默认为配置中的值

        Returns:
            归一化后的影像
        """
        if clip_range is None:
            clip_range = (self.hu_clip_min, self.hu_clip_max)
        if target_range is None:
            target_range = (self.normalize_range[0], self.normalize_range[1])

        image_clipped = np.clip(image, clip_range[0], clip_range[1])

        normalized = (image_clipped - clip_range[0]) / (clip_range[1] - clip_range[0])
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]

        return normalized.astype(np.float32)

    def denormalize(
        self,
        image: np.ndarray,
        clip_range: Tuple[float, float] = (-1000, 400)
    ) -> np.ndarray:
        """
        反归一化，将[0,1]范围的影像还原为HU值

        Args:
            image: 归一化后的影像
            clip_range: 原始HU值范围

        Returns:
            HU值影像
        """
        hu_image = image * (clip_range[1] - clip_range[0]) + clip_range[0]
        return hu_image.astype(np.float32)

    def crop_lung_region(
        self,
        image: np.ndarray,
        margin: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        自动裁剪肺部区域

        Args:
            image: CT影像
            margin: 边缘保留像素数

        Returns:
            Tuple[裁剪后的影像, 裁剪参数]
        """
        if margin is None:
            margin = self.crop_margin

        threshold = -300
        binary = image > threshold

        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            return image, {
                'bbox': ((0, image.shape[0]), (0, image.shape[1]), (0, image.shape[2])),
                'margin': margin,
                'original_shape': image.shape
            }

        largest_component = 0
        largest_size = 0
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled == i)
            if component_size > largest_size:
                largest_size = component_size
                largest_component = i

        lung_mask = (labeled == largest_component).astype(int)

        for i in range(3):
            sum_axis = tuple(j for j in range(3) if j != i)
            proj = np.any(lung_mask, axis=sum_axis)

            coords = np.where(proj)[0]
            if len(coords) > 0:
                min_coord = max(0, coords.min() - margin)
                max_coord = min(image.shape[i], coords.max() + margin + 1)

                if i == 0:
                    lung_mask = lung_mask[min_coord:max_coord, :, :]
                    image = image[min_coord:max_coord, :, :]
                elif i == 1:
                    lung_mask = lung_mask[:, min_coord:max_coord, :]
                    image = image[:, min_coord:max_coord, :]
                else:
                    lung_mask = lung_mask[:, :, min_coord:max_coord]
                    image = image[:, :, min_coord:max_coord]

        crop_params = {
            'margin': margin,
            'threshold': threshold,
            'original_shape': image.shape
        }

        return image, crop_params

    def pad_to_patch_size(
        self,
        image: np.ndarray,
        patch_size: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        将影像填充到指定的patch大小

        Args:
            image: 输入影像
            patch_size: 目标patch大小

        Returns:
            Tuple[填充后的影像, 填充参数]
        """
        current_shape = np.array(image.shape)
        target_shape = np.array(patch_size)

        pad_before = np.maximum(0, (target_shape - current_shape) // 2)
        pad_after = np.maximum(0, target_shape - current_shape - pad_before)

        if np.any(pad_before > 0) or np.any(pad_after > 0):
            pad_width = [
                (int(pad_before[0]), int(pad_after[0])),
                (int(pad_before[1]), int(pad_after[1])),
                (int(pad_before[2]), int(pad_after[2]))
            ]
            image_padded = np.pad(image, pad_width, mode='constant', constant_values=0)
        else:
            image_padded = image

        crop_params = {
            'pad_before': pad_before.tolist(),
            'pad_after': pad_after.tolist(),
            'original_shape': current_shape.tolist()
        }

        return image_padded, crop_params

    def augment(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        数据增强

        Args:
            image: 影像
            label: 标签（可选）
            seed: 随机种子

        Returns:
            Tuple[增强后的影像, 增强后的标签（如果提供）]
        """
        if not self.augment_enabled:
            return image, label

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if label is not None:
            combined = np.stack([image, label], axis=0)
        else:
            combined = image[np.newaxis, ...]

        if random.random() < self.flip_prob:
            flip_axis = random.choice([0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)])
            if isinstance(flip_axis, tuple):
                for ax in flip_axis:
                    combined = np.flip(combined, axis=ax + 1)
            else:
                combined = np.flip(combined, axis=flip_axis + 1)

        if self.rotation_range[0] != 0 or self.rotation_range[1] != 0:
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            axes = random.choice([(1, 2), (0, 2), (0, 1)])

            for i in range(combined.shape[0]):
                if label is not None and i == 1:
                    order = 0
                else:
                    order = 3
                combined[i] = ndimage.rotate(
                    combined[i],
                    angle=angle,
                    axes=axes,
                    reshape=False,
                    order=order,
                    mode='constant',
                    cval=0
                )

        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, combined[0:1].shape)
            combined[0:1] = combined[0:1] + noise

        if label is not None:
            return combined[0], combined[1]
        else:
            return combined[0], None

    def preprocess_full(
        self,
        image: np.ndarray,
        spacing: Tuple[float, float, float],
        crop: bool = True,
        normalize: bool = True,
        resample: bool = True
    ) -> Dict[str, Any]:
        """
        完整的预处理流程

        Args:
            image: 原始CT影像
            spacing: 原始体素间距
            crop: 是否裁剪
            normalize: 是否归一化
            resample: 是否重采样

        Returns:
            预处理结果字典
        """
        result = {
            'original_shape': image.shape,
            'original_spacing': spacing
        }

        if resample:
            image, new_spacing = self.resample_sitk(
                image, spacing, self.target_spacing
            )
            result['resampled_shape'] = image.shape
            result['resampled_spacing'] = new_spacing
            spacing = new_spacing
        else:
            result['resampled_shape'] = image.shape
            result['resampled_spacing'] = spacing

        if crop and self.auto_crop:
            image, crop_params = self.crop_lung_region(image)
            result['cropped_shape'] = image.shape
            result['crop_params'] = crop_params

        if normalize:
            image = self.normalize(image)
            result['normalized'] = True

        result['processed_image'] = image

        return result

    def preprocess_for_inference(
        self,
        image: np.ndarray,
        spacing: Tuple[float, float, float],
        original_shape: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        推理时的预处理

        Args:
            image: CT影像
            spacing: 体素间距
            original_shape: 原始形状（用于还原）

        Returns:
            预处理结果
        """
        if original_shape is None:
            original_shape = image.shape

        result = self.preprocess_full(image, spacing, crop=True, normalize=True, resample=True)
        result['original_shape'] = original_shape

        return result

    def get_stats(self, image: np.ndarray) -> Dict[str, float]:
        """
        获取影像统计信息

        Args:
            image: 影像

        Returns:
            统计信息字典
        """
        return {
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'median': float(np.median(image)),
            'percentile_1': float(np.percentile(image, 1)),
            'percentile_99': float(np.percentile(image, 99))
        }
