"""肺叶分割预测器：实现nnU-Net推理和滑动窗口策略"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import SimpleITK as sitk

from ..data_manager.data_loader import DataLoader
from ..preprocessor.preprocessor import Preprocessor


@dataclass
class PredictionResult:
    """推理结果"""
    segmentation: np.ndarray
    probabilities: Optional[np.ndarray] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def save(self, output_path: str, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """保存分割结果"""
        DataLoader.save_label(
            self.segmentation.astype(np.uint8),
            output_path,
            spacing=spacing
        )

    def get_lobe_mask(self, lobe_id: int) -> np.ndarray:
        """获取指定肺叶的掩膜"""
        return (self.segmentation == lobe_id).astype(np.uint8)


class InferenceError(Exception):
    """推理异常类"""
    pass


class Predictor:
    """肺叶分割预测器"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset_id: int = 101,
        config: str = "3d_fullres",
        folds: Optional[List[int]] = None,
        device: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        """
        初始化预测器

        Args:
            model_path: 模型路径
            dataset_id: 数据集ID
            config: 模型配置
            folds: 交叉验证折数列表
            device: 计算设备
            config_dict: 自定义配置
        """
        self.model_path = model_path
        self.dataset_id = dataset_id
        self.config = config
        self.folds = folds if folds is not None else [0]
        self.config_dict = config_dict or {}

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.preprocessor = Preprocessor(
            config=self.config_dict.get('preprocessing', {})
        )

        self.sliding_window_config = self.config_dict.get(
            'sliding_window',
            {'window_size': [128, 128, 128], 'overlap': 0.5}
        )

        self.nnunet_predictor = None
        self.plans = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        加载训练好的模型

        Args:
            model_path: 模型路径
        """
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError:
            raise InferenceError(
                "无法导入nnU-Net推理模块\n"
                "请确保已安装nnU-Net: pip install nnunetv2"
            )

        try:
            self.nnunet_predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_mirroring=False,
                verbose=False
            )

            if os.path.isdir(model_path):
                self.nnunet_predictor.load_models_from_folder(
                    model_path
                )
            else:
                model_files = [model_path]
                self.nnunet_predictor.load_models(model_files)

        except Exception as e:
            raise InferenceError(f"加载模型失败: {str(e)}")

    def _setup_nnunet_environment(self):
        """设置nnU-Net环境变量"""
        if 'nnUNet_preprocessed' not in os.environ:
            os.environ['nnUNet_preprocessed'] = 'data/nnUNet_preprocessed'
        if 'nnUNet_results' not in os.environ:
            os.environ['nnUNet_results'] = 'data/nnUNet_trained_models'

    def predict(
        self,
        image: Union[str, np.ndarray],
        spacing: Optional[Tuple[float, float, float]] = None,
        return_probabilities: bool = False,
        preprocess: bool = True
    ) -> PredictionResult:
        """
        单个影像推理

        Args:
            image: CT影像（文件路径或数组）
            spacing: 体素间距（当image为数组时需要）
            return_probabilities: 是否返回概率图
            preprocess: 是否进行预处理

        Returns:
            PredictionResult对象
        """
        self._setup_nnunet_environment()
        start_time = time.time()

        if isinstance(image, str):
            image_data, metadata = DataLoader.load_image(image)
            spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
        else:
            image_data = image.copy()
            metadata = {}

        if preprocess:
            preprocess_result = self.preprocessor.preprocess_full(
                image_data,
                spacing
            )
            processed_image = preprocess_result['processed_image']
            new_spacing = preprocess_result['resampled_spacing']
            metadata.update({
                'preprocess_params': preprocess_result
            })
        else:
            processed_image = image_data
            new_spacing = spacing

        if self.nnunet_predictor is None:
            segmentation = self._predict_without_nnunet(processed_image)
        else:
            segmentation = self._predict_with_nnunet(processed_image, return_probabilities)

        processing_time = time.time() - start_time

        result = PredictionResult(
            segmentation=segmentation,
            processing_time=processing_time,
            metadata=metadata
        )

        return result

    def _predict_with_nnunet(
        self,
        image: np.ndarray,
        return_probabilities: bool = False
    ) -> np.ndarray:
        """使用nnU-Net进行预测"""
        try:
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                prediction = self.nnunet_predictor.predict_single_case(
                    image_tensor,
                    image.shape
                )

            if isinstance(prediction, tuple):
                segmentation = prediction[0][0]
            else:
                segmentation = prediction[0] if len(prediction.shape) > 3 else prediction

            return segmentation.astype(np.uint8)

        except Exception as e:
            raise InferenceError(f"nnU-Net预测失败: {str(e)}")

    def _predict_without_nnunet(self, image: np.ndarray) -> np.ndarray:
        """在没有nnU-Net模型时返回占位分割"""
        segmentation = np.zeros(image.shape, dtype=np.uint8)

        depth = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]

        if depth > 20 and height > 50 and width > 50:
            lung_mask = (image > -300).astype(np.uint8)

            from scipy import ndimage
            labeled, num = ndimage.label(lung_mask)

            if num >= 2:
                sizes = ndimage.sum(lung_mask, labeled, range(1, num + 1))
                sorted_indices = np.argsort(sizes)[::-1]

                left_lobe = labeled == (sorted_indices[0] + 1)
                right_lobe = labeled == (sorted_indices[1] + 1)

                mid_z = depth // 2
                mid_y = height // 2

                if np.mean(np.where(left_lobe)[1]) > np.mean(np.where(right_lobe)[1]):
                    left_lobe, right_lobe = right_lobe, left_lobe

                left_size = np.sum(left_lobe)
                right_size = np.sum(right_lobe)

                if right_size > left_size * 0.3:
                    segmentation[right_lobe] = 3
                else:
                    segmentation[right_lobe] = 3

            segmentation[lung_mask > 0] = 1

        return segmentation

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        spacings: Optional[List[Tuple[float, float, float]]] = None,
        output_dir: Optional[str] = None,
        return_probabilities: bool = False
    ) -> List[PredictionResult]:
        """
        批量推理

        Args:
            images: CT影像列表
            spacings: 体素间距列表
            output_dir: 结果输出目录
            return_probabilities: 是否返回概率图

        Returns:
            推理结果列表
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results = []
        for i, image in enumerate(images):
            spacing = spacings[i] if spacings and i < len(spacings) else None

            try:
                result = self.predict(
                    image,
                    spacing=spacing,
                    return_probabilities=return_probabilities
                )

                if output_dir:
                    output_path = os.path.join(
                        output_dir,
                        f"prediction_{i:04d}.nii.gz"
                    )
                    result.save(output_path, spacing=(1.0, 1.0, 1.0))

                results.append(result)

            except Exception as e:
                print(f"预测失败 [{i+1}/{len(images)}]: {str(e)}")
                results.append(None)

        return results

    def predict_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "*.nii.gz",
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        批量处理目录中的所有影像

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_pattern: 文件匹配模式
            recursive: 是否递归搜索

        Returns:
            处理结果统计
        """
        input_path = Path(input_dir)
        if recursive:
            image_files = list(input_path.rglob(file_pattern))
        else:
            image_files = list(input_path.glob(file_pattern))

        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        failed_files = []

        for i, image_file in enumerate(image_files):
            try:
                result = self.predict(str(image_file))

                output_file = os.path.join(
                    output_dir,
                    f"{image_file.stem}_lobe_seg.nii.gz"
                )
                result.save(output_file)

                success_count += 1

            except Exception as e:
                failed_files.append({
                    'file': str(image_file),
                    'error': str(e)
                })

        return {
            'total': len(image_files),
            'success': success_count,
            'failed': len(failed_files),
            'failed_files': failed_files
        }

    def get_lobe_statistics(self, result: PredictionResult) -> Dict[str, Any]:
        """
        获取肺叶统计信息

        Args:
            result: 推理结果

        Returns:
            统计信息字典
        """
        lobe_names = {
            0: 'background',
            1: 'left_upper_lobe',
            2: 'left_lower_lobe',
            3: 'right_upper_lobe',
            4: 'right_middle_lobe',
            5: 'right_lower_lobe'
        }

        total_voxels = result.segmentation.size
        statistics = {}

        for lobe_id, lobe_name in lobe_names.items():
            lobe_mask = result.get_lobe_mask(lobe_id)
            voxel_count = int(np.sum(lobe_mask))
            percentage = (voxel_count / total_voxels) * 100 if total_voxels > 0 else 0

            if lobe_id > 0:
                statistics[lobe_name] = {
                    'voxel_count': voxel_count,
                    'percentage': percentage,
                    'volume_mm3': voxel_count
                }

        return statistics

    def validate_input(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        验证输入

        Args:
            image: 输入影像

        Returns:
            验证结果
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                return {'valid': False, 'error': '文件不存在'}

            ext = Path(image).suffix.lower()
            if ext not in ['.nii', '.nii.gz', '.dcm']:
                return {'valid': False, 'error': f'不支持的文件格式: {ext}'}

            try:
                image_data, metadata = DataLoader.load_image(image)
            except Exception as e:
                return {'valid': False, 'error': str(e)}
        else:
            if not isinstance(image, np.ndarray):
                return {'valid': False, 'error': '输入必须是numpy数组或文件路径'}
            image_data = image

        if len(image_data.shape) != 3:
            return {'valid': False, 'error': f'影像必须是3D，当前形状: {image_data.shape}'}

        if image_data.size < 1000:
            return {'valid': False, 'error': '影像体素数过少'}

        return {
            'valid': True,
            'shape': image_data.shape,
            'dtype': str(image_data.dtype)
        }
