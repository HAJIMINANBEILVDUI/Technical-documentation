"""数据加载器：支持NIfTI和DICOM格式的CT影像读取。"""

import os
from pathlib import Path
from typing import Union, Dict, Any, Tuple, Optional, List

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from pydicom import dcmread
from pydicom.tag import Tag


class DataLoadError(Exception):
    """数据加载异常类"""
    pass


class DataLoader:
    """CT影像数据加载器，支持NIfTI和DICOM格式"""

    SUPPORTED_EXTENSIONS = ['.nii', '.nii.gz', '.nrrd', '.dcm']

    LOBE_LABELS = {
        0: 'background',
        1: 'left_upper_lobe',
        2: 'left_lower_lobe',
        3: 'right_upper_lobe',
        4: 'right_middle_lobe',
        5: 'right_lower_lobe'
    }

    @staticmethod
    def is_nifti_file(file_path: str) -> bool:
        """检查文件是否为NIfTI格式"""
        ext = Path(file_path).suffix.lower()
        return ext in ['.nii', '.nii.gz']

    @staticmethod
    def is_dicom_dir(path: str) -> bool:
        """检查路径是否为DICOM目录"""
        path_obj = Path(path)
        if path_obj.is_file():
            return False

        try:
            dcm_files = list(path_obj.glob('*.dcm'))
            if len(dcm_files) > 0:
                return True
            for item in path_obj.iterdir():
                if item.is_dir():
                    if DataLoader.is_dicom_dir(str(item)):
                        return True
        except:
            pass
        return False

    @staticmethod
    def load_image(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        加载CT影像

        Args:
            file_path: 影像文件路径（支持NIfTI和DICOM目录）

        Returns:
            Tuple[CT影像数组 (D, H, W), 元数据字典]

        Raises:
            DataLoadError: 文件不存在或格式不支持
        """
        if not os.path.exists(file_path):
            raise DataLoadError(f"文件不存在: {file_path}")

        if DataLoader.is_nifti_file(file_path):
            return DataLoader._load_nifti(file_path)
        elif DataLoader.is_dicom_dir(file_path):
            return DataLoader._load_dicom(file_path)
        else:
            raise DataLoadError(
                f"不支持的文件格式: {file_path}\n"
                f"支持的格式: {', '.join(DataLoader.SUPPORTED_EXTENSIONS)}"
            )

    @staticmethod
    def _load_nifti(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载NIfTI格式影像"""
        try:
            nib_image = nib.load(file_path)
            image_data = np.ascontiguousarray(nib_image.get_fdata(), dtype=np.float32)

            spacing = nib_image.header.get_zooms()
            origin = nib_image.affine[:3, 3]

            direction_matrix = nib_image.affine[:3, :3]
            direction = np.array([
                [direction_matrix[0, 0], direction_matrix[0, 1], direction_matrix[0, 2]],
                [direction_matrix[1, 0], direction_matrix[1, 1], direction_matrix[1, 2]],
                [direction_matrix[2, 0], direction_matrix[2, 1], direction_matrix[2, 2]]
            ])

            metadata = {
                'spacing': tuple(float(spacing[i]) for i in range(min(len(spacing), 3))),
                'origin': tuple(float(origin[i]) for i in range(3)),
                'direction': direction,
                'shape': image_data.shape,
                'dtype': str(image_data.dtype),
                'modality': 'CT',
                'file_path': file_path,
                'format': 'nifti'
            }

            if len(image_data.shape) == 4:
                image_data = image_data[:, :, :, 0]
                metadata['shape'] = image_data.shape

            return image_data, metadata

        except Exception as e:
            raise DataLoadError(f"加载NIfTI文件失败: {str(e)}")

    @staticmethod
    def _load_dicom(dicom_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """加载DICOM格式影像"""
        try:
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)

            if len(dicom_files) == 0:
                reader = sitk.ImageSeriesReader()
                series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
                if len(series_IDs) == 0:
                    raise DataLoadError(f"未找到DICOM序列: {dicom_dir}")
                reader.SetSeriesOutputFiles(dicom_files, series_IDs[0])
                dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)

            reader.SetFileNames(dicom_files)
            sitk_image = reader.Execute()

            image_data = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
            image_data = np.transpose(image_data, (2, 1, 0))

            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            direction = np.array(sitk_image.GetDirection()).reshape(3, 3)

            metadata = {
                'spacing': tuple(float(spacing[i]) for i in range(min(len(spacing), 3))),
                'origin': tuple(float(origin[i]) for i in range(min(len(origin), 3))),
                'direction': direction,
                'shape': image_data.shape,
                'dtype': str(image_data.dtype),
                'modality': 'CT',
                'file_path': dicom_dir,
                'format': 'dicom'
            }

            return image_data, metadata

        except Exception as e:
            raise DataLoadError(f"加载DICOM文件失败: {str(e)}")

    @staticmethod
    def load_label(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        加载分割标签

        Args:
            file_path: 标签文件路径

        Returns:
            Tuple[标签数组 (D, H, W), 元数据字典]

        Raises:
            DataLoadError: 文件不存在或格式错误
        """
        if not os.path.exists(file_path):
            raise DataLoadError(f"标签文件不存在: {file_path}")

        try:
            if DataLoader.is_nifti_file(file_path):
                nib_label = nib.load(file_path)
                label_data = np.ascontiguousarray(
                    np.round(nib_label.get_fdata()).astype(np.uint8)
                )

                spacing = nib_label.header.get_zooms()
                metadata = {
                    'spacing': tuple(float(spacing[i]) for i in range(min(len(spacing), 3))),
                    'shape': label_data.shape,
                    'file_path': file_path,
                    'format': 'nifti'
                }

                if len(label_data.shape) == 4:
                    label_data = label_data[:, :, :, 0]

                return label_data, metadata
            else:
                raise DataLoadError(f"不支持的标签格式: {file_path}")

        except Exception as e:
            raise DataLoadError(f"加载标签文件失败: {str(e)}")

    @staticmethod
    def get_image_info(file_path: str) -> Dict[str, Any]:
        """
        获取影像元信息

        Args:
            file_path: 影像文件路径

        Returns:
            包含影像信息的字典
        """
        try:
            _, metadata = DataLoader.load_image(file_path)
            return {
                'file_path': metadata.get('file_path', file_path),
                'spacing': metadata.get('spacing'),
                'origin': metadata.get('origin'),
                'direction': metadata.get('direction'),
                'shape': metadata.get('shape'),
                'dtype': metadata.get('dtype'),
                'modality': metadata.get('modality', 'CT'),
                'format': metadata.get('format')
            }
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"获取影像信息失败: {str(e)}")

    @staticmethod
    def save_nifti(image: np.ndarray, output_path: str,
                   spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                   origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
        """
        保存影像为NIfTI格式

        Args:
            image: 影像数组
            output_path: 输出路径
            spacing: 体素间距
            origin: 原点位置

        Returns:
            保存的文件路径
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            affine = np.eye(4)
            for i in range(3):
                affine[i, i] = spacing[i]
                affine[i, 3] = origin[i]

            nib_image = nib.Nifti1Image(image.astype(np.float32), affine)
            nib.save(nib_image, str(output_path))

            return str(output_path)
        except Exception as e:
            raise DataLoadError(f"保存NIfTI文件失败: {str(e)}")

    @staticmethod
    def save_label(label: np.ndarray, output_path: str,
                  spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
        """
        保存标签为NIfTI格式

        Args:
            label: 标签数组
            output_path: 输出路径
            spacing: 体素间距
            origin: 原点位置

        Returns:
            保存的文件路径
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            affine = np.eye(4)
            for i in range(3):
                affine[i, i] = spacing[i]
                affine[i, 3] = origin[i]

            nib_label = nib.Nifti1Image(label.astype(np.uint8), affine)
            nib.save(nib_label, str(output_path))

            return str(output_path)
        except Exception as e:
            raise DataLoadError(f"保存标签文件失败: {str(e)}")

    @staticmethod
    def validate_lung_label(label: np.ndarray) -> Dict[str, Any]:
        """
        验证标签数据的正确性

        Args:
            label: 标签数组

        Returns:
            验证结果字典
        """
        valid_labels = set(range(6))
        unique_labels = set(np.unique(label))

        missing_labels = valid_labels - unique_labels
        invalid_labels = unique_labels - valid_labels

        lobe_volumes = {}
        for lobe_id, lobe_name in DataLoader.LOBE_LABELS.items():
            if lobe_id in unique_labels:
                lobe_volumes[lobe_name] = int(np.sum(label == lobe_id))

        return {
            'is_valid': len(invalid_labels) == 0,
            'has_all_lobes': 0 in unique_labels and len(unique_labels - {0}) == 5,
            'unique_labels': sorted(list(unique_labels)),
            'invalid_labels': sorted(list(invalid_labels)),
            'missing_lobes': [DataLoader.LOBE_LABELS[l] for l in missing_labels if l != 0],
            'lobe_volumes': lobe_volumes
        }

    @staticmethod
    def get_lobe_name(lobe_id: int) -> str:
        """获取肺叶名称"""
        return DataLoader.LOBE_LABELS.get(lobe_id, 'unknown')

    @staticmethod
    def get_lobe_color(lobe_id: int) -> Tuple[int, int, int]:
        """获取肺叶颜色 (RGB)"""
        colors = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (255, 255, 0),
            5: (255, 0, 255)
        }
        return colors.get(lobe_id, (128, 128, 128))
