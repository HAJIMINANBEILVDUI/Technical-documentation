"""数据集管理器：组织nnU-Net标准数据集结构"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random

from .data_loader import DataLoader, DataLoadError


class DatasetManager:
    """数据集管理器，负责组织nnU-Net标准数据集结构"""

    def __init__(self, dataset_root: str, dataset_id: int = 101):
        """
        初始化数据集管理器

        Args:
            dataset_root: 数据集根目录
            dataset_id: 数据集ID（nnU-Net要求）
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_id = dataset_id
        self.task_name = f"Task{self.dataset_id:03d}_LungLobeSegmentation"

        self.raw_data_base = self.dataset_root / 'nnUNet_raw_data_base' / 'nnUNet_raw_data'
        self.preprocessed_base = self.dataset_root / 'nnUNet_preprocessed'
        self.model_output_base = self.dataset_root / 'nnUNet_trained_models'

    def create_directories(self):
        """创建nnU-Net标准目录结构"""
        dirs = [
            self.raw_data_base / self.task_name / 'imagesTr',
            self.raw_data_base / self.task_name / 'imagesTs',
            self.raw_data_base / self.task_name / 'labelsTr',
            self.raw_data_base / self.task_name / 'imagesFltr',
            self.preprocessed_base / self.task_name,
            self.model_output_base / self.task_name
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        return [str(d) for d in dirs]

    def organize_dataset(
        self,
        image_paths: List[str],
        label_paths: List[str],
        train_ratio: float = 0.8,
        task_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        组织数据集为nnU-Net标准结构

        Args:
            image_paths: 影像文件路径列表
            label_paths: 标签文件路径列表
            train_ratio: 训练集比例
            task_name: 自定义任务名称

        Returns:
            组织结果字典
        """
        if task_name:
            self.task_name = task_name

        self.create_directories()

        if len(image_paths) != len(label_paths):
            raise ValueError(
                f"影像数量({len(image_paths)})与标签数量({len(label_paths)})不匹配"
            )

        paired_paths = list(zip(image_paths, label_paths))
        random.shuffle(paired_paths)

        split_idx = int(len(paired_paths) * train_ratio)
        train_pairs = paired_paths[:split_idx]
        test_pairs = paired_paths[split_idx:]

        task_dir = self.raw_data_base / self.task_name

        for idx, (img_path, lbl_path) in enumerate(train_pairs):
            case_name = f"case_{idx:04d}"
            img_dst = task_dir / 'imagesTr' / f"{case_name}_0000.nii.gz"
            lbl_dst = task_dir / 'labelsTr' / f"{case_name}.nii.gz"

            try:
                if DataLoader.is_nifti_file(img_path):
                    shutil.copy(img_path, img_dst)
                else:
                    image_data, _ = DataLoader.load_image(img_path)
                    spacing = (1.0, 1.0, 1.0)
                    DataLoader.save_nifti(image_data, str(img_dst), spacing)

                if DataLoader.is_nifti_file(lbl_path):
                    shutil.copy(lbl_path, lbl_dst)
                else:
                    label_data, _ = DataLoader.load_label(lbl_path)
                    DataLoader.save_label(label_data, str(lbl_dst))
            except Exception as e:
                print(f"复制训练数据失败 {img_path}: {e}")

        for idx, (img_path, lbl_path) in enumerate(test_pairs):
            case_name = f"case_{idx + len(train_pairs):04d}"
            img_dst = task_dir / 'imagesTs' / f"{case_name}_0000.nii.gz"

            try:
                if DataLoader.is_nifti_file(img_path):
                    shutil.copy(img_path, img_dst)
                else:
                    image_data, _ = DataLoader.load_image(img_path)
                    DataLoader.save_nifti(image_data, str(img_dst))
            except Exception as e:
                print(f"复制测试数据失败 {img_path}: {e}")

        return {
            'task_name': self.task_name,
            'task_dir': str(task_dir),
            'num_train': len(train_pairs),
            'num_test': len(test_pairs),
            'train_ratio': train_ratio
        }

    def generate_dataset_json(
        self,
        name: str = "LungLobeSegmentation",
        description: str = "Chest CT Lung Lobe Segmentation",
        reference: str = "Public Dataset",
        licence: str = "CC BY-NC-SA",
        release: str = "1.0"
    ) -> str:
        """
        生成nnU-Net的dataset.json配置文件

        Args:
            name: 数据集名称
            description: 数据集描述
            reference: 参考来源
            licence: 许可证
            release: 版本号

        Returns:
            生成的JSON文件路径
        """
        task_dir = self.raw_data_base / self.task_name
        images_tr_dir = task_dir / 'imagesTr'
        labels_tr_dir = task_dir / 'labelsTr'

        if not images_tr_dir.exists() or not labels_tr_dir.exists():
            raise FileNotFoundError("数据集目录不存在，请先调用organize_dataset")

        training_cases = []
        image_files = sorted(images_tr_dir.glob('*.nii.gz'))

        for img_file in image_files:
            case_name = img_file.stem.replace('_0000', '')
            lbl_file = labels_tr_dir / f"{case_name}.nii.gz"

            if lbl_file.exists():
                training_cases.append({
                    'image': str(img_file.relative_to(task_dir)),
                    'label': str(lbl_file.relative_to(task_dir))
                })

        dataset_json = {
            'name': name,
            'description': description,
            'reference': reference,
            'licence': licence,
            'release': release,
            'tensorImageSize': '3D',
            'modality': {
                '0': 'CT'
            },
            'labels': {
                '0': 'background',
                '1': 'left_upper_lobe',
                '2': 'left_lower_lobe',
                '3': 'right_upper_lobe',
                '4': 'right_middle_lobe',
                '5': 'right_lower_lobe'
            },
            'numTraining': len(training_cases),
            'training': training_cases
        }

        json_path = task_dir / 'dataset.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, indent=4, ensure_ascii=False)

        return str(json_path)

    def validate_dataset(self) -> Dict[str, Any]:
        """
        验证数据集完整性

        Returns:
            验证结果字典
        """
        task_dir = self.raw_data_base / self.task_name
        images_tr_dir = task_dir / 'imagesTr'
        labels_tr_dir = task_dir / 'labelsTr'
        json_path = task_dir / 'dataset.json'

        issues = []
        warnings = []

        if not task_dir.exists():
            issues.append(f"任务目录不存在: {task_dir}")
            return {
                'is_valid': False,
                'issues': issues,
                'warnings': warnings
            }

        if not json_path.exists():
            issues.append("dataset.json文件不存在")
        else:
            with open(json_path, 'r') as f:
                dataset_info = json.load(f)

            expected_labels = {
                '0': 'background',
                '1': 'left_upper_lobe',
                '2': 'left_lower_lobe',
                '3': 'right_upper_lobe',
                '4': 'right_middle_lobe',
                '5': 'right_lower_lobe'
            }
            if dataset_info.get('labels') != expected_labels:
                issues.append("标签定义不匹配")

        if not images_tr_dir.exists():
            issues.append("训练影像目录不存在")
        elif not labels_tr_dir.exists():
            issues.append("训练标签目录不存在")
        else:
            image_files = list(images_tr_dir.glob('*.nii.gz'))
            label_files = list(labels_tr_dir.glob('*.nii.gz'))

            if len(image_files) == 0:
                issues.append("训练影像文件为空")
            if len(label_files) == 0:
                issues.append("训练标签文件为空")

            matched = 0
            unmatched_images = []
            for img_file in image_files:
                case_name = img_file.stem.replace('_0000', '')
                lbl_file = labels_tr_dir / f"{case_name}.nii.gz"
                if lbl_file.exists():
                    matched += 1
                else:
                    unmatched_images.append(img_file.name)

            if unmatched_images:
                warnings.append(f"有{len(unmatched_images)}个影像缺少对应标签")

            image_label_ratio = len(label_files) / max(len(image_files), 1)
            if image_label_ratio < 0.8:
                warnings.append(f"影像-标签匹配率较低: {image_label_ratio:.2%}")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'num_images': len(image_files) if images_tr_dir.exists() else 0,
            'num_labels': len(label_files) if labels_tr_dir.exists() else 0
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集信息

        Returns:
            数据集信息字典
        """
        task_dir = self.raw_data_base / self.task_name
        images_tr_dir = task_dir / 'imagesTr'
        labels_tr_dir = task_dir / 'labelsTr'
        images_ts_dir = task_dir / 'imagesTs'

        num_train = len(list(images_tr_dir.glob('*.nii.gz'))) if images_tr_dir.exists() else 0
        num_test = len(list(images_ts_dir.glob('*.nii.gz'))) if images_ts_dir.exists() else 0
        num_labels = len(list(labels_tr_dir.glob('*.nii.gz'))) if labels_tr_dir.exists() else 0

        json_path = task_dir / 'dataset.json'
        dataset_info = None
        if json_path.exists():
            with open(json_path, 'r') as f:
                dataset_info = json.load(f)

        return {
            'task_name': self.task_name,
            'task_dir': str(task_dir),
            'dataset_id': self.dataset_id,
            'num_train': num_train,
            'num_test': num_test,
            'num_labels': num_labels,
            'dataset_json': dataset_info
        }

    def setup_from_directory(
        self,
        source_dir: str,
        file_pattern: str = '*.nii.gz',
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        从目录批量导入数据

        Args:
            source_dir: 源数据目录
            file_pattern: 文件匹配模式
            train_ratio: 训练集比例

        Returns:
            设置结果
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"源目录不存在: {source_dir}")

        image_paths = []
        label_paths = []

        for file_path in sorted(source_path.rglob(file_pattern)):
            file_name = file_path.name.lower()

            if '_seg' in file_name or '_label' in file_name or '_lobe' in file_name:
                label_paths.append(str(file_path))
            elif '_0000' in file_name:
                image_paths.append(str(file_path))

        for file_path in sorted(source_path.rglob('*.nii')):
            if str(file_path) not in image_paths and str(file_path) not in label_paths:
                image_paths.append(str(file_path))

        label_patterns = ['_seg', '_label', '_lobe', '_gt']
        for img_path in image_paths[:]:
            img_name = Path(img_path).stem.replace('_0000', '')
            found_label = False

            for pattern in label_patterns:
                for ext in ['.nii.gz', '.nii']:
                    label_candidate = source_path / f"{img_name}{pattern}{ext}"
                    if label_candidate.exists():
                        if img_path not in label_paths:
                            label_paths.append(str(label_candidate))
                            found_label = True
                        break
                if found_label:
                    break

        if len(image_paths) != len(label_paths):
            print(f"警告: 找到{len(image_paths)}个影像和{len(label_paths)}个标签")

        result = self.organize_dataset(
            image_paths,
            label_paths,
            train_ratio
        )

        json_path = self.generate_dataset_json()
        result['dataset_json'] = json_path

        return result
