"""数据集下载和准备脚本：支持公开肺部CT数据集的下载和nnU-Net格式转换"""

import os
import sys
import json
import argparse
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import nibabel as nib
import numpy as np

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    description: str
    url: str
    expected_files: int
    task_id: int
    labels: Dict[str, str]


class PublicDatasetDownloader:
    """公开数据集下载器"""

    SUPPORTED_DATASETS = {
        'luna16': DatasetInfo(
            name='LUNA16',
            description='Lung Nodule Analysis 2016 Challenge Dataset',
            url='https://zenodo.org/record/2594443/files/luna16.zip',
            expected_files=888,
            task_id=101,
            labels={
                '0': 'background',
                '1': 'lung_nodule'
            }
        ),
        'mosmed': DatasetInfo(
            name='MosMedData',
            description='MosMedData: Chest CT Scans with COVID-19 Related Findings',
            url='https://academictorrents.com/technical.php?q=MosMedData+Chest+CT+Scans+with+COVID-19+Related+Findings',
            expected_files=50,
            task_id=102,
            labels={
                '0': 'background',
                '1': 'lung'
            }
        )
    }

    def __init__(self, cache_dir: str = './data_cache'):
        """
        初始化下载器

        Args:
            cache_dir: 下载缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, output_path: Path) -> bool:
        """
        下载文件

        Args:
            url: 下载URL
            output_path: 输出路径

        Returns:
            是否成功
        """
        print(f"正在下载: {url}")

        if url.startswith('https://drive.google.com'):
            return self._download_google_drive(url, output_path)
        elif url.endswith('.zip'):
            return self._download_zip(url, output_path)
        elif url.endswith('.tar.gz') or url.endswith('.tgz'):
            return self._download_tar(url, output_path)
        else:
            print(f"不支持的下载格式: {url}")
            return False

    def _download_google_drive(self, url: str, output_path: Path) -> bool:
        """下载Google Drive文件"""
        if not HAS_GDOWN:
            print("请安装gdown: pip install gdown")
            return False

        try:
            file_id = url.split('/d/')[-1].split('?')[0]
            gdown.download(
                f'https://drive.google.com/uc?id={file_id}',
                str(output_path),
                quiet=False
            )
            return True
        except Exception as e:
            print(f"下载失败: {e}")
            return False

    def _download_zip(self, url: str, output_path: Path) -> bool:
        """下载并解压ZIP文件"""
        import urllib.request

        try:
            urllib.request.urlretrieve(url, output_path)

            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(output_path.parent)

            output_path.unlink()
            return True

        except Exception as e:
            print(f"下载失败: {e}")
            return False

    def _download_tar(self, url: str, output_path: Path) -> bool:
        """下载并解压TAR文件"""
        import urllib.request

        try:
            urllib.request.urlretrieve(url, output_path)

            with tarfile.open(output_path, 'r:gz') as tar_ref:
                tar_ref.extractall(output_path.parent)

            output_path.unlink()
            return True

        except Exception as e:
            print(f"下载失败: {e}")
            return False


class LungLobeDataPreparer:
    """肺叶数据准备器"""

    LOBE_LABELS = {
        0: 'background',
        1: 'left_upper_lobe',
        2: 'left_lower_lobe',
        3: 'right_upper_lobe',
        4: 'right_middle_lobe',
        5: 'right_lower_lobe'
    }

    def __init__(
        self,
        dataset_root: str = './data',
        dataset_id: int = 101
    ):
        """
        初始化数据准备器

        Args:
            dataset_root: 数据集根目录
            dataset_id: nnU-Net数据集ID
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_id = dataset_id
        self.task_name = f"Task{self.dataset_id:03d}_LungLobeSegmentation"

        self._setup_directories()

    def _setup_directories(self):
        """创建目录结构"""
        self.raw_base = self.dataset_root / 'nnUNet_raw_data_base' / 'nnUNet_raw_data'
        self.preprocessed_base = self.dataset_root / 'nnUNet_preprocessed'
        self.model_base = self.dataset_root / 'nnUNet_trained_models'

        self.task_dir = self.raw_base / self.task_name

        for subdir in ['imagesTr', 'imagesTs', 'labelsTr']:
            (self.task_dir / subdir).mkdir(parents=True, exist_ok=True)

    def import_from_directory(
        self,
        source_dir: str,
        file_pattern: str = '*.nii.gz',
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        从目录导入数据

        Args:
            source_dir: 源目录
            file_pattern: 文件匹配模式
            train_ratio: 训练集比例

        Returns:
            导入结果
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"源目录不存在: {source_dir}")

        print(f"正在扫描目录: {source_dir}")

        image_files = []
        label_files = []

        for nii_file in sorted(source_path.rglob(file_pattern)):
            file_name = nii_file.name.lower()

            if any(keyword in file_name for keyword in ['_seg', '_label', '_lobe', '_gt', '_mask']):
                label_files.append(nii_file)
            elif '_0000' in file_name or '_ct' in file_name:
                image_files.append(nii_file)

        if not image_files:
            for nii_file in sorted(source_path.rglob('*.nii*')):
                file_name = nii_file.name.lower()
                if not any(k in file_name for k in ['_seg', '_label', '_lobe', '_gt', '_mask']):
                    image_files.append(nii_file)

        print(f"找到 {len(image_files)} 个影像文件和 {len(label_files)} 个标签文件")

        paired_files = self._match_files(image_files, label_files)
        print(f"成功匹配 {len(paired_files)} 个文件对")

        import random
        random.shuffle(paired_files)

        split_idx = int(len(paired_files) * train_ratio)
        train_pairs = paired_files[:split_idx]
        test_pairs = paired_files[split_idx:]

        print(f"训练集: {len(train_pairs)}, 测试集: {len(test_pairs)}")

        self._copy_files(train_pairs, 'imagesTr', 'labelsTr', 'train')
        self._copy_files(test_pairs, 'imagesTs', None, 'test')

        self._generate_dataset_json()

        return {
            'task_name': self.task_name,
            'task_dir': str(self.task_dir),
            'num_train': len(train_pairs),
            'num_test': len(test_pairs),
            'train_ratio': train_ratio
        }

    def _match_files(
        self,
        image_files: List[Path],
        label_files: List[Path]
    ) -> List[Tuple[Path, Optional[Path]]]:
        """匹配影像和标签文件"""
        paired = []
        used_labels = set()

        for img_file in image_files:
            img_stem = img_file.stem.replace('_0000', '').replace('_ct', '')

            matched_label = None
            for lbl_file in label_files:
                lbl_stem = lbl_file.stem.replace('_seg', '').replace('_label', '').replace('_lobe', '').replace('_gt', '').replace('_mask', '')

                if img_stem == lbl_stem or img_stem in lbl_stem or lbl_stem in img_stem:
                    if str(lbl_file) not in used_labels:
                        matched_label = lbl_file
                        used_labels.add(str(lbl_file))
                        break

            paired.append((img_file, matched_label))

        return paired

    def _copy_files(
        self,
        file_pairs: List[Tuple[Path, Optional[Path]]],
        image_subdir: str,
        label_subdir: Optional[str],
        split_name: str
    ):
        """复制文件到目标目录"""
        for idx, (img_file, lbl_file) in enumerate(file_pairs):
            case_name = f"case_{idx:04d}"

            try:
                img_dst = self.task_dir / image_subdir / f"{case_name}_0000.nii.gz"
                self._copy_nifti(img_file, img_dst)

                if lbl_file and label_subdir:
                    lbl_dst = self.task_dir / label_subdir / f"{case_name}.nii.gz"
                    self._copy_nifti(lbl_file, lbl_dst)

            except Exception as e:
                print(f"复制文件失败: {img_file} - {e}")

    def _copy_nifti(self, src: Path, dst: Path):
        """复制NIfTI文件"""
        if src.suffix == '.gz' and src.stem.endswith('.nii'):
            shutil.copy2(src, dst)
        else:
            img = nib.load(str(src))
            nib.save(img, str(dst))

    def _generate_dataset_json(self):
        """生成dataset.json"""
        training_cases = []

        for img_file in sorted((self.task_dir / 'imagesTr').glob('*.nii.gz')):
            case_name = img_file.stem.replace('_0000', '')
            lbl_file = self.task_dir / 'labelsTr' / f"{case_name}.nii.gz"

            if lbl_file.exists():
                training_cases.append({
                    'image': str(img_file.relative_to(self.task_dir)),
                    'label': str(lbl_file.relative_to(self.task_dir))
                })

        dataset_json = {
            'name': 'LungLobeSegmentation',
            'description': 'Chest CT Lung Lobe Segmentation Dataset',
            'reference': 'Public Dataset',
            'licence': 'CC BY-NC-SA',
            'release': '1.0',
            'tensorImageSize': '3D',
            'modality': {
                '0': 'CT'
            },
            'labels': self.LOBE_LABELS,
            'numTraining': len(training_cases),
            'training': training_cases
        }

        json_path = self.task_dir / 'dataset.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, indent=4, ensure_ascii=False)

        print(f"dataset.json 已生成: {json_path}")

    def create_synthetic_dataset(
        self,
        num_samples: int = 10,
        volume_size: Tuple[int, int, int] = (64, 128, 128)
    ) -> Dict[str, Any]:
        """
        创建合成数据集（用于测试）

        Args:
            num_samples: 样本数量
            volume_size: 体积大小 (D, H, W)

        Returns:
            创建结果
        """
        print(f"正在创建 {num_samples} 个合成样本...")

        import random

        for idx in range(num_samples):
            case_name = f"case_{idx:04d}"

            ct_volume = np.random.uniform(-1000, -200, volume_size).astype(np.float32)

            lung_mask = np.zeros(volume_size, dtype=np.uint8)
            center_z = volume_size[0] // 2
            center_y = volume_size[1] // 2
            center_x = volume_size[2] // 2

            for z in range(volume_size[0]):
                for y in range(volume_size[1]):
                    for x in range(volume_size[2]):
                        dist = ((z - center_z) ** 2 / (volume_size[0] // 3) ** 2 +
                               (y - center_y) ** 2 / (volume_size[1] // 3) ** 2 +
                               (x - center_x) ** 2 / (volume_size[2] // 3) ** 2)
                        if dist < 1:
                            lung_mask[z, y, x] = 1

            lobe_mask = np.zeros(volume_size, dtype=np.uint8)
            lobe_mask[(lung_mask == 1) & (center_x < volume_size[2] // 2)] = 1
            lobe_mask[(lung_mask == 1) & (center_x >= volume_size[2] // 2)] = 3

            noise = np.random.normal(0, 20, volume_size).astype(np.float32)
            ct_volume = ct_volume + noise * lung_mask
            ct_volume = np.clip(ct_volume, -1024, 400)

            img_path = self.task_dir / 'imagesTr' / f"{case_name}_0000.nii.gz"
            lbl_path = self.task_dir / 'labelsTr' / f"{case_name}.nii.gz"

            img_nifti = nib.Nifti1Image(ct_volume, np.eye(4))
            lbl_nifti = nib.Nifti1Image(lobe_mask, np.eye(4))

            nib.save(img_nifti, str(img_path))
            nib.save(lbl_nifti, str(lbl_path))

        self._generate_dataset_json()

        return {
            'task_name': self.task_name,
            'num_samples': num_samples,
            'volume_size': volume_size
        }

    def validate_dataset(self) -> Dict[str, Any]:
        """验证数据集完整性"""
        issues = []
        warnings = []

        if not self.task_dir.exists():
            return {'is_valid': False, 'issues': ['任务目录不存在']}

        images_tr = self.task_dir / 'imagesTr'
        labels_tr = self.task_dir / 'labelsTr'

        image_files = list(images_tr.glob('*.nii.gz'))
        label_files = list(labels_tr.glob('*.nii.gz'))

        if len(image_files) == 0:
            issues.append('训练影像为空')

        if len(label_files) == 0:
            warnings.append('训练标签为空')

        json_path = self.task_dir / 'dataset.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                dataset_info = json.load(f)

            if dataset_info.get('labels') != self.LOBE_LABELS:
                issues.append('标签定义不匹配')

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'num_images': len(image_files),
            'num_labels': len(label_files)
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        json_path = self.task_dir / 'dataset.json'

        info = {
            'task_name': self.task_name,
            'task_dir': str(self.task_dir),
            'dataset_id': self.dataset_id
        }

        if json_path.exists():
            with open(json_path, 'r') as f:
                info['dataset_json'] = json.load(f)

        images_tr = self.task_dir / 'imagesTr'
        labels_tr = self.task_dir / 'labelsTr'

        info['num_train'] = len(list(images_tr.glob('*.nii.gz')))
        info['num_labels'] = len(list(labels_tr.glob('*.nii.gz')))

        return info


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='肺叶分割数据集准备工具'
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        help='源数据目录'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='./data',
        help='数据集根目录'
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=101,
        help='nnU-Net数据集ID'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='训练集比例'
    )
    parser.add_argument(
        '--create_synthetic',
        action='store_true',
        help='创建合成数据集'
    )
    parser.add_argument(
        '--num_synthetic',
        type=int,
        default=10,
        help='合成样本数量'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='验证数据集'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='显示数据集信息'
    )

    args = parser.parse_args()

    preparer = LungLobeDataPreparer(
        dataset_root=args.dataset_root,
        dataset_id=args.dataset_id
    )

    if args.create_synthetic:
        result = preparer.create_synthetic_dataset(
            num_samples=args.num_synthetic
        )
        print(f"合成数据集创建完成: {result}")

    elif args.source_dir:
        result = preparer.import_from_directory(
            source_dir=args.source_dir,
            train_ratio=args.train_ratio
        )
        print(f"数据导入完成: {result}")

    if args.validate:
        result = preparer.validate_dataset()
        print(f"验证结果: {result}")

    if args.info:
        info = preparer.get_dataset_info()
        print(f"数据集信息:")
        print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
