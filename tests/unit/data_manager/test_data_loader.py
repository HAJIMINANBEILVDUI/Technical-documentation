"""数据加载器单元测试"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

import nibabel as nib


class TestDataLoader:
    """测试DataLoader类"""

    def setup_method(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_test_nifti(self, shape=(32, 32, 32), data=None):
        """创建测试NIfTI文件"""
        if data is None:
            data = np.random.randn(*shape).astype(np.float32)

        img = nib.Nifti1Image(data, np.eye(4))
        path = os.path.join(self.test_dir, f"test_{np.random.randint(10000)}.nii.gz")
        nib.save(img, path)
        return path

    def test_load_nifti_image(self):
        """测试加载NIfTI图像"""
        from src.engine.data_manager.data_loader import DataLoader

        shape = (32, 48, 64)
        test_data = np.random.randn(*shape).astype(np.float32)
        test_path = self.create_test_nifti(shape=shape, data=test_data)

        image_data, metadata = DataLoader.load_image(test_path)

        assert image_data.shape == shape
        assert 'spacing' in metadata
        assert 'origin' in metadata
        assert metadata['format'] == 'nifti'

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        from src.engine.data_manager.data_loader import DataLoader, DataLoadError

        with pytest.raises(DataLoadError):
            DataLoader.load_image('/nonexistent/path/file.nii.gz')

    def test_save_nifti(self):
        """测试保存NIfTI"""
        from src.engine.data_manager.data_loader import DataLoader

        data = np.random.randn(32, 32, 32).astype(np.float32)
        output_path = os.path.join(self.test_dir, 'output.nii.gz')

        saved_path = DataLoader.save_nifti(
            data, output_path,
            spacing=(1.0, 1.0, 1.0)
        )

        assert os.path.exists(saved_path)

        loaded_data, _ = DataLoader.load_image(saved_path)
        assert loaded_data.shape == data.shape

    def test_get_lobe_name(self):
        """测试获取肺叶名称"""
        from src.engine.data_manager.data_loader import DataLoader

        assert DataLoader.get_lobe_name(1) == 'left_upper_lobe'
        assert DataLoader.get_lobe_name(5) == 'right_lower_lobe'
        assert DataLoader.get_lobe_name(99) == 'unknown'

    def test_get_lobe_color(self):
        """测试获取肺叶颜色"""
        from src.engine.data_manager.data_loader import DataLoader

        color1 = DataLoader.get_lobe_color(1)
        assert color1 == (255, 0, 0)

        color_default = DataLoader.get_lobe_color(99)
        assert color_default == (128, 128, 128)

    def test_validate_lung_label(self):
        """测试验证肺叶标签"""
        from src.engine.data_manager.data_loader import DataLoader

        label = np.zeros((32, 32, 32), dtype=np.uint8)
        label[10:20, 10:20, 10:20] = 1
        label[5:15, 15:25, 5:15] = 3

        result = DataLoader.validate_lung_label(label)

        assert 'is_valid' in result
        assert 'unique_labels' in result
        assert 1 in result['unique_labels']
        assert 3 in result['unique_labels']

    def test_is_nifti_file(self):
        """测试NIfTI文件判断"""
        from src.engine.data_manager.data_loader import DataLoader

        assert DataLoader.is_nifti_file('test.nii') == True
        assert DataLoader.is_nifti_file('test.nii.gz') == True
        assert DataLoader.is_nifti_file('test.dcm') == False
        assert DataLoader.is_nifti_file('test.nrrd') == False
