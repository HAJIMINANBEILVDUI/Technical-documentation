"""预处理器单元测试"""

import pytest
import numpy as np
import tempfile
import os


class TestPreprocessor:
    """测试Preprocessor类"""

    def setup_method(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()

        self.default_config = {
            'resample': {
                'target_spacing': [1.0, 1.0, 1.0],
                'interpolate': 'bspline'
            },
            'normalize': {
                'hu_clip_range': [-1000, 400],
                'normalize_range': [0, 1]
            },
            'crop': {
                'auto_crop_background': True,
                'margin': 10
            },
            'augmentation': {
                'enable': True,
                'rotation_range': [-15, 15],
                'flip_prob': 0.5,
                'noise_std': 0.01
            }
        }

    def teardown_method(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init_default_config(self):
        """测试使用默认配置初始化"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor()

        assert preprocessor.target_spacing == (1.0, 1.0, 1.0)
        assert preprocessor.hu_clip_min == -1000
        assert preprocessor.hu_clip_max == 400
        assert preprocessor.auto_crop == True

    def test_init_custom_config(self):
        """测试使用自定义配置初始化"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        config = {
            'resample': {
                'target_spacing': [2.0, 2.0, 2.0]
            },
            'normalize': {
                'hu_clip_range': [-500, 500]
            }
        }

        preprocessor = Preprocessor(config)

        assert preprocessor.target_spacing == (2.0, 2.0, 2.0)
        assert preprocessor.hu_clip_min == -500
        assert preprocessor.hu_clip_max == 500

    def test_normalize(self):
        """测试归一化"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        hu_values = np.array([-1200, -1000, -500, 0, 400, 600])
        normalized = preprocessor.normalize(hu_values)

        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert np.isclose(normalized[1], 0.0)
        assert np.isclose(normalized[4], 1.0)

    def test_normalize_custom_range(self):
        """测试自定义范围归一化"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        hu_values = np.array([-1000, -500, 0, 400])
        normalized = preprocessor.normalize(
            hu_values,
            clip_range=(-1000, 400),
            target_range=(0, 1)
        )

        assert np.isclose(normalized[0], 0.0)
        assert np.isclose(normalized[-1], 1.0)

    def test_denormalize(self):
        """测试反归一化"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        normalized = np.array([0.0, 0.5, 1.0])
        denormalized = preprocessor.denormalize(normalized)

        assert denormalized[0] == -1000
        assert denormalized[2] == 400

    def test_resample(self):
        """测试重采样"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        original_spacing = (2.0, 2.0, 2.0)
        original_shape = (50, 50, 50)
        image = np.random.randn(*original_shape).astype(np.float32)

        resampled, new_spacing, new_shape = preprocessor.resample(
            image,
            original_spacing,
            target_spacing=(1.0, 1.0, 1.0)
        )

        assert new_spacing == (1.0, 1.0, 1.0)
        assert np.allclose(new_shape, [100, 100, 100])

    def test_crop_lung_region(self):
        """测试肺部区域裁剪"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        image = np.zeros((100, 100, 100), dtype=np.float32)
        image[20:80, 20:80, 20:80] = -500

        cropped, crop_params = preprocessor.crop_lung_region(image, margin=5)

        assert cropped.shape[0] < image.shape[0]
        assert cropped.shape[1] < image.shape[1]
        assert cropped.shape[2] < image.shape[2]

    def test_crop_lung_region_no_lung(self):
        """测试无肺部区域的裁剪"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        image = np.random.randn(50, 50, 50).astype(np.float32)

        cropped, crop_params = preprocessor.crop_lung_region(image, margin=5)

        assert cropped.shape == image.shape

    def test_augment(self):
        """测试数据增强"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        image = np.random.randn(32, 64, 64).astype(np.float32)
        label = np.zeros((32, 64, 64), dtype=np.uint8)
        label[10:20, 20:40, 20:40] = 1

        augmented_image, augmented_label = preprocessor.augment(image, label, seed=42)

        assert augmented_image.shape == image.shape
        assert augmented_label.shape == label.shape

    def test_augment_disabled(self):
        """测试禁用数据增强"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        config = self.default_config.copy()
        config['augmentation']['enable'] = False

        preprocessor = Preprocessor(config)

        image = np.random.randn(32, 64, 64).astype(np.float32)
        label = np.zeros((32, 64, 64), dtype=np.uint8)

        augmented_image, _ = preprocessor.augment(image, label)

        np.testing.assert_array_equal(augmented_image, image)

    def test_pad_to_patch_size(self):
        """测试填充到patch大小"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        image = np.random.randn(32, 64, 64).astype(np.float32)
        patch_size = (128, 128, 128)

        padded, pad_params = preprocessor.pad_to_patch_size(image, patch_size)

        assert padded.shape == patch_size
        assert 'pad_before' in pad_params

    def test_get_stats(self):
        """测试获取统计信息"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        image = np.random.randn(32, 32, 32).astype(np.float32)

        stats = preprocessor.get_stats(image)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats

    def test_preprocess_full(self):
        """测试完整预处理流程"""
        from src.engine.preprocessor.preprocessor import Preprocessor

        preprocessor = Preprocessor(self.default_config)

        image = np.random.randn(32, 64, 64).astype(np.float32) * 100 - 500
        spacing = (2.0, 2.0, 2.0)

        result = preprocessor.preprocess_full(image, spacing)

        assert 'processed_image' in result
        assert 'original_shape' in result
        assert 'resampled_spacing' in result
        assert result['processed_image'].min() >= 0
        assert result['processed_image'].max() <= 1
