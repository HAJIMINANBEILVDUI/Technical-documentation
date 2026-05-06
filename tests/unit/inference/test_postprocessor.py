"""后处理器单元测试"""

import pytest
import numpy as np
import tempfile
import os


class TestPostprocessor:
    """测试Postprocessor类"""

    def setup_method(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()

        self.default_config = {
            'postprocess': {
                'enable': True,
                'remove_small_objects': True,
                'min_object_size': 100,
                'smooth_boundary': True,
                'smooth_iterations': 2,
                'ensure_connectivity': True
            }
        }

    def teardown_method(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init_default_config(self):
        """测试默认配置初始化"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor()

        assert postprocessor.enable == True
        assert postprocessor.min_object_size == 100

    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        from src.engine.inference.postprocessor import Postprocessor

        config = {
            'postprocess': {
                'enable': True,
                'min_object_size': 500,
                'smooth_iterations': 5
            }
        }

        postprocessor = Postprocessor(config)

        assert postprocessor.min_object_size == 500
        assert postprocessor.smooth_iterations == 5

    def test_remove_small_regions(self):
        """测试去除小区域"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor(self.default_config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[10:20, 10:20, 10:20] = 1
        segmentation[30:31, 30:31, 30:31] = 1

        result = postprocessor.remove_small_regions(segmentation, min_size=100)

        assert np.sum(result == 1) > 0
        assert np.sum(result == 1) < np.sum(segmentation == 1)

    def test_remove_small_regions_large_object(self):
        """测试保留大区域"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor(self.default_config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[10:30, 10:30, 10:30] = 1

        result = postprocessor.remove_small_regions(segmentation, min_size=100)

        assert np.sum(result == 1) == np.sum(segmentation == 1)

    def test_smooth_boundaries(self):
        """测试边界平滑"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor(self.default_config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[10:50, 10:50, 10:50] = 1

        result = postprocessor.smooth_boundaries(segmentation, iterations=1)

        assert result.shape == segmentation.shape

    def test_ensure_lobe_connectivity(self):
        """测试确保连通性"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor(self.default_config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)

        segmentation[10:20, 10:20, 10:20] = 1
        segmentation[40:50, 40:50, 40:50] = 1

        result = postprocessor.ensure_lobe_connectivity(segmentation)

        assert result.shape == segmentation.shape

    def test_fill_holes(self):
        """测试填充空洞"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor(self.default_config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[10:50, 10:50, 10:50] = 1
        segmentation[25:35, 25:35, 25:35] = 0

        result = postprocessor.fill_holes(segmentation, max_hole_size=1000)

        hole_filled = np.sum((segmentation == 0) & (result == 1))
        assert hole_filled > 0

    def test_postprocess_disabled(self):
        """测试禁用后处理"""
        from src.engine.inference.postprocessor import Postprocessor

        config = {
            'postprocess': {
                'enable': False
            }
        }

        postprocessor = Postprocessor(config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[10:50, 10:50, 10:50] = 1

        result = postprocessor.postprocess(segmentation)

        assert result['segmentation'].shape == segmentation.shape
        assert len(result['applied_steps']) == 0

    def test_postprocess_enabled(self):
        """测试启用后处理"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor(self.default_config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[10:50, 10:50, 10:50] = 1

        result = postprocessor.postprocess(segmentation)

        assert 'segmentation' in result
        assert 'applied_steps' in result
        assert len(result['applied_steps']) > 0

    def test_enforce_anatomical_constraints(self):
        """测试解剖学约束"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor(self.default_config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[10:50, 10:50, 0:32] = 1
        segmentation[10:50, 10:50, 32:64] = 2

        result = postprocessor.enforce_anatomical_constraints(segmentation)

        assert result.shape == segmentation.shape

    def test_get_quality_metrics(self):
        """测试获取质量指标"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor(self.default_config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[10:40, 10:40, 0:32] = 1
        segmentation[10:40, 10:40, 32:64] = 3

        metrics = postprocessor.get_quality_metrics(segmentation)

        assert 'total_lung_voxels' in metrics
        assert 'lobe_counts' in metrics
        assert 'lobe_percentages' in metrics
        assert metrics['expected_lobes_present'] >= 0

    def test_compute_statistics(self):
        """测试计算统计信息"""
        from src.engine.inference.postprocessor import Postprocessor

        postprocessor = Postprocessor(self.default_config)

        segmentation = np.zeros((64, 64, 64), dtype=np.uint8)
        segmentation[10:40, 10:40, 10:40] = 1
        segmentation[20:50, 20:50, 20:50] = 3

        stats = postprocessor._compute_statistics(segmentation)

        assert 'total_lung' in stats
        assert 'num_labels' in stats
        assert stats['num_labels'] >= 0
