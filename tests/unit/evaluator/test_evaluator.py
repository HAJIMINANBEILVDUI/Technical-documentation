"""评估器单元测试"""

import pytest
import numpy as np
import tempfile
import os


class TestEvaluator:
    """测试Evaluator类"""

    def setup_method(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_compute_dice_perfect_match(self):
        """测试完美匹配的Dice系数"""
        from src.engine.evaluator.evaluator import Evaluator

        prediction = np.zeros((32, 32, 32), dtype=np.uint8)
        prediction[10:20, 10:20, 10:20] = 1

        ground_truth = np.zeros((32, 32, 32), dtype=np.uint8)
        ground_truth[10:20, 10:20, 10:20] = 1

        dice = Evaluator.compute_dice(prediction, ground_truth, class_id=1)

        assert dice == 1.0

    def test_compute_dice_no_overlap(self):
        """测试无重叠的Dice系数"""
        from src.engine.evaluator.evaluator import Evaluator

        prediction = np.zeros((32, 32, 32), dtype=np.uint8)
        prediction[5:10, 5:10, 5:10] = 1

        ground_truth = np.zeros((32, 32, 32), dtype=np.uint8)
        ground_truth[20:25, 20:25, 20:25] = 1

        dice = Evaluator.compute_dice(prediction, ground_truth, class_id=1)

        assert dice == 0.0

    def test_compute_dice_partial_overlap(self):
        """测试部分重叠的Dice系数"""
        from src.engine.evaluator.evaluator import Evaluator

        prediction = np.zeros((10, 10, 10), dtype=np.uint8)
        prediction[2:8, 2:8, 2:8] = 1

        ground_truth = np.zeros((10, 10, 10), dtype=np.uint8)
        ground_truth[4:10, 4:10, 4:10] = 1

        dice = Evaluator.compute_dice(prediction, ground_truth, class_id=1)

        assert 0 < dice < 1

    def test_compute_iou(self):
        """测试IoU计算"""
        from src.engine.evaluator.evaluator import Evaluator

        prediction = np.zeros((32, 32, 32), dtype=np.uint8)
        prediction[10:20, 10:20, 10:20] = 1

        ground_truth = np.zeros((32, 32, 32), dtype=np.uint8)
        ground_truth[10:20, 10:20, 10:20] = 1

        iou = Evaluator.compute_iou(prediction, ground_truth, class_id=1)

        assert iou == 1.0

    def test_compute_precision(self):
        """测试精确率计算"""
        from src.engine.evaluator.evaluator import Evaluator

        prediction = np.zeros((32, 32, 32), dtype=np.uint8)
        prediction[10:20, 10:20, 10:20] = 1

        ground_truth = np.zeros((32, 32, 32), dtype=np.uint8)
        ground_truth[10:25, 10:25, 10:25] = 1

        precision = Evaluator.compute_precision(prediction, ground_truth, class_id=1)

        assert 0 < precision < 1

    def test_compute_recall(self):
        """测试召回率计算"""
        from src.engine.evaluator.evaluator import Evaluator

        prediction = np.zeros((32, 32, 32), dtype=np.uint8)
        prediction[10:25, 10:25, 10:25] = 1

        ground_truth = np.zeros((32, 32, 32), dtype=np.uint8)
        ground_truth[10:20, 10:20, 10:20] = 1

        recall = Evaluator.compute_recall(prediction, ground_truth, class_id=1)

        assert 0 < recall < 1

    def test_compute_f1_score(self):
        """测试F1分数计算"""
        from src.engine.evaluator.evaluator import Evaluator

        precision = 0.8
        recall = 0.6

        f1 = Evaluator.compute_f1_score(precision, recall)

        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert np.isclose(f1, expected_f1)

    def test_compute_f1_score_zero(self):
        """测试F1分数零值"""
        from src.engine.evaluator.evaluator import Evaluator

        f1 = Evaluator.compute_f1_score(0, 0)

        assert f1 == 0.0

    def test_calculate_metrics(self):
        """测试完整指标计算"""
        from src.engine.evaluator.evaluator import Evaluator

        evaluator = Evaluator()

        prediction = np.zeros((32, 32, 32), dtype=np.uint8)
        prediction[10:20, 10:20, 10:20] = 1
        prediction[15:18, 15:18, 15:18] = 2

        ground_truth = np.zeros((32, 32, 32), dtype=np.uint8)
        ground_truth[10:20, 10:20, 10:20] = 1
        ground_truth[16:19, 16:19, 16:19] = 2

        result = evaluator.calculate_metrics(
            prediction, ground_truth,
            case_id="test_case"
        )

        assert result.case_id == "test_case"
        assert 'dice' in result.metrics
        assert 'iou' in result.metrics
        assert 'precision' in result.metrics
        assert 'recall' in result.metrics

    def test_get_dice(self):
        """测试获取Dice系数"""
        from src.engine.evaluator.evaluator import Evaluator

        evaluator = Evaluator()

        prediction = np.zeros((32, 32, 32), dtype=np.uint8)
        prediction[10:20, 10:20, 10:20] = 1

        ground_truth = np.zeros((32, 32, 32), dtype=np.uint8)
        ground_truth[10:20, 10:20, 10:20] = 1

        result = evaluator.calculate_metrics(
            prediction, ground_truth,
            case_id="test_case"
        )

        dice_overall = result.get_dice()
        dice_class = result.get_dice(class_id=1)

        assert 0 <= dice_overall <= 1
        assert 0 <= dice_class <= 1

    def test_shape_mismatch_error(self):
        """测试形状不匹配错误"""
        from src.engine.evaluator.evaluator import Evaluator, EvaluatorError

        evaluator = Evaluator()

        prediction = np.zeros((32, 32, 32), dtype=np.uint8)
        ground_truth = np.zeros((16, 16, 16), dtype=np.uint8)

        with pytest.raises(EvaluatorError):
            evaluator.calculate_metrics(prediction, ground_truth)

    def test_analyze_errors(self):
        """测试错误分析"""
        from src.engine.evaluator.evaluator import Evaluator

        evaluator = Evaluator()

        prediction = np.zeros((32, 32, 32), dtype=np.uint8)
        prediction[10:20, 10:20, 10:20] = 1
        prediction[25:30, 25:30, 25:30] = 1

        ground_truth = np.zeros((32, 32, 32), dtype=np.uint8)
        ground_truth[10:20, 10:20, 10:20] = 1

        error_analysis = evaluator.analyze_errors(prediction, ground_truth)

        assert 'total_false_positive' in error_analysis
        assert 'total_false_negative' in error_analysis
        assert error_analysis['total_false_positive'] > 0
