"""主窗口：PyQt5图形用户界面主窗口"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar,
    QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QMessageBox, QSplitter, QStatusBar, QMenuBar, QMenu, QAction,
    QInputDialog, QDialog, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TrainingThread(QThread):
    """训练线程"""
    progress_updated = pyqtSignal(dict)
    training_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def run(self):
        try:
            result = self.trainer.train()
            self.training_finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class InferenceThread(QThread):
    """推理线程"""
    progress_updated = pyqtSignal(dict)
    inference_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, predictor, input_path, output_dir):
        super().__init__()
        self.predictor = predictor
        self.input_path = input_path
        self.output_dir = output_dir

    def run(self):
        try:
            if os.path.isdir(self.input_path):
                result = self.predictor.predict_directory(
                    self.input_path,
                    self.output_dir
                )
            else:
                result = self.predictor.predict(self.input_path)
            self.inference_finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("胸部CT肺叶自动分割系统")
        self.setGeometry(100, 100, 1400, 900)

        self.current_dataset_info = {}
        self.training_state = None
        self.inference_thread = None
        self.training_thread = None

        self._init_ui()
        self._create_menu()

    def _init_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_data_tab(), "数据管理")
        self.tabs.addTab(self._create_training_tab(), "模型训练")
        self.tabs.addTab(self._create_inference_tab(), "推理分割")
        self.tabs.addTab(self._create_visualization_tab(), "可视化")
        self.tabs.addTab(self._create_evaluation_tab(), "评估")

        main_layout.addWidget(self.tabs)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def _create_menu(self):
        """创建菜单"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")

        open_action = QAction("打开数据", self)
        open_action.triggered.connect(self._on_open_data)
        file_menu.addAction(open_action)

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _create_data_tab(self) -> QWidget:
        """创建数据管理标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        import_group = QGroupBox("数据导入")
        import_layout = QFormLayout()

        self.source_dir_edit = QLineEdit()
        self.source_dir_edit.setPlaceholderText("选择源数据目录...")
        import_btn = QPushButton("浏览...")
        import_btn.clicked.connect(self._on_browse_source_dir)

        source_layout = QHBoxLayout()
        source_layout.addWidget(self.source_dir_edit)
        source_layout.addWidget(import_btn)
        import_layout.addRow("源目录:", source_layout)

        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.5, 0.95)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setSuffix(" %")
        import_layout.addRow("训练集比例:", self.train_ratio_spin)

        import_btn2 = QPushButton("导入数据")
        import_btn2.clicked.connect(self._on_import_data)
        import_layout.addRow("", import_btn2)

        import_group.setLayout(import_layout)
        layout.addWidget(import_group)

        synthetic_group = QGroupBox("创建合成数据（测试用）")
        synthetic_layout = QFormLayout()

        self.num_samples_spin = QSpinBox()
        self.num_samples_spin.setRange(5, 100)
        self.num_samples_spin.setValue(10)
        synthetic_layout.addRow("样本数量:", self.num_samples_spin)

        create_synthetic_btn = QPushButton("创建合成数据")
        create_synthetic_btn.clicked.connect(self._on_create_synthetic)
        synthetic_layout.addRow("", create_synthetic_btn)

        synthetic_group.setLayout(synthetic_layout)
        layout.addWidget(synthetic_group)

        info_group = QGroupBox("数据集信息")
        info_layout = QVBoxLayout()

        self.dataset_info_text = QTextEdit()
        self.dataset_info_text.setReadOnly(True)
        self.dataset_info_text.setMaximumHeight(200)
        info_layout.addWidget(self.dataset_info_text)

        refresh_btn = QPushButton("刷新信息")
        refresh_btn.clicked.connect(self._on_refresh_dataset_info)
        info_layout.addWidget(refresh_btn)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()

        return widget

    def _create_training_tab(self) -> QWidget:
        """创建模型训练标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        config_group = QGroupBox("训练配置")
        config_layout = QFormLayout()

        self.dataset_id_spin = QSpinBox()
        self.dataset_id_spin.setRange(1, 999)
        self.dataset_id_spin.setValue(101)
        config_layout.addRow("数据集ID:", self.dataset_id_spin)

        self.nnunet_config_combo = QComboBox()
        self.nnunet_config_combo.addItems(['2d', '3d_lowres', '3d_fullres', '3d_cascade'])
        self.nnunet_config_combo.setCurrentText('3d_fullres')
        config_layout.addRow("nnU-Net配置:", self.nnunet_config_combo)

        self.fold_spin = QSpinBox()
        self.fold_spin.setRange(0, 4)
        self.fold_spin.setValue(0)
        config_layout.addRow("交叉验证折:", self.fold_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(250)
        config_layout.addRow("训练轮数:", self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(2)
        config_layout.addRow("批次大小:", self.batch_size_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda:0', 'cuda:1', 'cpu'])
        self.device_combo.setCurrentIndex(0)
        config_layout.addRow("计算设备:", self.device_combo)

        self.resume_check = QCheckBox()
        config_layout.addRow("恢复训练:", self.resume_check)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        control_group = QGroupBox("训练控制")
        control_layout = QHBoxLayout()

        self.start_train_btn = QPushButton("开始训练")
        self.start_train_btn.clicked.connect(self._on_start_training)
        control_layout.addWidget(self.start_train_btn)

        self.stop_train_btn = QPushButton("停止训练")
        self.stop_train_btn.clicked.connect(self._on_stop_training)
        self.stop_train_btn.setEnabled(False)
        control_layout.addWidget(self.stop_train_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout()

        self.train_progress_bar = QProgressBar()
        self.train_progress_bar.setMaximum(100)
        progress_layout.addWidget(self.train_progress_bar)

        self.train_log_text = QTextEdit()
        self.train_log_text.setReadOnly(True)
        self.train_log_text.setMaximumHeight(300)
        progress_layout.addWidget(self.train_log_text)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        return widget

    def _create_inference_tab(self) -> QWidget:
        """创建推理分割标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        input_group = QGroupBox("输入配置")
        input_layout = QFormLayout()

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择模型文件或目录...")
        model_btn = QPushButton("浏览...")
        model_btn.clicked.connect(self._on_browse_model)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(model_btn)
        input_layout.addRow("模型路径:", model_layout)

        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("选择输入影像或目录...")
        input_btn = QPushButton("浏览...")
        input_btn.clicked.connect(self._on_browse_input)

        input_layout2 = QHBoxLayout()
        input_layout2.addWidget(self.input_path_edit)
        input_layout2.addWidget(input_btn)
        input_layout.addRow("输入路径:", input_layout2)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择输出目录...")
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(self._on_browse_output)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_btn)
        input_layout.addRow("输出目录:", output_layout)

        self.postprocess_check = QCheckBox()
        self.postprocess_check.setChecked(True)
        input_layout.addRow("启用后处理:", self.postprocess_check)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        infer_btn = QPushButton("开始推理")
        infer_btn.clicked.connect(self._on_start_inference)
        layout.addWidget(infer_btn)

        result_group = QGroupBox("推理结果")
        result_layout = QVBoxLayout()

        self.inference_result_text = QTextEdit()
        self.inference_result_text.setReadOnly(True)
        self.inference_result_text.setMaximumHeight(300)
        result_layout.addWidget(self.inference_result_text)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        layout.addStretch()

        return widget

    def _create_visualization_tab(self) -> QWidget:
        """创建可视化标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        if not HAS_MATPLOTLIB:
            no_matplotlib_label = QLabel("请安装matplotlib以启用可视化功能")
            layout.addWidget(no_matplotlib_label)
            return widget

        load_group = QGroupBox("加载数据")
        load_layout = QFormLayout()

        self.vis_input_edit = QLineEdit()
        self.vis_input_edit.setPlaceholderText("选择CT影像...")
        vis_input_btn = QPushButton("浏览...")
        vis_input_btn.clicked.connect(self._on_browse_vis_input)

        vis_layout = QHBoxLayout()
        vis_layout.addWidget(self.vis_input_edit)
        vis_layout.addWidget(vis_input_btn)
        load_layout.addRow("CT影像:", vis_layout)

        self.seg_input_edit = QLineEdit()
        self.seg_input_edit.setPlaceholderText("选择分割结果（可选）...")
        vis_seg_btn = QPushButton("浏览...")
        vis_seg_btn.clicked.connect(self._on_browse_seg_input)

        vis_seg_layout = QHBoxLayout()
        vis_seg_layout.addWidget(self.seg_input_edit)
        vis_seg_layout.addWidget(vis_seg_btn)
        load_layout.addRow("分割结果:", vis_seg_layout)

        load_btn = QPushButton("加载并显示")
        load_btn.clicked.connect(self._on_load_visualization)
        load_layout.addRow("", load_btn)

        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        self.viewer_widget = QLabel("请加载影像以开始可视化")
        self.viewer_widget.setAlignment(Qt.AlignCenter)
        self.viewer_widget.setMinimumHeight(400)
        self.viewer_widget.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout.addWidget(self.viewer_widget)

        controls_group = QGroupBox("显示控制")
        controls_layout = QHBoxLayout()

        self.slice_slider = QSpinBox()
        self.slice_slider.setRange(0, 100)
        controls_layout.addWidget(QLabel("切片:"))
        controls_layout.addWidget(self.slice_slider)

        self.axis_combo = QComboBox()
        self.axis_combo.addItems(['横断面', '矢状面', '冠状面'])
        controls_layout.addWidget(QLabel("视图:"))
        controls_layout.addWidget(self.axis_combo)

        self.alpha_slider = QDoubleSpinBox()
        self.alpha_slider.setRange(0, 1)
        self.alpha_slider.setSingleStep(0.1)
        self.alpha_slider.setValue(0.5)
        controls_layout.addWidget(QLabel("透明度:"))
        controls_layout.addWidget(self.alpha_slider)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        return widget

    def _create_evaluation_tab(self) -> QWidget:
        """创建评估标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        input_group = QGroupBox("评估输入")
        input_layout = QFormLayout()

        self.prediction_dir_edit = QLineEdit()
        self.prediction_dir_edit.setPlaceholderText("选择预测结果目录...")
        pred_btn = QPushButton("浏览...")
        pred_btn.clicked.connect(self._on_browse_prediction_dir)

        pred_layout = QHBoxLayout()
        pred_layout.addWidget(self.prediction_dir_edit)
        pred_layout.addWidget(pred_btn)
        input_layout.addRow("预测结果:", pred_layout)

        self.ground_truth_dir_edit = QLineEdit()
        self.ground_truth_dir_edit.setPlaceholderText("选择金标准目录...")
        gt_btn = QPushButton("浏览...")
        gt_btn.clicked.connect(self._on_browse_ground_truth_dir)

        gt_layout = QHBoxLayout()
        gt_layout.addWidget(self.ground_truth_dir_edit)
        gt_layout.addWidget(gt_btn)
        input_layout.addRow("金标准:", gt_layout)

        self.report_path_edit = QLineEdit()
        self.report_path_edit.setText("evaluation_report.json")
        input_layout.addRow("报告路径:", self.report_path_edit)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        eval_btn = QPushButton("开始评估")
        eval_btn.clicked.connect(self._on_start_evaluation)
        layout.addWidget(eval_btn)

        results_group = QGroupBox("评估结果")
        results_layout = QVBoxLayout()

        self.evaluation_table = QTableWidget()
        self.evaluation_table.setColumnCount(7)
        self.evaluation_table.setHorizontalHeaderLabels([
            '案例', 'Dice', 'IoU', 'Precision', 'Recall', 'Hausdorff', 'ASD'
        ])
        results_layout.addWidget(self.evaluation_table)

        self.eval_summary_text = QTextEdit()
        self.eval_summary_text.setReadOnly(True)
        self.eval_summary_text.setMaximumHeight(150)
        results_layout.addWidget(self.eval_summary_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        return widget

    def _on_open_data(self):
        """打开数据"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if dir_path:
            self.source_dir_edit.setText(dir_path)

    def _on_browse_source_dir(self):
        """浏览源目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择源数据目录")
        if dir_path:
            self.source_dir_edit.setText(dir_path)

    def _on_import_data(self):
        """导入数据"""
        source_dir = self.source_dir_edit.text()
        if not source_dir:
            QMessageBox.warning(self, "警告", "请选择源数据目录")
            return

        try:
            from src.engine.data_manager.dataset_manager import DatasetManager

            manager = DatasetManager('./data')
            result = manager.setup_from_directory(
                source_dir,
                train_ratio=self.train_ratio_spin.value()
            )

            QMessageBox.information(
                self, "成功",
                f"数据导入成功!\n训练样本: {result['num_train']}\n测试样本: {result['num_test']}"
            )

            self._on_refresh_dataset_info()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")

    def _on_create_synthetic(self):
        """创建合成数据"""
        try:
            from src.engine.data_manager.dataset_manager import DatasetManager

            manager = DatasetManager('./data')
            result = manager.create_synthetic_dataset(
                num_samples=self.num_samples_spin.value()
            )

            QMessageBox.information(
                self, "成功",
                f"合成数据创建成功!\n样本数量: {result['num_samples']}"
            )

            self._on_refresh_dataset_info()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建失败: {str(e)}")

    def _on_refresh_dataset_info(self):
        """刷新数据集信息"""
        try:
            from src.engine.data_manager.dataset_manager import DatasetManager

            manager = DatasetManager('./data')
            info = manager.get_dataset_info()

            info_text = json.dumps(info, indent=2, ensure_ascii=False)
            self.dataset_info_text.setPlainText(info_text)

        except Exception as e:
            self.dataset_info_text.setPlainText(f"获取信息失败: {str(e)}")

    def _on_browse_model(self):
        """浏览模型路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self.model_path_edit.setText(path)

    def _on_browse_input(self):
        """浏览输入路径"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择CT影像", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        if not path:
            path = QFileDialog.getExistingDirectory(self, "选择输入目录")

        if path:
            self.input_path_edit.setText(path)

    def _on_browse_output(self):
        """浏览输出目录"""
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir_edit.setText(path)

    def _on_start_inference(self):
        """开始推理"""
        model_path = self.model_path_edit.text()
        input_path = self.input_path_edit.text()
        output_dir = self.output_dir_edit.text()

        if not model_path or not input_path or not output_dir:
            QMessageBox.warning(self, "警告", "请填写所有路径")
            return

        try:
            from src.engine.inference.predictor import Predictor

            self.predictor = Predictor(
                model_path=model_path,
                dataset_id=self.dataset_id_spin.value()
            )

            self.inference_thread = InferenceThread(
                self.predictor, input_path, output_dir
            )
            self.inference_thread.inference_finished.connect(self._on_inference_finished)
            self.inference_thread.error_occurred.connect(self._on_inference_error)
            self.inference_thread.start()

            self.status_bar.showMessage("推理进行中...")
            self.inference_result_text.append("推理开始...\n")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理失败: {str(e)}")

    def _on_inference_finished(self, result):
        """推理完成"""
        self.status_bar.showMessage("推理完成")
        self.inference_result_text.append(f"\n推理完成!\n{json.dumps(result, indent=2)}")
        QMessageBox.information(self, "完成", "推理完成!")

    def _on_inference_error(self, error_msg):
        """推理错误"""
        self.status_bar.showMessage("推理失败")
        self.inference_result_text.append(f"\n错误: {error_msg}")
        QMessageBox.critical(self, "错误", f"推理失败: {error_msg}")

    def _on_browse_vis_input(self):
        """浏览可视化输入"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择CT影像", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        if path:
            self.vis_input_edit.setText(path)

    def _on_browse_seg_input(self):
        """浏览分割结果"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择分割结果", "", "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        if path:
            self.seg_input_edit.setText(path)

    def _on_load_visualization(self):
        """加载可视化"""
        self.viewer_widget.setText("可视化功能正在开发中...")

    def _on_browse_prediction_dir(self):
        """浏览预测结果目录"""
        path = QFileDialog.getExistingDirectory(self, "选择预测结果目录")
        if path:
            self.prediction_dir_edit.setText(path)

    def _on_browse_ground_truth_dir(self):
        """浏览金标准目录"""
        path = QFileDialog.getExistingDirectory(self, "选择金标准目录")
        if path:
            self.ground_truth_dir_edit.setText(path)

    def _on_start_evaluation(self):
        """开始评估"""
        pred_dir = self.prediction_dir_edit.text()
        gt_dir = self.ground_truth_dir_edit.text()
        report_path = self.report_path_edit.text()

        if not pred_dir or not gt_dir:
            QMessageBox.warning(self, "警告", "请选择预测结果和金标准目录")
            return

        try:
            self.evaluation_table.setRowCount(0)
            self.eval_summary_text.clear()
            self.eval_summary_text.append("正在评估...\n")

            QMessageBox.information(self, "完成", "评估功能正在开发中")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"评估失败: {str(e)}")

    def _on_start_training(self):
        """开始训练"""
        try:
            from src.engine.trainer.trainer import NNUNetTrainer

            self.trainer = NNUNetTrainer(
                dataset_id=self.dataset_id_spin.value(),
                config=self.nnunet_config_combo.currentText()
            )

            self.start_train_btn.setEnabled(False)
            self.stop_train_btn.setEnabled(True)

            self.train_log_text.append("训练已开始...\n")
            self.status_bar.showMessage("训练进行中...")

            QMessageBox.information(
                self, "提示",
                "训练功能需要nnU-Net库支持。\n"
                "请使用命令行脚本进行训练:\n"
                "python scripts/train_model.py"
            )

            self.start_train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练初始化失败: {str(e)}")
            self.start_train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)

    def _on_stop_training(self):
        """停止训练"""
        if self.trainer:
            self.trainer.stop_training()

        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.status_bar.showMessage("训练已停止")

    def _on_about(self):
        """关于"""
        QMessageBox.about(
            self,
            "关于",
            "胸部CT肺叶自动分割系统\n\n"
            "基于nnU-Net的深度学习医学影像分割系统\n"
            "支持胸部CT影像的肺叶自动分割\n\n"
            "版本: 1.0"
        )

    def closeEvent(self, event):
        """关闭窗口"""
        reply = QMessageBox.question(
            self, '确认',
            "确定要退出吗?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
