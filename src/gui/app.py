"""Streamlit Web应用入口点"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.inference.predictor import Predictor
from src.engine.inference.postprocessor import Postprocessor
from src.engine.data_manager.data_loader import DataLoader
from src.engine.visualizer.visualizer import Visualizer2D
from src.engine.evaluator.evaluator import Evaluator
from src.engine.preprocessor.preprocessor import Preprocessor

import numpy as np
import tempfile
import os


st.set_page_config(
    page_title="胸部CT肺叶分割系统",
    page_icon="🫁",
    layout="wide"
)


def main():
    st.title("🫁 胸部CT肺叶自动分割系统")
    st.markdown("基于nnU-Net深度学习框架的医学影像分割系统")

    st.sidebar.title("功能选择")
    app_mode = st.sidebar.selectbox(
        "选择功能",
        ["推理分割", "结果可视化", "性能评估"]
    )

    if app_mode == "推理分割":
        run_inference_tab()
    elif app_mode == "结果可视化":
        run_visualization_tab()
    elif app_mode == "性能评估":
        run_evaluation_tab()


def run_inference_tab():
    """推理分割标签页"""
    st.header("CT影像肺叶分割推理")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("输入配置")

        input_type = st.radio(
            "输入类型",
            ["上传文件", "输入路径"]
        )

        model_path = st.text_input(
            "模型路径",
            value="data/models/nnUNet/3d_fullres",
            help="训练好的模型目录"
        )

        use_postprocess = st.checkbox("启用后处理", value=True)

        if input_type == "上传文件":
            uploaded_file = st.file_uploader(
                "上传CT影像",
                type=['nii', 'nii.gz']
            )
            input_path = None
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as f:
                    f.write(uploaded_file.getvalue())
                    input_path = f.name
        else:
            input_path = st.text_input("CT影像路径")

        output_dir = st.text_input("输出目录", value="./results")

        if st.button("开始推理"):
            if not input_path:
                st.error("请提供输入影像")
                return

            with st.spinner("正在推理..."):
                try:
                    predictor = Predictor(
                        model_path=model_path,
                        device='cuda' if os.path.exists('/dev/nvidia0') else 'cpu'
                    )

                    result = predictor.predict(input_path)

                    if use_postprocess:
                        postprocessor = Postprocessor()
                        postprocess_result = postprocessor.postprocess(result.segmentation)
                        result.segmentation = postprocess_result['segmentation']

                    image_data, metadata = DataLoader.load_image(input_path)
                    spacing = metadata.get('spacing', (1.0, 1.0, 1.0))

                    lobe_stats = predictor.get_lobe_statistics(result)

                    with col2:
                        st.subheader("分割结果")

                        for lobe_name, stats in lobe_stats.items():
                            volume_mm3 = stats['volume_mm3'] * spacing[0] * spacing[1] * spacing[2]
                            st.metric(lobe_name, f"{volume_mm3:.2f} mm³")

                        st.success(f"处理完成! 耗时: {result.processing_time:.2f}秒")

                        save_path = os.path.join(output_dir, "prediction.nii.gz")
                        os.makedirs(output_dir, exist_ok=True)
                        result.save(save_path, spacing)
                        st.info(f"结果已保存: {save_path}")

                except Exception as e:
                    st.error(f"推理失败: {str(e)}")


def run_visualization_tab():
    """可视化标签页"""
    st.header("分割结果可视化")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("加载数据")

        image_file = st.file_uploader(
            "上传CT影像",
            type=['nii', 'nii.gz'],
            key="vis_image"
        )

        seg_file = st.file_uploader(
            "上传分割结果（可选）",
            type=['nii', 'nii.gz'],
            key="vis_seg"
        )

        slice_idx = st.slider("切片位置", 0, 100, 50)
        view_axis = st.selectbox("视图方向", ["横断面", "矢状面", "冠状面"])
        alpha = st.slider("叠加透明度", 0.0, 1.0, 0.5)

    with col2:
        st.subheader("可视化结果")

        if image_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as f:
                    f.write(image_file.getvalue())
                    image_path = f.name

                image_data, _ = DataLoader.load_image(image_path)

                if seg_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as f:
                        f.write(seg_file.getvalue())
                        seg_path = f.name

                    seg_data, _ = DataLoader.load_label(seg_path)

                    st.info(f"影像形状: {image_data.shape}, 切片数: {slice_idx}")
                else:
                    seg_data = None
                    st.info(f"影像形状: {image_data.shape}, 切片数: {slice_idx}")

            except Exception as e:
                st.error(f"加载失败: {str(e)}")
        else:
            st.info("请上传CT影像以开始可视化")


def run_evaluation_tab():
    """评估标签页"""
    st.header("分割性能评估")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("输入配置")

        pred_dir = st.text_input("预测结果目录")
        gt_dir = st.text_input("金标准目录")

        if st.button("开始评估"):
            if not pred_dir or not gt_dir:
                st.error("请提供预测结果和金标准目录")
                return

            with st.spinner("正在评估..."):
                try:
                    evaluator = Evaluator()

                    pred_files = list(Path(pred_dir).glob('*.nii.gz'))
                    gt_files = list(Path(gt_dir).glob('*.nii.gz'))

                    predictions = []
                    ground_truths = []
                    case_ids = []

                    for pred_file, gt_file in zip(pred_files, gt_files):
                        try:
                            pred_data, _ = DataLoader.load_label(str(pred_file))
                            gt_data, _ = DataLoader.load_label(str(gt_file))

                            predictions.append(pred_data)
                            ground_truths.append(gt_data)
                            case_ids.append(pred_file.stem)
                        except:
                            continue

                    if not predictions:
                        st.error("没有可评估的结果")
                        return

                    results = evaluator.evaluate_dataset(
                        predictions, ground_truths, case_ids
                    )

                    with col2:
                        st.subheader("评估结果")

                        summary = results['summary']

                        st.metric("平均Dice系数", f"{summary.get('mean_dice', 0):.4f}")
                        st.metric("平均IoU", f"{summary.get('mean_iou', 0):.4f}")
                        st.metric("平均Hausdorff距离", f"{summary.get('mean_hausdorff_distance', 0):.2f} mm")

                        st.subheader("各肺叶Dice系数")
                        per_class = summary.get('per_class_summary', {})

                        for lobe_name, stats in per_class.items():
                            st.write(f"**{lobe_name}**: {stats['mean_dice']:.4f} ± {stats['std_dice']:.4f}")

                except Exception as e:
                    st.error(f"评估失败: {str(e)}")


if __name__ == "__main__":
    main()
