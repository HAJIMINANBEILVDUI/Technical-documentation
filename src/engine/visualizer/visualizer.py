"""可视化器：提供2D/3D可视化功能"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class VisualizerError(Exception):
    """可视化异常类"""
    pass


LOBE_COLORS = {
    0: (0, 0, 0, 0),
    1: (1, 0, 0, 0.5),
    2: (0, 1, 0, 0.5),
    3: (0, 0, 1, 0.5),
    4: (1, 1, 0, 0.5),
    5: (1, 0, 1, 0.5)
}

LOBE_NAMES = {
    0: 'Background',
    1: 'Left Upper Lobe',
    2: 'Left Lower Lobe',
    3: 'Right Upper Lobe',
    4: 'Right Middle Lobe',
    5: 'Right Lower Lobe'
}

LOBE_HEX_COLORS = {
    0: '#000000',
    1: '#FF0000',
    2: '#00FF00',
    3: '#0000FF',
    4: '#FFFF00',
    5: '#FF00FF'
}


class Visualizer2D:
    """2D切片可视化器"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化2D可视化器

        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        self.current_slice_idx = None
        self.window_center = -400
        self.window_width = 1500

    def create_colormap(self, view: str = 'axial') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建肺叶颜色映射

        Args:
            view: 视图类型 ('axial', 'sagittal', 'coronal')

        Returns:
            颜色映射数组
        """
        labels = np.arange(6)
        colors = [LOBE_COLORS[i] for i in labels]

        return labels, np.array(colors)

    def apply_window_level(
        self,
        image: np.ndarray,
        center: float,
        width: float
    ) -> np.ndarray:
        """
        应用窗宽窗位

        Args:
            image: CT影像
            center: 窗位
            width: 窗宽

        Returns:
            窗口化后的影像
        """
        min_val = center - width / 2
        max_val = center + width / 2

        windowed = np.clip(image, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)

        return windowed

    def show_slice(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        slice_idx: Optional[int] = None,
        axis: int = 0,
        title: Optional[str] = None,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None
    ) -> Figure:
        """
        显示单个切片

        Args:
            image: CT影像 (D, H, W)
            segmentation: 分割结果
            slice_idx: 切片索引
            axis: 切片轴 (0: 横断面, 1: 矢状面, 2: 冠状面)
            title: 标题
            window_center: 窗位
            window_width: 窗宽

        Returns:
            matplotlib Figure对象
        """
        if window_center is not None:
            self.window_center = window_center
        if window_width is not None:
            self.window_width = window_width

        if slice_idx is None:
            slice_idx = image.shape[axis] // 2

        if axis == 0:
            slice_data = image[slice_idx, :, :]
            if segmentation is not None:
                seg_slice = segmentation[slice_idx, :, :]
        elif axis == 1:
            slice_data = image[:, slice_idx, :]
            if segmentation is not None:
                seg_slice = segmentation[:, slice_idx, :]
        else:
            slice_data = image[:, :, slice_idx]
            if segmentation is not None:
                seg_slice = segmentation[:, :, slice_idx]

        windowed = self.apply_window_level(
            slice_data,
            self.window_center,
            self.window_width
        )

        fig, axes = plt.subplots(1, 2 if segmentation is not None else 1, figsize=self.figsize)

        if segmentation is None:
            axes = [axes]

        axes[0].imshow(windowed, cmap='gray', origin='lower')
        axes[0].set_title(title or f'Slice {slice_idx}')
        axes[0].axis('off')

        if segmentation is not None:
            axes[1].imshow(windowed, cmap='gray', origin='lower')

            colored_seg = np.zeros((*seg_slice.shape, 4))
            for label_id, color in LOBE_COLORS.items():
                if label_id > 0:
                    mask = seg_slice == label_id
                    colored_seg[mask] = color

            axes[1].imshow(colored_seg, origin='lower', alpha=0.5)
            axes[1].set_title('Segmentation')
            axes[1].axis('off')

        plt.tight_layout()
        return fig

    def show_overlay(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        slice_idx: Optional[int] = None,
        axis: int = 0,
        alpha: float = 0.5,
        title: Optional[str] = None
    ) -> Figure:
        """
        显示叠加结果

        Args:
            image: CT影像
            segmentation: 分割结果
            slice_idx: 切片索引
            axis: 切片轴
            alpha: 叠加透明度
            title: 标题

        Returns:
            matplotlib Figure对象
        """
        if slice_idx is None:
            slice_idx = image.shape[axis] // 2

        windowed = self.apply_window_level(
            image,
            self.window_center,
            self.window_width
        )

        if axis == 0:
            windowed = windowed[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :]
        elif axis == 1:
            windowed = windowed[:, slice_idx, :]
            seg_slice = segmentation[:, slice_idx, :]
        else:
            windowed = windowed[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx]

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(windowed, cmap='gray', origin='lower')

        colored_seg = np.zeros((*seg_slice.shape, 4))
        for label_id, color in LOBE_COLORS.items():
            if label_id > 0:
                mask = seg_slice == label_id
                colored_seg[mask] = color

        ax.imshow(colored_seg, origin='lower', alpha=alpha)
        ax.set_title(title or f'Overlay - Slice {slice_idx}')
        ax.axis('off')

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=LOBE_HEX_COLORS[i], alpha=alpha)
            for i in range(1, 6)
        ]
        labels = [LOBE_NAMES[i] for i in range(1, 6)]
        ax.legend(handles, labels, loc='upper right', fontsize=8)

        plt.tight_layout()
        return fig

    def show_comparison(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        slice_idx: Optional[int] = None,
        axis: int = 0
    ) -> Figure:
        """
        显示预测与金标准的对比

        Args:
            image: CT影像
            prediction: 预测结果
            ground_truth: 金标准
            slice_idx: 切片索引
            axis: 切片轴

        Returns:
            matplotlib Figure对象
        """
        if slice_idx is None:
            slice_idx = image.shape[axis] // 2

        windowed = self.apply_window_level(
            image,
            self.window_center,
            self.window_width
        )

        if axis == 0:
            windowed = windowed[slice_idx, :, :]
            pred_slice = prediction[slice_idx, :, :]
            gt_slice = ground_truth[slice_idx, :, :]
        elif axis == 1:
            windowed = windowed[:, slice_idx, :]
            pred_slice = prediction[:, slice_idx, :]
            gt_slice = ground_truth[:, slice_idx, :]
        else:
            windowed = windowed[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]
            gt_slice = ground_truth[:, :, slice_idx]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(windowed, cmap='gray', origin='lower')
        axes[0].set_title('CT Image')
        axes[0].axis('off')

        colored_pred = np.zeros((*pred_slice.shape, 4))
        for label_id, color in LOBE_COLORS.items():
            if label_id > 0:
                mask = pred_slice == label_id
                colored_pred[mask] = color
        axes[1].imshow(colored_pred, origin='lower')
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        colored_gt = np.zeros((*gt_slice.shape, 4))
        for label_id, color in LOBE_COLORS.items():
            if label_id > 0:
                mask = gt_slice == label_id
                colored_gt[mask] = color
        axes[2].imshow(colored_gt, origin='lower')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

        diff = np.zeros((*pred_slice.shape, 3))
        diff[(pred_slice != gt_slice) & (gt_slice > 0)] = [1, 0, 0]
        diff[(pred_slice != gt_slice) & (pred_slice > 0)] = [0, 1, 0]

        axes[3].imshow(windowed, cmap='gray', origin='lower')
        axes[3].imshow(diff, origin='lower', alpha=0.5)
        axes[3].set_title('Difference\n(Red: GT only, Green: Pred only)')
        axes[3].axis('off')

        plt.tight_layout()
        return fig

    def save_slice(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray],
        output_path: str,
        slice_idx: Optional[int] = None,
        axis: int = 0,
        dpi: int = 150
    ):
        """
        保存切片为图片

        Args:
            image: CT影像
            segmentation: 分割结果
            output_path: 输出路径
            slice_idx: 切片索引
            axis: 切片轴
            dpi: 图片分辨率
        """
        if segmentation is not None:
            fig = self.show_slice(
                image, segmentation,
                slice_idx=slice_idx,
                axis=axis
            )
        else:
            fig = self.show_slice(
                image,
                slice_idx=slice_idx,
                axis=axis
            )

        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    def create_slice_navigator(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        axis: int = 0
    ) -> Figure:
        """
        创建可交互的切片导航器

        Args:
            image: CT影像
            segmentation: 分割结果
            axis: 切片轴

        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.subplots_adjust(bottom=0.25)

        slice_idx = image.shape[axis] // 2

        windowed = self.apply_window_level(
            image,
            self.window_center,
            self.window_width
        )

        if axis == 0:
            slice_data = windowed[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :] if segmentation is not None else None
        elif axis == 1:
            slice_data = windowed[:, slice_idx, :]
            seg_slice = segmentation[:, slice_idx, :] if segmentation is not None else None
        else:
            slice_data = windowed[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx] if segmentation is not None else None

        im = ax.imshow(slice_data, cmap='gray', origin='lower')

        if seg_slice is not None:
            colored_seg = np.zeros((*seg_slice.shape, 4))
            for label_id, color in LOBE_COLORS.items():
                if label_id > 0:
                    mask = seg_slice == label_id
                    colored_seg[mask] = color
            ax.imshow(colored_seg, origin='lower', alpha=0.5)

        ax.set_title(f'Slice {slice_idx}')
        ax.axis('off')

        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        slider = Slider(
            ax_slider,
            'Slice',
            0,
            image.shape[axis] - 1,
            valinit=slice_idx,
            valfmt='%d'
        )

        def update(val):
            idx = int(slider.val)

            if axis == 0:
                slice_data = windowed[idx, :, :]
                seg_slice = segmentation[idx, :, :] if segmentation is not None else None
            elif axis == 1:
                slice_data = windowed[:, idx, :]
                seg_slice = segmentation[:, idx, :] if segmentation is not None else None
            else:
                slice_data = windowed[:, :, idx]
                seg_slice = segmentation[:, :, idx] if segmentation is not None else None

            im.set_data(slice_data)
            ax.set_title(f'Slice {idx}')
            fig.canvas.draw_idle()

        slider.on_changed(update)

        return fig

    def create_multiplanar_view(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        slice_idxs: Optional[Dict[str, int]] = None
    ) -> Figure:
        """
        创建多平面视图（MPR）

        Args:
            image: CT影像
            segmentation: 分割结果
            slice_idxs: 三个平面的切片索引

        Returns:
            matplotlib Figure对象
        """
        if slice_idxs is None:
            slice_idxs = {
                'axial': image.shape[0] // 2,
                'sagittal': image.shape[1] // 2,
                'coronal': image.shape[2] // 2
            }

        windowed = self.apply_window_level(
            image,
            self.window_center,
            self.window_width
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(windowed[slice_idxs['axial'], :, :], cmap='gray', origin='lower')
        axes[0].set_title(f'Axial (Slice {slice_idxs["axial"]})')
        axes[0].axis('off')

        axes[1].imshow(windowed[:, slice_idxs['sagittal'], :], cmap='gray', origin='lower')
        axes[1].set_title(f'Sagittal (Slice {slice_idxs["sagittal"]})')
        axes[1].axis('off')

        axes[2].imshow(windowed[:, :, slice_idxs['coronal']], cmap='gray', origin='lower')
        axes[2].set_title(f'Coronal (Slice {slice_idxs["coronal"]})')
        axes[2].axis('off')

        if segmentation is not None:
            for ax, axis, idx_key in zip(
                axes,
                [0, 1, 2],
                ['axial', 'sagittal', 'coronal']
            ):
                seg_slice = np.take(segmentation, slice_idxs[idx_key], axis=axis)

                colored_seg = np.zeros((*seg_slice.shape, 4))
                for label_id, color in LOBE_COLORS.items():
                    if label_id > 0:
                        mask = seg_slice == label_id
                        colored_seg[mask] = color

                ax.imshow(colored_seg, origin='lower', alpha=0.5)

        plt.tight_layout()
        return fig


class Visualizer3D:
    """3D体积渲染可视化器"""

    def __init__(self):
        """初始化3D可视化器"""
        self.alpha = 0.3

    def render_volume(
        self,
        segmentation: np.ndarray,
        show_labels: Optional[List[int]] = None,
        title: str = "3D Lung Lobe Rendering"
    ) -> Figure:
        """
        3D体积渲染

        Args:
            segmentation: 分割结果
            show_labels: 要显示的标签列表
            title: 标题

        Returns:
            matplotlib Figure对象
        """
        if show_labels is None:
            show_labels = [1, 2, 3, 4, 5]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for label_id in show_labels:
            mask = (segmentation == label_id)

            if not np.any(mask):
                continue

            verts, faces = self._marching_cubes(mask)

            if len(verts) > 0:
                color = LOBE_HEX_COLORS[label_id]
                ax.plot_trisurf(
                    verts[:, 0],
                    verts[:, 1],
                    verts[:, 2],
                    triangles=faces,
                    color=color,
                    alpha=self.alpha,
                    label=LOBE_NAMES[label_id]
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend(loc='upper right')

        return fig

    def render_comparison(
        self,
        segmentation1: np.ndarray,
        segmentation2: np.ndarray,
        label1_name: str = "Prediction",
        label2_name: str = "Ground Truth"
    ) -> Figure:
        """
        渲染对比结果

        Args:
            segmentation1: 第一个分割结果
            segmentation2: 第二个分割结果
            label1_name: 第一个分割的名称
            label2_name: 第二个分割的名称

        Returns:
            matplotlib Figure对象
        """
        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(121, projection='3d')
        for label_id in [1, 2, 3, 4, 5]:
            mask = (segmentation1 == label_id)
            if np.any(mask):
                verts, faces = self._marching_cubes(mask)
                if len(verts) > 0:
                    ax1.plot_trisurf(
                        verts[:, 0], verts[:, 1], verts[:, 2],
                        triangles=faces,
                        color=LOBE_HEX_COLORS[label_id],
                        alpha=self.alpha
                    )
        ax1.set_title(label1_name)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2 = fig.add_subplot(122, projection='3d')
        for label_id in [1, 2, 3, 4, 5]:
            mask = (segmentation2 == label_id)
            if np.any(mask):
                verts, faces = self._marching_cubes(mask)
                if len(verts) > 0:
                    ax2.plot_trisurf(
                        verts[:, 0], verts[:, 1], verts[:, 2],
                        triangles=faces,
                        color=LOBE_HEX_COLORS[label_id],
                        alpha=self.alpha
                    )
        ax2.set_title(label2_name)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.tight_layout()
        return fig

    def _marching_cubes(
        self,
        binary_mask: np.ndarray,
        level: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 marching cubes 算法提取等值面

        Args:
            binary_mask: 二值掩膜
            level: 等值面阈值

        Returns:
            Tuple[顶点数组, 面数组]
        """
        try:
            from skimage import measure

            verts, faces, normals, values = measure.marching_cubes(
                binary_mask.astype(np.float32),
                level=level
            )

            return verts, faces

        except ImportError:
            print("警告: skimage未安装，无法进行3D渲染")
            return np.array([]), np.array([])
        except Exception as e:
            print(f"3D渲染错误: {e}")
            return np.array([]), np.array([])

    def save_3d_view(
        self,
        segmentation: np.ndarray,
        output_path: str,
        dpi: int = 150
    ):
        """
        保存3D视图

        Args:
            segmentation: 分割结果
            output_path: 输出路径
            dpi: 图片分辨率
        """
        fig = self.render_volume(segmentation)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
