"""项目主入口点"""

import sys
import argparse


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='胸部CT肺叶自动分割系统'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['gui', 'cli', 'streamlit'],
        default='gui',
        help='运行模式'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='输入CT影像路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='输出目录'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='data/models/nnUNet/3d_fullres',
        help='模型路径'
    )

    args = parser.parse_args()

    if args.mode == 'gui':
        run_gui()
    elif args.mode == 'cli':
        run_cli(args)
    elif args.mode == 'streamlit':
        run_streamlit()


def run_gui():
    """运行PyQt5 GUI"""
    try:
        from PyQt5.QtWidgets import QApplication
        from src.gui.main_window import MainWindow

        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except ImportError as e:
        print(f"PyQt5未安装: {e}")
        print("请使用 pip install PyQt5 安装")
        print("或者使用 streamlit 模式: python -m src.main --mode streamlit")


def run_cli(args):
    """运行命令行模式"""
    if not args.input:
        print("错误: CLI模式需要指定 --input 参数")
        print("使用 --help 查看帮助")
        return

    from src.engine.inference.predictor import Predictor
    from src.engine.inference.postprocessor import Postprocessor
    from src.engine.data_manager.data_loader import DataLoader

    print(f"输入影像: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"模型路径: {args.model}")

    predictor = Predictor(model_path=args.model)
    postprocessor = Postprocessor()

    result = predictor.predict(args.input)

    postprocess_result = postprocessor.postprocess(result.segmentation)
    result.segmentation = postprocess_result['segmentation']

    import os
    os.makedirs(args.output, exist_ok=True)

    output_file = f"{args.output}/prediction.nii.gz"
    result.save(output_file)

    print(f"\n分割完成!")
    print(f"结果已保存: {output_file}")


def run_streamlit():
    """运行Streamlit应用"""
    try:
        import subprocess
        subprocess.run(['streamlit', 'run', 'src/gui/app.py'])
    except FileNotFoundError:
        print("Streamlit未安装")
        print("请使用 pip install streamlit 安装")


if __name__ == '__main__':
    main()
