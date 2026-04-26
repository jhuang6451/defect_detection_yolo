import argparse
from ultralytics import YOLO
import os

def export_model(weights_path, format='onnx'):
    """
    将指定的 YOLOv8 权重文件导出为指定格式。
    支持格式: onnx, torchscript, tensorrt, coreml, tflite, edgetpu, openvino等。
    """
    if not os.path.exists(weights_path):
        print(f"❌ 错误: 找不到权重文件 {weights_path}")
        return

    print(f"🚀 正在加载模型: {weights_path}")
    model = YOLO(weights_path)

    print(f"📦 正在导出为 {format} 格式...")
    # export 方法会自动处理转换逻辑
    # dynamic=True 对 ONNX 很有用，允许推理时改变输入分辨率
    # simplify=True 会优化 ONNX 图结构
    path = model.export(format=format, dynamic=True, simplify=True)
    
    print(f"✅ 导出成功！文件保存在: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 模型导出脚本")
    parser.add_argument("--weights", type=str, required=True, help="训练好的 best.pt 路径")
    parser.add_argument("--format", type=str, default="onnx", 
                        help="导出格式 (onnx, engine, torchscript, tflite等, 默认 onnx)")
    
    args = parser.parse_args()
    
    export_model(args.weights, args.format)
