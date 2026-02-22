 # ROS 2  YOLOv11 & FastSAM 3D Perception

[![ROS 2](https://img.shields.io/badge/ROS2-Humble%20%7C%20Iron-blue.svg)](https://docs.ros.org/en/humble/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU_Accelerated-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-yellow.svg)](https://github.com/ultralytics/ultralytics)

## 📖 简介 (Overview)

基于 ROS 2 的 3D 视觉感知节点，本节点将 **YOLOv11**（目标检测）与 **FastSAM**（实例分割）集成，实现对目标 3D 点云的实时提取。本节点将深度图掩码过滤、针孔相机 3D 投影（Pinhole Camera Model）以及 TF 空间坐标变换（Homogeneous Transformation）全部迁移至 PyTorch Tensor，在 GPU 上完成百万级像素的并行计算，大大降低了 CPU 负载并大幅提高了推理速度。

## ✨ 核心特性 (Key Features)

* **实时目标检测与分割**：支持 YOLOv11 目标检测，并无缝对接 FastSAM 实现精确的像素级实例分割。
* **全 GPU 3D 点云重建**：基于 `torch.meshgrid` 和矩阵运算，在 GPU 上以 $O(1)$ 时间复杂度（相对像素数量）完成 2D 像素到 3D 空间点的映射。
* **硬件级 TF 坐标变换加速**：将 ROS TF 树生成的齐次变换矩阵加载至 CUDA，实现点云坐标系的极速转换（默认转换至 `world` 坐标系）。
* **动态主题发布机制**：根据当前视野中检测到的目标数量和类别，动态创建和销毁对应的局部点云（PointCloud2）发布器，节省带宽。
* **极致的内存管理**：针对 Python 和 PyTorch 在高频 ROS 回调中易发内存泄漏的问题，进行了严格的 `del`、`gc.collect()` 以及 `torch.cuda.empty_cache()` 显存碎片管理。

## 🛠️ 系统要求 (Prerequisites)

* **OS**: Ubuntu 22.04
* **ROS 2**: Humble / Iron / Jazzy
* **Sensor**: RGB-D 深度相机 (默认仿真型号为 Intel RealSense L515)
* **CUDA**: GTX5060+CUDA12.8
