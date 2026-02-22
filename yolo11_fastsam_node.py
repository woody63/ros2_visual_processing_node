"""ROS 2 节点：利用 YOLOv11 与 FastSAM 进行基于 GPU 的 3D 视觉感知。

该模块集成 Ultralytics YOLO 和 FastSAM 模型，处理来自 RGB-D 相机
的图像与深度流。所有掩码处理、3D 点云反投影以及 TF 齐次坐标变换
均在 PyTorch 张量层面（GPU）完成，以实现极致的性能优化。
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String, Header
from cv_bridge import CvBridge
from rclpy.duration import Duration
from ultralytics import YOLO, FastSAM

import numpy as np
import open3d as o3d
import json
import cv2
import threading
import gc
import torch
import torch.nn.functional as F
from tf2_ros import Buffer, TransformListener
import tf_transformations as tft


def _free_ultra_result(res):
    """安全释放 Ultralytics 结果中的大对象。

    强制清除推断结果中可能导致显存/内存泄漏的缓存属性。

    Args:
        res (ultralytics.engine.results.Results): YOLO 或 FastSAM 的预测结果对象。
    """
    try:
        if hasattr(res, "orig_img"): res.orig_img = None
        if hasattr(res, "masks"): res.masks = None
        if hasattr(res, "probs"): res.probs = None
        if hasattr(res, "speed"): res.speed = None
        if hasattr(res, "boxes"): res.boxes = None
    except Exception:
        pass


class Yolov11Node(Node):
    """处理 YOLO 目标检测与 FastSAM 分割的 ROS 2 节点。

    该节点订阅相机的彩色和深度图像流，使用 GPU 加速提取特定目标的
    3D 点云，并发布处理后的掩码图像、可视化图像及独立的点云消息。

    Attributes:
        device (torch.device): 用于模型推理与张量运算的硬件设备（默认 cuda:0）。
        depth_threshold (float): 有效深度阈值上限，单位为米。
        threshold (float): YOLO 目标检测的置信度阈值。
        enable_yolo (bool): 控制视觉处理流水线是否激活的标志位。
        tf_update_rate (float): 拉取 TF 坐标树的频率。
    """

    def __init__(self):
        """初始化 Yolov11Node，加载参数、模型以及配置 ROS 发布/订阅器。"""
        super().__init__("yolov11_node")
        rclpy.logging.set_logger_level('yolov11_node', rclpy.logging.LoggingSeverity.INFO)

        # 参数声明与获取
        self.declare_parameter("detection_model", "yolo11s.pt")
        self.declare_parameter("segmentation_model", "FastSAM-x.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("depth_threshold", 2.5)
        self.declare_parameter("threshold", 0.6)
        self.declare_parameter("enable_yolo", True)
        self.declare_parameter("tf_update_rate", 100.0)

        detection_model = self.get_parameter("detection_model").get_parameter_value().string_value
        segmentation_model = self.get_parameter("segmentation_model").get_parameter_value().string_value
        self.device = torch.device(self.get_parameter("device").get_parameter_value().string_value)
        self.depth_threshold = self.get_parameter("depth_threshold").get_parameter_value().double_value
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.enable_yolo = self.get_parameter("enable_yolo").get_parameter_value().bool_value
        self.tf_update_rate = self.get_parameter("tf_update_rate").get_parameter_value().double_value

        # TF2 监听器配置
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_lock = threading.Lock()
        self.cached_transform_tensor = None  # 存储于 GPU 的 [4, 4] 变换矩阵
        self.tf_update_counter = 0

        # 视觉模型初始化
        self.cv_bridge = CvBridge()
        self.get_logger().info(f"Loading detection model: {detection_model}")
        self.yolo_detector = YOLO(detection_model)
        try:
            self.yolo_detector.fuse()
        except Exception:
            pass

        self.get_logger().info(f"Loading segmentation model: {segmentation_model}")
        self.fastsam = FastSAM(segmentation_model)

        # 消息与状态缓存
        self.image_lock = threading.Lock()
        self.color_image_msg = None
        self.depth_image_msg = None
        self.camera_intrinsics = None
        self.pred_image_msg = Image()

        self.fx = self.fy = self.cx = self.cy = None
        
        # GPU 网格缓存
        self._gpu_u_grid = None
        self._gpu_v_grid = None
        self._image_shape = None

        # 回调组与发布器/订阅器初始化
        self.image_callback_group = ReentrantCallbackGroup()
        self.tf_callback_group = MutuallyExclusiveCallbackGroup()
        self.processing_callback_group = MutuallyExclusiveCallbackGroup()

        self._item_dict_pub = self.create_publisher(String, "/yolo/prediction/item_dict", 10)
        self._pred_pub = self.create_publisher(Image, "/yolo/prediction/image", 10)
        self._camera_info_pub = self.create_publisher(CameraInfo, "/yolo/prediction/camera_info", 10)
        self._mask_image_pub = self.create_publisher(Image, "/yolo/prediction/mask_image", 10)
        self._mask_info_pub = self.create_publisher(String, "/yolo/prediction/mask_info", 10)
        
        self._mask_pointclouds_pub = {}
        self.last_frame_topics = set()

        image_qos = qos_profile_sensor_data
        self.create_subscription(Image, "/l515/image_raw", self.color_image_callback, image_qos, callback_group=self.image_callback_group)
        self.create_subscription(Image, "/l515/depth/image_raw", self.depth_image_callback, image_qos, callback_group=self.image_callback_group)
        self.create_subscription(CameraInfo, "/l515/camera_info", self.camera_info_callback, image_qos, callback_group=self.image_callback_group)

        self._tf_timer = self.create_timer(1.0 / self.tf_update_rate, self.update_tf_cache, callback_group=self.tf_callback_group)
        self._vision_timer = self.create_timer(0.01, self.vision_processing_loop, callback_group=self.processing_callback_group)

        self.get_logger().info("YOLOv11 + FastSAM Tensor-Optimized Node initialized.")

    def color_image_callback(self, msg):
        """彩色图像订阅回调。

        Args:
            msg (sensor_msgs.msg.Image): ROS 2 图像消息。
        """
        with self.image_lock:
            self.color_image_msg = msg

    def depth_image_callback(self, msg):
        """深度图像订阅回调。

        Args:
            msg (sensor_msgs.msg.Image): ROS 2 深度图像消息。
        """
        with self.image_lock:
            self.depth_image_msg = msg

    def camera_info_callback(self, msg):
        """相机内参订阅回调。

        提取并缓存焦距 (fx, fy) 与主点 (cx, cy)。

        Args:
            msg (sensor_msgs.msg.CameraInfo): ROS 2 相机信息消息。
        """
        if self.camera_intrinsics is None:
            self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
            self.camera_intrinsics.set_intrinsics(msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            self._camera_info_pub.publish(msg)

    def update_tf_cache(self):
        """定期拉取 TF 树并将变换矩阵缓存至 GPU。

        获取从 'l515_depth_optical_frame' 到 'world' 的齐次变换矩阵，
        并转换为 torch.Tensor 保存在指定的 device 上。
        """
        try:
            t = self.tf_buffer.lookup_transform(
                "world", "l515_depth_optical_frame", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01)
            )
            trans = t.transform.translation
            rot = t.transform.rotation
            T_np = tft.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
            T_np[:3, 3] = [trans.x, trans.y, trans.z]
            
            # 将变换矩阵加载到 GPU 上
            T_tensor = torch.tensor(T_np, dtype=torch.float32, device=self.device)
            
            with self.tf_lock:
                self.cached_transform_tensor = T_tensor
                
            self.tf_update_counter += 1
            if self.tf_update_counter % 20 == 0:
                gc.collect()
        except Exception as e:
            if self.cached_transform_tensor is None:
                self.get_logger().warn(f"Waiting TF 'world'->'l515_link': {str(e)[:50]}", throttle_duration_sec=3.0)

    def _init_gpu_grid(self, h, w):
        """初始化与图像等大的像素坐标网格 (U, V) 到 GPU。

        为避免逐帧重复计算，根据输入尺寸缓存网格张量。

        Args:
            h (int): 图像高度。
            w (int): 图像宽度。
        """
        if self._image_shape != (h, w) or self._gpu_u_grid is None:
            v_grid, u_grid = torch.meshgrid(
                torch.arange(h, device=self.device, dtype=torch.float32),
                torch.arange(w, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            self._gpu_u_grid = u_grid
            self._gpu_v_grid = v_grid
            self._image_shape = (h, w)

    def extract_pointcloud_gpu(self, mask_tensor, depth_tensor, color_tensor, T_world_optical_gpu):
        """全量利用 PyTorch GPU 进行点云提取、过滤和坐标变换。

        Args:
            mask_tensor (torch.Tensor): 目标物体的二维布尔掩码，形状 [H, W]。
            depth_tensor (torch.Tensor): 深度图，单位为米，形状 [H, W]。
            color_tensor (torch.Tensor): BGR 彩色图像张量，形状 [H, W, 3]。
            T_world_optical_gpu (torch.Tensor): 4x4 齐次变换矩阵，位于 GPU。

        Returns:
            tuple: 包含两个元素的元组:
                - points_np (numpy.ndarray): 变换到世界坐标系的 3D 坐标数组，形状 (N, 3)。
                - colors_np (numpy.ndarray): BGR 颜色数组，形状 (N, 3)。
        """
        h, w = depth_tensor.shape
        self._init_gpu_grid(h, w)

        # 掩码条件过滤：深度合法且在分割掩码内
        valid_mask = mask_tensor & (depth_tensor > 0.0) & (depth_tensor <= self.depth_threshold)
        
        # 提取有效区域的深度、UV和颜色
        z = depth_tensor[valid_mask]
        if z.numel() == 0:
            return np.array([]), np.array([])
            
        u = self._gpu_u_grid[valid_mask]
        v = self._gpu_v_grid[valid_mask]
        colors = color_tensor[valid_mask]  # [N, 3] (BGR)

        # 针孔相机模型：在 GPU 上计算 3D 相机坐标
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # 组装为齐次坐标 [N, 4]
        points_camera_h = torch.stack((x, y, z, torch.ones_like(z)), dim=1)

        # 坐标系变换 (点云矩阵乘法) [N, 4] @ [4, 4] -> [N, 4]
        points_world = points_camera_h @ T_world_optical_gpu.T

        # 仅在最后一步将结果拉回 CPU
        return points_world[:, :3].cpu().numpy(), colors.cpu().numpy()

    def build_pointcloud2_msg(self, points_np, colors_np_bgr, frame_id="world"):
        """将 NumPy 数组封装为 ROS PointCloud2 消息格式。

        利用 NumPy 向量化与位运算（Bitwise operations）加速 RGB 浮点数打包。

        Args:
            points_np (numpy.ndarray): Nx3 形状的空间坐标数组。
            colors_np_bgr (numpy.ndarray): Nx3 形状的 BGR 颜色数组。
            frame_id (str, optional): 目标坐标系名称。默认为 "world"。

        Returns:
            sensor_msgs.msg.PointCloud2: 序列化后的 ROS 点云消息。
        """
        if len(points_np) == 0:
            return self.create_empty_pointcloud2(frame_id)

        # 原地提取 BGR 并组装成 RGB Float
        b = colors_np_bgr[:, 0].astype(np.uint32)
        g = colors_np_bgr[:, 1].astype(np.uint32)
        r = colors_np_bgr[:, 2].astype(np.uint32)
        
        rgb_packed = (255 << 24) | (r << 16) | (g << 8) | b
        rgb_float = rgb_packed.view(np.float32)

        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)]
        cloud_arr = np.empty(len(points_np), dtype=dtype)
        cloud_arr['x'] = points_np[:, 0]
        cloud_arr['y'] = points_np[:, 1]
        cloud_arr['z'] = points_np[:, 2]
        cloud_arr['rgb'] = rgb_float

        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = frame_id
        cloud_msg.height = 1
        cloud_msg.width = len(points_np)
        
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 16 
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True
        cloud_msg.data = cloud_arr.tobytes()
        
        return cloud_msg

    def create_empty_pointcloud2(self, frame_id="world"):
        """创建一个空的 PointCloud2 消息对象。

        用于在未检测到目标时清空 RViz 等可视化工具中的点云显示。

        Args:
            frame_id (str, optional): 目标坐标系名称。默认为 "world"。

        Returns:
            sensor_msgs.msg.PointCloud2: 空点云消息。
        """
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 16
        msg.is_dense = True
        return msg

    def vision_processing_loop(self):
        """主视觉处理流水线。

        按固定频率调用，执行如下步骤：
        1. 获取彩色与深度帧；
        2. YOLOv11 目标检测；
        3. FastSAM 实例分割；
        4. 掩码交并比匹配与 GPU 点云投影；
        5. CPU/OpenCV 图像渲染及 ROS 话题发布。
        """
        if not self.enable_yolo or self.camera_intrinsics is None:
            return

        with self.image_lock:
            if self.color_image_msg is None or self.depth_image_msg is None:
                return
            color_msg, depth_msg = self.color_image_msg, self.depth_image_msg

        with self.tf_lock:
            if self.cached_transform_tensor is None:
                return
            T_world_optical_gpu = self.cached_transform_tensor

        current_frame_topics = set()
        detection_found = False

        try:
            # 1. 准备图像张量
            cv_color = self.cv_bridge.imgmsg_to_cv2(color_msg, "bgr8")
            cv_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            H, W = cv_color.shape[:2]

            # 载入数据至 GPU (深度图转换为米单位)
            color_tensor = torch.from_numpy(cv_color).to(self.device)
            if cv_depth.dtype == np.uint16:
                depth_tensor = torch.from_numpy(cv_depth.astype(np.int32)).to(self.device).float() * 0.001
            else:
                depth_tensor = torch.from_numpy(cv_depth).to(self.device).float()

            # 2. YOLO 目标检测 (禁用梯度)
            with torch.no_grad():
                results = self.yolo_detector(cv_color, show=False, verbose=False, conf=self.threshold, device=self.device)
                detection = results[0]

            if detection is None or detection.boxes is None or len(detection.boxes) == 0:
                self.clear_all_pointclouds()
                self.cleanup_inactive_publishers(set())
                return

            detection_found = True
            boxes_gpu = detection.boxes.xyxy    # 保持在 GPU
            classes_cpu = detection.boxes.cls.detach().cpu().numpy()

            # 3. FastSAM 实例分割
            with torch.no_grad():
                fastsam_results = self.fastsam(source=cv_color, device=self.device, retina_masks=True, imgsz=640, conf=0.4, iou=0.9)

            fastsam_masks_gpu = None
            if len(fastsam_results) > 0 and fastsam_results[0].masks is not None:
                raw_masks = fastsam_results[0].masks.data
                fastsam_masks_gpu = F.interpolate(raw_masks.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1) > 0.5
                _free_ultra_result(fastsam_results[0])
            del fastsam_results

            # 4. 掩码匹配与点云生成
            mask_dict = {}
            matched_masks_cpu = []

            for idx in range(len(boxes_gpu)):
                box = boxes_gpu[idx]
                best_mask_gpu = None
                
                # 边界框与 Mask 在 GPU 上基于 IoU 的基础相交检查
                if fastsam_masks_gpu is not None and fastsam_masks_gpu.shape[0] > 0:
                    x1, y1, x2, y2 = box.int()
                    x1, x2 = torch.clamp(x1, 0, W), torch.clamp(x2, 0, W)
                    y1, y2 = torch.clamp(y1, 0, H), torch.clamp(y2, 0, H)
                    
                    if x2 > x1 and y2 > y1:
                        box_area_mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)
                        box_area_mask[y1:y2, x1:x2] = True
                        
                        intersection = (fastsam_masks_gpu & box_area_mask).sum(dim=(1, 2)).float()
                        union = (fastsam_masks_gpu | box_area_mask).sum(dim=(1, 2)).float()
                        iou = intersection / (union + 1e-6)
                        
                        best_idx = torch.argmax(iou)
                        if iou[best_idx] > 0.1:
                            best_mask_gpu = fastsam_masks_gpu[best_idx]
                
                cls_id = classes_cpu[idx]
                cname = self.yolo_detector.names[int(cls_id)]
                
                if best_mask_gpu is not None:
                    matched_masks_cpu.append((best_mask_gpu.cpu().numpy().astype(np.uint8) * 255))
                    mask_dict[f"item_{idx}"] = {"class": cname, "has_mask": True}
                    
                    # 批量计算点云
                    points_np, colors_np = self.extract_pointcloud_gpu(best_mask_gpu, depth_tensor, color_tensor, T_world_optical_gpu)
                    
                    if len(points_np) > 0:
                        topic_name = f"/yolo/prediction/pointcloud/item_{idx}_{cname.replace(' ', '_')}"
                        current_frame_topics.add(topic_name)
                        
                        if topic_name not in self._mask_pointclouds_pub:
                            self._mask_pointclouds_pub[topic_name] = self.create_publisher(PointCloud2, topic_name, 10)
                        
                        cloud_msg = self.build_pointcloud2_msg(points_np, colors_np, frame_id="world")
                        self._mask_pointclouds_pub[topic_name].publish(cloud_msg)
                else:
                    matched_masks_cpu.append(None)

            # 5. 发布二值合成掩码与数据字典
            if len(matched_masks_cpu) > 0:
                combined_mask = np.zeros((H, W), dtype=np.uint8)
                for m in matched_masks_cpu:
                    if m is not None: combined_mask = cv2.bitwise_or(combined_mask, m)
                    
                mask_msg = self.cv_bridge.cv2_to_imgmsg(combined_mask, "mono8")
                mask_msg.header.stamp = self.get_clock().now().to_msg()
                mask_msg.header.frame_id = "l515_link"
                self._mask_image_pub.publish(mask_msg)

                mask_info_msg = String(data=json.dumps(mask_dict))
                self._mask_info_pub.publish(mask_info_msg)
            
            # 清理失效的发布器句柄
            self.cleanup_inactive_publishers(current_frame_topics)
            self.last_frame_topics = current_frame_topics

            # 6. 可视化渲染与发布
            annotated_frame = detection.plot()

            if len(matched_masks_cpu) > 0:
                overlay = np.zeros_like(annotated_frame)
                for m in matched_masks_cpu:
                    if m is not None:
                        color = np.random.randint(0, 255, 3).tolist()
                        overlay[m > 127] = color
                
                cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0.0, annotated_frame)

            self.pred_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.pred_image_msg.header.stamp = self.get_clock().now().to_msg()
            self.pred_image_msg.header.frame_id = "l515_link"
            self._pred_pub.publish(self.pred_image_msg)

            del annotated_frame
            if 'overlay' in locals():
                del overlay

        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")
            import traceback; self.get_logger().error(traceback.format_exc())
            
        finally:
            # 显式清理 GPU 引用以防止显存溢出
            if 'color_tensor' in locals(): del color_tensor
            if 'depth_tensor' in locals(): del depth_tensor
            if 'fastsam_masks_gpu' in locals(): del fastsam_masks_gpu
            if 'detection' in locals() and detection is not None: _free_ultra_result(detection)
            
            torch.cuda.empty_cache()

            if not detection_found:
                self.clear_all_pointclouds()
                self.last_frame_topics.clear()

    def clear_all_pointclouds(self):
        """向所有活跃的话题发布空点云消息，清空 RViz 界面残留。"""
        empty_cloud = self.create_empty_pointcloud2()
        for topic in self.last_frame_topics:
            if topic in self._mask_pointclouds_pub:
                self._mask_pointclouds_pub[topic].publish(empty_cloud)
        self._mask_info_pub.publish(String(data="{}"))

    def cleanup_inactive_publishers(self, current_topics):
        """销毁当前帧不再出现的目标点云发布器。

        Args:
            current_topics (set): 当前帧活跃的目标话题名集合。
        """
        inactive = set(self._mask_pointclouds_pub.keys()) - current_topics
        for topic in inactive:
            self.destroy_publisher(self._mask_pointclouds_pub[topic])
            del self._mask_pointclouds_pub[topic]
            
    def shutdown_callback(self):
        """节点关闭回调函数，释放 GPU 显存及清空最后状态。"""
        self.clear_all_pointclouds()
        torch.cuda.empty_cache()


def main(args=None):
    """主执行函数，初始化 rclpy 并启动多线程执行器下的视觉节点。"""
    rclpy.init(args=args)
    node = Yolov11Node()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_callback()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()