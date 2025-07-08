import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from lib.config import cfg
from lib.utils.camera_utils import Camera
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.utils.general_utils import safe_state
import copy
import json

class ChunkTrainer:
    """
    分块训练管理器，将长序列分成多个重叠的块进行训练
    减少显存占用并支持更长的序列
    """
    
    def __init__(self, chunk_size: int = 5, overlap_size: int = 1, 
                 min_chunk_frames: int = 3, max_memory_usage: float = 0.8):
        """
        Args:
            chunk_size: 每个块的帧数
            overlap_size: 相邻块之间的重叠帧数  
            min_chunk_frames: 最小块大小
            max_memory_usage: 最大显存使用率
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_frames = min_chunk_frames
        self.max_memory_usage = max_memory_usage
        self.chunks_info = []
        self.trained_chunks = {}
        
    def create_chunks(self, total_frames: int, camera_list: List[Camera]) -> List[Dict]:
        """
        将总帧数分成多个训练块
        
        Args:
            total_frames: 总帧数
            camera_list: 所有相机列表
            
        Returns:
            chunks_info: 每个块的信息列表
        """
        chunks = []
        start_frame = 0
        chunk_id = 0
        
        # 按照帧ID对相机排序
        camera_list_sorted = sorted(camera_list, key=lambda x: x.meta['frame'])
        frame_to_cameras = {}
        for cam in camera_list_sorted:
            frame = cam.meta['frame']
            if frame not in frame_to_cameras:
                frame_to_cameras[frame] = []
            frame_to_cameras[frame].append(cam)
        
        available_frames = sorted(frame_to_cameras.keys())
        
        while start_frame < len(available_frames):
            end_frame = min(start_frame + self.chunk_size, len(available_frames))
            
            # 确保最后一个块有足够的帧数
            if end_frame < len(available_frames) and (len(available_frames) - end_frame) < self.min_chunk_frames:
                end_frame = len(available_frames)
            
            # 收集当前块的相机
            chunk_cameras = []
            chunk_frame_ids = available_frames[start_frame:end_frame]
            
            for frame_id in chunk_frame_ids:
                chunk_cameras.extend(frame_to_cameras[frame_id])
            
            chunk_info = {
                'id': chunk_id,
                'start_frame': available_frames[start_frame],
                'end_frame': available_frames[end_frame-1],
                'frame_ids': chunk_frame_ids,
                'cameras': chunk_cameras,
                'num_frames': len(chunk_frame_ids),
                'num_cameras': len(chunk_cameras),
                'overlap_start': start_frame > 0,
                'overlap_end': end_frame < len(available_frames)
            }
            
            chunks.append(chunk_info)
            chunk_id += 1
            
            # 移动到下一个块，考虑重叠
            if end_frame < len(available_frames):
                start_frame = end_frame - self.overlap_size
            else:
                break
        
        self.chunks_info = chunks
        print(f"Created {len(chunks)} chunks from {total_frames} frames")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: frames {chunk['start_frame']}-{chunk['end_frame']} "
                  f"({chunk['num_frames']} frames, {chunk['num_cameras']} cameras)")
        
        return chunks
    
    def get_chunk_dataset(self, chunk_info: Dict, original_dataset: Dataset) -> Dataset:
        """
        为特定块创建数据集
        
        Args:
            chunk_info: 块信息
            original_dataset: 原始数据集
            
        Returns:
            chunk_dataset: 块数据集
        """
        # 创建新的数据集实例
        chunk_dataset = Dataset.__new__(Dataset)
        chunk_dataset.cfg = original_dataset.cfg
        chunk_dataset.model_path = os.path.join(original_dataset.model_path, f"chunk_{chunk_info['id']}")
        chunk_dataset.source_path = original_dataset.source_path
        chunk_dataset.images = original_dataset.images
        
        # 复制场景信息但过滤相机
        chunk_dataset.scene_info = copy.deepcopy(original_dataset.scene_info)
        
        # 过滤训练和测试相机
        chunk_train_cameras = []
        chunk_test_cameras = []
        
        chunk_frame_set = set(chunk_info['frame_ids'])
        
        for cam_info in original_dataset.scene_info.train_cameras:
            if hasattr(cam_info, 'frame') and cam_info.frame in chunk_frame_set:
                chunk_train_cameras.append(cam_info)
        
        for cam_info in original_dataset.scene_info.test_cameras:
            if hasattr(cam_info, 'frame') and cam_info.frame in chunk_frame_set:
                chunk_test_cameras.append(cam_info)
        
        chunk_dataset.scene_info.train_cameras = chunk_train_cameras
        chunk_dataset.scene_info.test_cameras = chunk_test_cameras
        
        # 创建相机字典
        chunk_dataset.train_cameras = {}
        chunk_dataset.test_cameras = {}
        
        # 从块相机列表构建相机字典
        train_cameras_1 = [cam for cam in chunk_info['cameras'] if not cam.meta.get('is_val', False)]
        test_cameras_1 = [cam for cam in chunk_info['cameras'] if cam.meta.get('is_val', False)]
        
        chunk_dataset.train_cameras[1] = train_cameras_1
        chunk_dataset.test_cameras[1] = test_cameras_1
        
        # 创建输出目录
        os.makedirs(chunk_dataset.model_path, exist_ok=True)
        
        print(f"Chunk {chunk_info['id']} dataset: "
              f"{len(train_cameras_1)} train cameras, {len(test_cameras_1)} test cameras")
        
        return chunk_dataset
    
    def estimate_memory_usage(self, chunk_info: Dict) -> float:
        """
        估算块的显存使用量
        
        Args:
            chunk_info: 块信息
            
        Returns:
            estimated_memory: 估算的显存使用量(GB)
        """
        # 基于相机数量和帧数的简单估算
        base_memory = 1.0  # 基础显存(GB)
        per_camera_memory = 0.1  # 每个相机的显存(GB)
        per_frame_memory = 0.2  # 每帧的显存(GB)
        
        estimated = (base_memory + 
                    chunk_info['num_cameras'] * per_camera_memory + 
                    chunk_info['num_frames'] * per_frame_memory)
        
        return estimated
    
    def should_reduce_chunk_size(self, chunk_info: Dict) -> bool:
        """
        判断是否需要减小块大小
        
        Args:
            chunk_info: 块信息
            
        Returns:
            should_reduce: 是否需要减小
        """
        estimated_memory = self.estimate_memory_usage(chunk_info)
        
        # 获取当前显存使用情况
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            projected_usage = (current_memory + estimated_memory) / total_memory
            
            return projected_usage > self.max_memory_usage
        
        return False
    
    def save_chunk_info(self, output_dir: str):
        """
        保存块信息到文件
        
        Args:
            output_dir: 输出目录
        """
        chunk_info_path = os.path.join(output_dir, "chunks_info.json")
        
        # 准备可序列化的数据
        serializable_chunks = []
        for chunk in self.chunks_info:
            serializable_chunk = {
                'id': chunk['id'],
                'start_frame': int(chunk['start_frame']),
                'end_frame': int(chunk['end_frame']),
                'frame_ids': [int(fid) for fid in chunk['frame_ids']],
                'num_frames': chunk['num_frames'],
                'num_cameras': chunk['num_cameras'],
                'overlap_start': chunk['overlap_start'],
                'overlap_end': chunk['overlap_end']
            }
            serializable_chunks.append(serializable_chunk)
        
        with open(chunk_info_path, 'w') as f:
            json.dump({
                'chunk_size': self.chunk_size,
                'overlap_size': self.overlap_size,
                'total_chunks': len(self.chunks_info),
                'chunks': serializable_chunks
            }, f, indent=2)
        
        print(f"Saved chunk information to {chunk_info_path}")
    
    def load_chunk_info(self, info_path: str):
        """
        从文件加载块信息
        
        Args:
            info_path: 信息文件路径
        """
        with open(info_path, 'r') as f:
            data = json.load(f)
        
        self.chunk_size = data['chunk_size']
        self.overlap_size = data['overlap_size']
        
        # 恢复块信息（不包含相机对象）
        self.chunks_info = []
        for chunk_data in data['chunks']:
            chunk_info = {
                'id': chunk_data['id'],
                'start_frame': chunk_data['start_frame'],
                'end_frame': chunk_data['end_frame'],
                'frame_ids': chunk_data['frame_ids'],
                'num_frames': chunk_data['num_frames'],
                'num_cameras': chunk_data['num_cameras'],
                'overlap_start': chunk_data['overlap_start'],
                'overlap_end': chunk_data['overlap_end'],
                'cameras': []  # 需要重新填充
            }
            self.chunks_info.append(chunk_info)
        
        print(f"Loaded chunk information from {info_path}")