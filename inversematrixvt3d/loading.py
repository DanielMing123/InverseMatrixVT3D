# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Union
import cv2
import mmcv
import numpy as np
import torch
import os
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles, Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS
import pykitti.utils as utils

Number = Union[int, float]


@TRANSFORMS.register_module()
class SemanticKITTI_Image_Load(LoadMultiViewImageFromFiles):
    def transform(self, result: dict) -> Optional[dict]:
        calib_filepath = result['calib_path']
        filedata = utils.read_calib_file(calib_filepath)
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T_cam0_velo = np.reshape(filedata['Tr'], (3, 4))
        T_cam0_velo = np.vstack([T_cam0_velo, [0, 0, 0, 1]])
        T_cam2_velo = T2.dot(T_cam0_velo)
        K_P2 = np.eye(4)
        K_P2[:3,:3] = P_rect_20[:3,:3]
        lidar2img = K_P2.dot(T_cam2_velo)
        result['lidar2img'] = np.stack([lidar2img], axis=0)
        
        img_byte = get(result['img_path'], backend_args=self.backend_args) 
        img = mmcv.imfrombytes(img_byte, flag=self.color_type)
        result['img'] = [img]
        
        return result

@TRANSFORMS.register_module()
class LoadSemanticKITTI_Occupancy(BaseTransform):
    def get_remap_lut(self, label_map):
        maxkey = max(label_map.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(label_map.keys())] = list(label_map.values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut
        
    
    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed
    
    def transform(self, result: dict) -> dict:
        gt = np.fromfile(result['voxel_gt_path'],dtype=np.uint16).astype(np.float32)
        invalid = np.fromfile(result['voxel_invalid_path'], dtype=np.uint8)
        invalid = self.unpack(invalid)
        remap_lut = self.get_remap_lut(result['label_mapping'])
        gt = remap_lut[gt.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
        # gt[gt!=0] = 1
        gt[np.isclose(invalid, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
        gt = gt.reshape([256, 256, 32])
        gt = torch.from_numpy(gt)
        idx = torch.where(gt > 0)
        label = gt[idx[0],idx[1],idx[2]]
        semantickitti_occ = torch.stack([idx[0],idx[1],idx[2],label],dim=1).long()
        
        result['occ_semantickitti'] = semantickitti_occ
        return result
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    
    

@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename, cam2img, lidar2cam, lidar2img = [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam'],dtype=np.float64)
            cam2img_array = np.eye(4).astype(np.float64)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img'],dtype=np.float64)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        # gbr follow tpvformer
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # imgs = [
        #     cv2.resize(mmcv.imfrombytes(img_byte, flag=self.color_type,backend='cv2'),(0,0),fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
        #     for img_byte in img_bytes
        # ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 0.2 # 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results


@TRANSFORMS.register_module()
class LoadOccupancy(BaseTransform):
     
    def transform(self, results: dict) -> dict:
        occ_file_name = results['lidar_points']['lidar_path'].split('/')[-1] + '.npy'
        occ_200_folder = results['lidar_points']['lidar_path'].split('samples')[0] + 'occ_samples'
        occ_200_path = os.path.join(occ_200_folder, occ_file_name)
        occ_200 = np.load(occ_200_path)
        occ_200[:,3][occ_200[:,3]==0]=255
        occ_200 = torch.from_numpy(occ_200)
        results['occ_200'] = occ_200
        
        return results
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

