from interface.general_interface import GeneralInterface

import sys
from pathlib import Path

packages_path = Path(__file__).parent.resolve() / "packages"
sys.path.append(str(packages_path))

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motion_inbetween.config import load_config_by_name
from motion_inbetween.data import utils_np
from motion_inbetween.data.utils_torch import to_start_centered_data, reverse_root_pos_rot_offset, matrix9D_to_quat_torch, remove_quat_discontinuities
from motion_inbetween.model import ContextTransformer, DetailTransformer
from motion_inbetween.train import context_model as ctx_mdl, detail_model as det_mdl, utils as train_utils, rmi

sys.path.remove(str(packages_path))

class ModelInterface(GeneralInterface):

    '''
    Implementation of Motion Inbetweening model.
    '''

    def check_frames_range(self, start_frame, end_frame, scene_start_frame, scene_end_frame) -> tuple[bool, str]:
        if start_frame < scene_start_frame + 10:  return (False, "Must be at least 10 frames before selected range.") 
        if end_frame + 2 > scene_end_frame:  return (False, "Must be at least 2 frames after selected range.") 
        return (True, "")
    

    def get_additional_infer_params(self) -> list[tuple[type, str, str]]:
        return [
                (torch.device, "Device", "Select device to compute on"),
                (bool, "Post processing", "Apply post processing on inferred data")         
            ]
    

    class BlenderDataSetSingle(torch.utils.data.Dataset):
        
        '''
        Dataset for the Motion Inbetweening model.
        '''

        def __init__(self, anim, window, start_frame, device, dtype=torch.float32):
            super().__init__()
            self.window = window
            self.start_frame = start_frame
            self.device = device
            self.dtype = dtype

            self.positions = []
            self.rotations = []
            self.global_positions = []
            self.global_rotations = []
            self.foot_contact = []
            self.frames = []
            self.parents = []

            # global joint rotation, position
            gr, gp = utils_np.fk(anim["rotations"][start_frame:], anim["positions"][start_frame:], anim["parents"])

            # left, right foot contact
            cl, cr = utils_np.extract_feet_contacts(gp, [3, 4], [7, 8], vel_threshold=0.2)

            self.positions.append(self._to_tensor(anim["positions"][start_frame:]))
            self.rotations.append(self._to_tensor(anim["rotations"][start_frame:]))
            self.global_positions.append(self._to_tensor(gp))
            self.global_rotations.append(self._to_tensor(gr))
            self.foot_contact.append(self._to_tensor(np.concatenate([cl, cr], axis=-1)))
            self.frames.append(anim["positions"][start_frame:].shape[0])
            self.parents = anim["parents"]

        def _to_tensor(self, array):
            return torch.tensor(array, dtype=self.dtype, device=self.device)

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            start_idx = idx
            end_idx = start_idx + self.window

            positions = self.positions[0][start_idx: end_idx]
            rotations = self.rotations[0][start_idx: end_idx]
            global_positions = self.global_positions[0][start_idx: end_idx]
            global_rotations = self.global_rotations[0][start_idx: end_idx]
            foot_contact = self.foot_contact[0][start_idx: end_idx]

            return (
                positions,
                rotations,
                global_positions,
                global_rotations,
                foot_contact,
                self.parents,
                idx
            )


    def matrix9D_to_euler_angles(self, mat):
        quat_data = matrix9D_to_quat_torch(mat)
        quat_data = remove_quat_discontinuities(quat_data)
        rotations = Rotation.from_quat(quat_data.cpu().numpy().flatten().reshape((-1, 4)))
        return rotations.as_euler('ZYX', degrees=True).reshape(1, -1, mat.shape[2], 3)


    def infer_anim(self, anim_data, start_frame, end_frame, **kwargs):
        # Model arguments
        device = kwargs.get("Device", "cpu") 
        post_processing = kwargs.get("Post processing", False)
        offset = start_frame - 10
        trans = end_frame - start_frame + 1
        
        # Model specific - loading config from a JSON file
        det_config = load_config_by_name("lafan1_detail_model")
        ctx_config = load_config_by_name("lafan1_context_model")

        # IMPORTANT: Empty model initialization (no parameters loaded, no GPU/CPU memory used by model)
        detail_model = DetailTransformer(det_config["model"]).to(device)
        context_model = ContextTransformer(ctx_config["model"]).to(device)

        # IMPORTANT: Checkpoint loading (loading parameters, now it uses GPU/CPU memory)
        train_utils.load_checkpoint(det_config, detail_model)
        train_utils.load_checkpoint(ctx_config, context_model)

        # Model specific - model configuration
        indices = det_config["indices"]
        context_len = det_config["train"]["context_len"]
        target_idx = context_len + trans
        seq_slice = slice(context_len, target_idx)
        window_len = context_len + trans + 2
        
        dataset = self.BlenderDataSetSingle(anim_data, window=window_len, start_frame=offset, device=device)
        dtype = dataset.dtype

        # Model specific - attention mask for context model
        atten_mask_ctx = ctx_mdl.get_attention_mask(window_len, context_len, target_idx, device)

        # Model specific - attention mask for detail model
        atten_mask = det_mdl.get_attention_mask(window_len, target_idx, device)

        # Model specific - mean and std loaded/calculated on training data
        mean_ctx, std_ctx = ctx_mdl.get_train_stats_torch(ctx_config, dtype, device)
        mean_state, std_state, _, _ = det_mdl.get_train_stats_torch(det_config, dtype, device)

        # Model specific - mean and std calculated on benchmark data (here it is the same as training data)
        mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(det_config, dtype, device)

        # Script specific - actual animation data extracted from dataset
        # (in this version of the script it is just one clip from one file)
        data = dataset[0]

        # IMPORTANT: Actual input of the model
        # positions: local positions of each joint (x,y,z)
        # rotations: local rotations of each joint represented as rotation matrix (9D representation)
        # global_positions: global positions of each joint (x,y,z)
        # global_rotations: global rotations of each joint represented as rotation matrix (9D representation)
        # foot_contact: binary array with left and right foot contact (1 when foot is stationary)
        # parents: skeleton architecture, used for FK (forward kinematics) to get global positions for counting metrics
        (positions, rotations, global_positions, global_rotations, foot_contact, parents, _) = data

        # Script specific - get actual motion clip with desired length
        # (before these operations the clip consists of 65 frames)
        positions = positions[..., :window_len, :, :]
        rotations = rotations[..., :window_len, :, :, :]
        global_positions = global_positions[..., :window_len, :, :]
        global_rotations = global_rotations[..., :window_len, :, :, :]

        # Used only for metric calculation
        foot_contact = foot_contact[..., :window_len, :]

        # Model specific - formatting data
        positions = positions.unsqueeze(0)
        rotations = rotations.unsqueeze(0)
        foot_contact = foot_contact.unsqueeze(0)

        # Model specific - center and rotation front of the skeleton towards X axis
        positions, rotations, root_position_offset, root_rotation_offset = to_start_centered_data(
            positions,
            rotations,
            context_len,
            return_offset=True
        )

        # IMPORTANT: Perform two-stage inference (context transformer -> detail transformer)
        pos_new, rot_new, foot_contact_new = det_mdl.evaluate(
            detail_model,
            context_model,
            positions,
            rotations,
            foot_contact,
            seq_slice,
            indices,
            mean_ctx,
            std_ctx,
            mean_state,
            std_state,
            atten_mask,
            atten_mask_ctx,
            post_processing
        )

        # reverse data transformations on the inferred data
        true_inferred_pos, true_inferred_rot = reverse_root_pos_rot_offset(
            pos_new,
            rot_new,
            root_position_offset,
            root_rotation_offset
        )
        
        # change inferred results from matrices to euler    
        true_inferred_rot_euler = self.matrix9D_to_euler_angles(true_inferred_rot)
        
        return true_inferred_pos[0][10:-2], true_inferred_rot_euler[0][10:-2]
    

    def is_skeleton_supported(self, skeleton) -> bool:
        return sorted(skeleton) == sorted([('Hips', None), ('LeftUpLeg', 'Hips'), ('LeftLeg', 'LeftUpLeg'), ('LeftFoot', 'LeftLeg'), ('LeftToe', 'LeftFoot'), ('RightUpLeg', 'Hips'), 
            ('RightLeg', 'RightUpLeg'), ('RightFoot', 'RightLeg'), ('RightToe', 'RightFoot'), ('Spine', 'Hips'), ('Spine1', 'Spine'), ('Spine2', 'Spine1'), ('Neck', 'Spine2'), ('Head', 'Neck'), 
                ('LeftShoulder', 'Spine2'), ('LeftArm', 'LeftShoulder'), ('LeftForeArm', 'LeftArm'), ('LeftHand', 'LeftForeArm'), ('RightShoulder', 'Spine2'), ('RightArm', 'RightShoulder'), 
                    ('RightForeArm', 'RightArm'), ('RightHand', 'RightForeArm')])
    

# ModelInterface
