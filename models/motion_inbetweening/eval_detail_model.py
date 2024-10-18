import argparse
import os

import torch

from motion_inbetween import benchmark
from motion_inbetween.config import load_config_by_name
from motion_inbetween.data import utils_torch as data_utils
from motion_inbetween.data.bvh import save_bvh
from motion_inbetween.data.utils_torch import matrix9D_to_euler_angles, reverse_root_pos_rot_offset
from motion_inbetween.model import ContextTransformer, DetailTransformer
from motion_inbetween.train import context_model as ctx_mdl
from motion_inbetween.train import detail_model as det_mdl
from motion_inbetween.train import rmi
from motion_inbetween.train import utils as train_utils

if __name__ == "__main__":
    # IMPORTANT: Input arguments - in our case that data should be passed from Blender by the plugin
    parser = argparse.ArgumentParser(description="Evaluate detail model. "
                                                 "No post-processing applied by default.")
    parser.add_argument("det_config", help="detail config name")
    parser.add_argument("ctx_config", help="context config name")
    parser.add_argument("animation", help="path to BVH file with animation")
    parser.add_argument("output_dir",
                        help="output directory where both reference animation and inferred animation should be placed")
    parser.add_argument("-o", "--offset",
                        help="animation clip frame offset from start, should be >= 175 (default=175)",
                        type=int, default=175)
    parser.add_argument("-t", "--trans", type=int, default=30,
                        help="transition length (default=30)")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="debug mode")
    parser.add_argument("-p", "--post_processing", action="store_true",
                        default=False, help="apply post-processing")

    args = parser.parse_args()

    # Model specific - loading config from a JSON file
    det_config = load_config_by_name(args.det_config)
    ctx_config = load_config_by_name(args.ctx_config)

    # IMPORTANT:  Device to store most of the data (Nvidia GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load animation data to memory (GPU/CPU) - originally used for whole test dataset evaluation
    dataset, data_loader = train_utils.init_bvh_dataset_single(
        det_config,
        args.animation,
        args.offset,
        device=device,
        shuffle=False
    )

    # IMPORTANT: Empty model initialization (no parameters loaded, no GPU/CPU memory used by model)
    detail_model = DetailTransformer(det_config["model"]).to(device)
    context_model = ContextTransformer(ctx_config["model"]).to(device)

    # IMPORTANT: Checkpoint loading (loading parameters, now it uses GPU/CPU memory)
    train_utils.load_checkpoint(det_config, detail_model)
    train_utils.load_checkpoint(ctx_config, context_model)

    # Model specific - model configuration
    indices = det_config["indices"]
    context_len = det_config["train"]["context_len"]
    target_idx = context_len + args.trans
    seq_slice = slice(context_len, target_idx)
    window_len = context_len + args.trans + 2
    dtype = dataset.dtype

    # Model specific - attention mask for context model
    atten_mask_ctx = ctx_mdl.get_attention_mask(
        window_len, context_len, target_idx, device)

    # Model specific - attention mask for detail model
    atten_mask = det_mdl.get_attention_mask(window_len, target_idx, device)

    # Model specific - mean and std loaded/calculated on training data
    mean_ctx, std_ctx = ctx_mdl.get_train_stats_torch(
        ctx_config, dtype, device)
    mean_state, std_state, _, _ = \
        det_mdl.get_train_stats_torch(det_config, dtype, device)

    # Model specific - mean and std calculated on benchmark data (here it is the same as training data)
    mean_rmi, std_rmi = rmi.get_rmi_benchmark_stats_torch(
        det_config, dtype, device)

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
    (positions, rotations, global_positions, global_rotations,
     foot_contact, parents, _) = data

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
    positions, rotations, root_position_offset, root_rotation_offset = data_utils.to_start_centered_data(
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
        args.post_processing
    )

    # Debug: output file prefix
    base_animation_name = os.path.basename(args.animation).split('.')[0]

    # Debug: reverse transformations of original animation data
    true_ref_pos, true_ref_rot = reverse_root_pos_rot_offset(
        positions,
        rotations,
        root_position_offset,
        root_rotation_offset
    )

    # Debug: Change rotation representation used by model to the one matching the output for original animation data
    true_ref_rot_euler = matrix9D_to_euler_angles(true_ref_rot)

    # Debug: Save original animation clip as another file
    save_bvh(args.animation, true_ref_pos, true_ref_rot_euler,
             output_file=os.path.join(args.output_dir, f"{base_animation_name}_ref.bvh"))

    # Reverse transformation on inferred data (reverse centering and rotation of the front of skeleton towards X axis)
    true_inferred_pos, true_inferred_rot = reverse_root_pos_rot_offset(
        pos_new,
        rot_new,
        root_position_offset,
        root_rotation_offset
    )

    # Change rotation representation used by model to the one matching the output
    # (in our case the output representation is the one used by Blender)
    true_inferred_rot_euler = matrix9D_to_euler_angles(true_inferred_rot)

    # IMPORTANT: Return output - here we save it in BVH format as a file
    save_bvh(args.animation, true_inferred_pos, true_inferred_rot_euler,
             output_file=os.path.join(args.output_dir, f"{base_animation_name}_inferred.bvh"))

    # IMPORTANT: Metric calculation - these are comparative
    gpos_batch_loss, gquat_batch_loss, _, _ = \
        benchmark.get_rmi_style_batch_loss(
            positions, rotations, pos_new, rot_new, parents,
            context_len, target_idx, mean_rmi, std_rmi)

    # IMPORTANT: Metric display
    print("{}, trans: {}, gpos: {:.4f}, gquat: {:.4f}{}".format(
        args.det_config, args.trans,
        gpos_batch_loss[0], gquat_batch_loss[0],
        " (w/ post-processing)" if args.post_processing else ""))
