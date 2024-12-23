from typing import Optional
import bpy
import math


def calculate_metrics(
    start_frame: int,
    end_frame: int,
    generated_obj: bpy.types.Object,
    orginal_obj = Optional[bpy.types.Object],
    round_digits: int | None = 3
) -> dict[str, float]:
        
    '''
    Calculate metrics for the generated object.
    '''
    
    metrics = {}

    # Comparative metrics
    if orginal_obj:
        metrics['Position MSE'] = position_mse(orginal_obj, generated_obj, start_frame, end_frame)
        metrics['Rotation MSE'] = rotation_mse(orginal_obj, generated_obj, start_frame, end_frame)

    # Non-comparative metrics
    metrics['Position Smoothness Error'] = position_acc_smoothness_error(generated_obj, start_frame, end_frame)
    metrics['Rotation Smoothness Error'] = rotation_acc_smoothness_error(generated_obj, start_frame, end_frame)

    # Round generated metrics
    if round_digits is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], round_digits)

    return metrics



# --- Metric functions

def position_mse(obj1, obj2, start_frame: int, end_frame: int) -> float:
    
    '''
    Compute the mean squared error between the keypoints of two armatures.
    '''

    # Check if both objects have the same number of bones
    if len(obj1.pose.bones) != len(obj2.pose.bones):
        raise ValueError("Number of bones in armatures do not match")

    sum_error = 0.0
    scene = bpy.context.scene

    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)

        for bone in obj1.pose.bones:
            try:
                bone1 = obj1.pose.bones[bone.name]
                bone2 = obj2.pose.bones[bone.name]
            except KeyError:
                raise ValueError(f"Bone {bone.name} not found in both armatures")
            
            # Get global head positions
            bone1_head_global = obj1.matrix_world @ bone1.head
            bone2_head_global = obj2.matrix_world @ bone2.head
            
            sum_error += (bone1_head_global - bone2_head_global).length_squared
    
    mse = sum_error / ( len(obj1.pose.bones) * (end_frame - start_frame + 1) )
    return mse


def rotation_mse(obj1, obj2, start_frame: int, end_frame: int) -> float:
        
    '''
    Compute the mean squared error between the rotations of two armatures.
    '''

    # Check if both objects have the same number of bones
    if len(obj1.pose.bones) != len(obj2.pose.bones):
        raise ValueError("Number of bones in armatures do not match")

    sum_error = 0.0
    scene = bpy.context.scene

    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)

        for bone in obj1.pose.bones:
            try:
                bone1 = obj1.pose.bones[bone.name]
                bone2 = obj2.pose.bones[bone.name]
            except KeyError:
                raise ValueError(f"Bone {bone.name} not found in both armatures")
            
            # Get global rotations
            bone1_rot_global = obj1.matrix_world @ bone1.matrix
            bone2_rot_global = obj2.matrix_world @ bone2.matrix

            # Convert to quaternions
            bone1_rot_global_q = bone1_rot_global.to_quaternion()
            bone2_rot_global_q = bone2_rot_global.to_quaternion()
            
            # Compute the angle between the two quaternions
            angle = _compute_angle(bone1_rot_global_q, bone2_rot_global_q)

            sum_error += angle ** 2
    
    mse = sum_error / ( len(obj1.pose.bones) * (end_frame - start_frame + 1) )
    return mse


def position_acc_smoothness_error(obj, start_frame: int, end_frame: int) -> float:
    
    '''
    Compute the smoothness loss of the armature positions.
    It considers transitions between frames in the range [start_frame - 1, end_frame + 1].
    '''

    bone_smoothness: dict[str, float] = {}
    more_previous_bone_pos: dict = {}
    previous_bone_pos: dict = {}

    scene = bpy.context.scene

    # Initialize bone smoothness and get data from the (start_frame - 1) and start_frame frames
    scene.frame_set(start_frame - 1)
    for bone in obj.pose.bones:
        bone_smoothness[bone.name] = 0.0
        more_previous_bone_pos[bone.name] = obj.matrix_world @ bone.head
    scene.frame_set(start_frame)
    for bone in obj.pose.bones:
        previous_bone_pos[bone.name] = obj.matrix_world @ bone.head

    for frame in range(start_frame + 1, end_frame + 2):
        scene.frame_set(frame)

        for bone in obj.pose.bones:

            # Get global head positions
            bone_pos = obj.matrix_world @ bone.head

            # Compute the acceleration
            pos_diff = (more_previous_bone_pos[bone.name] - 2 * previous_bone_pos[bone.name] + bone_pos).length
            bone_smoothness[bone.name] += pos_diff
            
            # Update the previous bone positions
            more_previous_bone_pos[bone.name] = previous_bone_pos[bone.name]
            previous_bone_pos[bone.name] = bone_pos
    
    # Compute the smoothness loss
    all_bones_smoothness = sum(bone_smoothness.values())
    denominator = len(obj.pose.bones) * (end_frame - start_frame + 1)

    return all_bones_smoothness / denominator


def rotation_acc_smoothness_error(obj, start_frame: int, end_frame: int) -> float:
    
    '''
    Compute the smoothness loss of the armature rotations.
    It considers transitions between frames in the range [start_frame - 1, end_frame + 1].
    '''

    bone_smoothness: dict[str, float] = {}
    more_previous_bone_rot: dict = {}
    previous_bone_rot: dict = {}

    scene = bpy.context.scene

    # Initialize bone smoothness and get data from the (start_frame - 1) and start_frame frames
    scene.frame_set(start_frame - 1)
    for bone in obj.pose.bones:
        bone_smoothness[bone.name] = 0.0
        more_previous_bone_rot[bone.name] = (obj.matrix_world @ bone.matrix).to_quaternion()
    scene.frame_set(start_frame)
    for bone in obj.pose.bones:
        previous_bone_rot[bone.name] = (obj.matrix_world @ bone.matrix).to_quaternion()

    for frame in range(start_frame + 1, end_frame + 2):
        scene.frame_set(frame)

        for bone in obj.pose.bones:

            # Get global rotation in quaternion
            bone_rot_q = (obj.matrix_world @ bone.matrix).to_quaternion()

            # Compute the rotation difference
            angle1 = _compute_angle(more_previous_bone_rot[bone.name], previous_bone_rot[bone.name])
            angle2 = _compute_angle(previous_bone_rot[bone.name], bone_rot_q)

            bone_smoothness[bone.name] += abs( angle1 - angle2 )
            
            # Update the previous bone rotation
            more_previous_bone_rot[bone.name] = previous_bone_rot[bone.name]
            previous_bone_rot[bone.name] = bone_rot_q
    
    # Compute the smoothness loss
    all_bones_smoothness = sum(bone_smoothness.values())
    denominator = len(obj.pose.bones) * (end_frame - start_frame + 1)

    return all_bones_smoothness / denominator



# --- Helper functions

def _compute_angle(q1, q2, degrees=True) -> float:

    '''
    Compute the angle between two quaternions.
    angle = 2 * acos( |q1 . q2| )
    '''
    
    dot_product = abs(q1.dot(q2))
    dot_product = max(min(dot_product, 1.0), -1.0)  # to avoid NaN

    angle = 2 * math.acos(dot_product)
    if degrees:
        angle = math.degrees(angle)
    
    return angle
