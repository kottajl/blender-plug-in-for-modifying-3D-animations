from typing import Optional
import bpy
import math
# import mathutils


def calculate_metrics(
    start_frame: int,
    end_frame: int,
    generated_obj: bpy.types.Object,
    orginal_obj = Optional[bpy.types.Object]
) -> dict[str, float]:
        
    '''
    Calculate metrics between two armatures.
    '''

    metrics = {}

    # Comparative metrics
    if orginal_obj:
        metrics['position_mse'] = position_mse(orginal_obj, generated_obj, start_frame, end_frame)
        metrics['rotation_mse'] = rotation_mse(orginal_obj, generated_obj, start_frame, end_frame)

    # Non-comparative metrics
    metrics['position_smooth_loss_error'] = position_smoothness_error(generated_obj, start_frame, end_frame)
    metrics['rotation_smooth_loss_error'] = rotation_smoothness_error(generated_obj, start_frame, end_frame)

    return metrics
#calculate_metrics


def position_mse(
    obj1,
    obj2,
    start_frame: int,
    end_frame: int
) -> float:
    
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
#position_mse


def rotation_mse(
    obj1,
    obj2,
    start_frame: int,
    end_frame: int
) -> float:
        
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
            
            # Compute the rotation difference -> angle = 2 * acos( |q1 . q2| )
            dot_product = abs(bone1_rot_global_q.dot(bone2_rot_global_q))
            angle = 2 * math.acos(min(dot_product, 1.0))        # min() to avoid NaN
            
            angle_deg = math.degrees(angle)
            sum_error += angle_deg ** 2
    
    mse = sum_error / ( len(obj1.pose.bones) * (end_frame - start_frame + 1) )
    return mse
#rotation_mse


def position_smoothness_error(
    obj,
    start_frame: int,
    end_frame: int
) -> float:
    
    '''
    Compute the smoothness loss of the armature positions.
    It considers transitions between frames in the range [start_frame - 1, end_frame + 1].
    '''

    bone_smoothness: dict[str, float] = {}
    previous_bone_pos: dict = {}

    scene = bpy.context.scene

    # Initialize bone smoothness and get data from the (start_frame - 1) frame
    scene.frame_set(start_frame - 1)
    for bone in obj.pose.bones:
        bone_smoothness[bone.name] = 0.0
        previous_bone_pos[bone.name] = obj.matrix_world @ bone.head

    for frame in range(start_frame, end_frame + 2):
        scene.frame_set(frame)

        for bone in obj.pose.bones:

            # Get global head positions
            bone_pos = obj.matrix_world @ bone.head

            # Compute the difference between the current and previous frame
            pos_diff = (bone_pos - previous_bone_pos[bone.name]).length
            bone_smoothness[bone.name] += pos_diff
            
            # Update the previous bone position
            previous_bone_pos[bone.name] = bone_pos
    
    # Compute the smoothness loss
    all_bones_smoothness = sum(bone_smoothness.values())
    denominator = len(obj.pose.bones) * (end_frame - start_frame + 2)

    return all_bones_smoothness / denominator
#position_smoothness_error


def rotation_smoothness_error(
    obj,
    start_frame: int,
    end_frame: int
) -> float:
    
    '''
    Compute the smoothness loss of the armature rotations.
    It considers transitions between frames in the range [start_frame - 1, end_frame + 1].
    '''

    bone_smoothness: dict[str, float] = {}
    previous_bone_rot: dict = {}

    scene = bpy.context.scene

    # Initialize bone smoothness and get data from the (start_frame - 1) frame
    scene.frame_set(start_frame - 1)
    for bone in obj.pose.bones:
        bone_smoothness[bone.name] = 0.0
        previous_bone_rot[bone.name] = (obj.matrix_world @ bone.matrix).to_quaternion()

    for frame in range(start_frame, end_frame + 2):
        scene.frame_set(frame)

        for bone in obj.pose.bones:

            # Get global rotation in quaternion
            bone_rot_q = (obj.matrix_world @ bone.matrix).to_quaternion()

            # Compute the rotation difference -> angle = 2 * acos( |q1 . q2| )
            dot_product = abs(bone_rot_q.dot(previous_bone_rot[bone.name]))
            angle = 2 * math.acos(min(dot_product, 1.0))        # min() to avoid NaN

            bone_smoothness[bone.name] += angle
            
            # Update the previous bone rotation
            previous_bone_rot[bone.name] = bone_rot_q
    
    # Compute the smoothness loss
    all_bones_smoothness = sum(bone_smoothness.values())
    denominator = len(obj.pose.bones) * (end_frame - start_frame + 2)

    return math.degrees(all_bones_smoothness / denominator)
#rotation_smoothness_error
