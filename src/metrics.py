import bpy
import math
# import mathutils


def calculate_metrics(
    obj1,
    obj2,
    start_frame: int,
    end_frame: int
) -> dict[str, float]:
        
    '''
    Calculate metrics between two armatures.
    '''

    metrics = {}
    metrics['position_mse'] = position_mse_error(obj1, obj2, start_frame, end_frame)
    metrics['rotation_mse'] = rotation_mse_error(obj1, obj2, start_frame, end_frame)
    return metrics
#calculate_metrics


def position_mse_error(
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
#position_mse_error


def rotation_mse_error(
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
#rotation_mse_error