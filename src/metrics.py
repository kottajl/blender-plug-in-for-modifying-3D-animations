import bpy
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
    metrics['mse'] = keypoint_mse_error(obj1, obj2, start_frame, end_frame)
    return metrics
#calculate_metrics


def keypoint_mse_error(
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
#keypoint_mse_error
