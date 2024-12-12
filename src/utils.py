import json
import csv

import numpy as np
from typing import Literal
from scipy.spatial.transform import Rotation


def convert_array_3x3matrix_to_euler_zyx(mat: np.array) -> np.ndarray: 
  
    '''
    Convert an array of 3x3 rotation matrix to array of Euler angles ZYX.
    '''

    assert mat.shape[2:4] == (3, 3), "Shape of array must be (X, Y, 3, 3)!"


    def rotation_matrix3x3_to_quat(R):
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S

        return np.array([qw, qx, qy, qz])
    

    # Create array for quats (X, Y, 4)
    X, Y = mat.shape[0:2]
    quats = np.zeros((X, Y, 4))  

    for i in range(X):
        for j in range(Y):
            quats[i, j] = rotation_matrix3x3_to_quat(mat[i, j])

    # Reshape quats array and convert to Euler ZYX
    quats = quats.reshape(-1, 4)
    rotations = Rotation.from_quat(quats)
    euler_angles = rotations.as_euler('ZYX', degrees=True)
    
    # Reshape to (X, Y, 3)
    euler_angles = euler_angles.reshape((X, Y, 3))

    return euler_angles


def copy_object(obj, context):
   
    '''
    Copy an object and link it to the context collection.
    '''

    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    new_obj.animation_data_clear()
    context.collection.objects.link(new_obj)
           
    obj.select_set(False)
    new_obj.select_set(True)                                        
    new_obj.animation_data_create()
    
    return new_obj


def has_missing_keyframes_between(obj, keyframes_range: tuple[int, int]) -> bool:
    
    '''
    Check if object has missing keyframes inside the defined range.
    '''

    frame_a, frame_b = keyframes_range

    # Return false if object has no animation data
    if not obj.animation_data or not obj.animation_data.action:
        return False
    
    # Build set of all existing keyframes
    fcurves = obj.animation_data.action.fcurves
    keyframes = {int(k.co[0]) for fcurve in fcurves for k in fcurve.keyframe_points}

    # Check if any keyframe is missing inside the range
    for frame in range(frame_a + 1, frame_b):
        if frame not in keyframes:
            return True
        
    return False


def export_dict_to_file(data: dict, filename: str, export_type: Literal['TXT', 'JSON', 'CSV']) -> str: # Filename without extension
    
    '''
    Export dictionary to file in defined format.
    Returns full filename with extension.
    '''

    match export_type:

        case 'TXT':
            filename += '.txt'
            with open(filename, 'w') as file:
                for key, value in data.items():
                    file.write(f"{key}: {value}\n")

        case 'JSON':
            filename += '.json'
            with open(filename, 'w') as file:
                json.dump(data, file, indent=4)
        
        case 'CSV':
            filename += '.csv'
            data_list = [
                {'Name': key, 'Value': value} for key, value in data.items()
            ]
            with open(filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['Name', 'Value'])
                writer.writeheader()
                writer.writerows(data_list)

    return filename
