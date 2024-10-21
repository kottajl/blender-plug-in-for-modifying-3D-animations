import numpy as np
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

# end function convert_array_3x3matrix_to_euler_zyx


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

# end function copy_object
