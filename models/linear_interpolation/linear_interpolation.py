from interface.general_interface import GeneralInterface

import numpy as np
from scipy.spatial.transform import Rotation

class ModelInterface(GeneralInterface):

    '''
    Simple linear interpolation (without AI).
    '''

    def check_frames_range(self, start_frame, end_frame, scene_start_frame, scene_end_frame) -> tuple[bool, str]:
        if start_frame <= scene_start_frame: return (False, "Must be at least 1 frame before selected range.") 
        if end_frame >= scene_end_frame: return (False, "Must be at least 1 frame after selected range.") 
        return (True, "")
    
    # end function check_frames_range

    def get_infer_anim_kwargs(self) -> list[tuple[type, str, str]]:
        return []

    # end get_infer_anim_kwargs

    def infer_anim(self, anim_data, start_frame, end_frame, **kwargs):  
                
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

        # Interpolation
        trans = end_frame - start_frame + 1
        positions = anim_data['positions']
        rotations = anim_data['rotations']

        start_positions = positions[..., start_frame, :, :]
        end_positions = positions[..., end_frame, :, :]  
        start_rotations = rotations[..., start_frame, :, :, :]
        end_rotations = rotations[..., end_frame, :, :, :]

        interpolated_positions = np.zeros((trans, positions.shape[-2], 3))  # (trans, joints, 3)
        interpolated_rotations = np.zeros((trans, rotations.shape[-3], 3, 3))  # (trans, joints, 3x3 rotation matrix)

        # Perform linear interpolation for positions
        for t in range(trans):
            alpha = t / (trans - 1)  # Interpolation coefficient
            interpolated_positions[t] = (1 - alpha) * start_positions + alpha * end_positions

        # Perform linear interpolation for rotations
        for t in range(trans):
            alpha = t / (trans - 1)  # Interpolation coefficient 
            interpolated_rotations[t] = (1 - alpha) * start_rotations + alpha * end_rotations

        # Convert to Euler ZYX
        interpolated_rotations = convert_array_3x3matrix_to_euler_zyx(interpolated_rotations)

        print("Linear interpolation done")
        
        return interpolated_positions, interpolated_rotations
            
    # end function infer_anim

# end class ModelInterface
