import numpy as np
from math import radians
from mathutils import Euler, Matrix, Vector

import bpy
from bpy.utils import escape_identifier
from bpy.types import Object, Armature


class DecoratedBone:

    '''
    Modified version of the Bone class from the BVH importer.
    '''


    __slots__ = (
        # Bone name, used as key in many places.
        "name",
        # Decorated bone parent, set in a later loop
        "parent",
        # Blender armature bone.
        "rest_bone",
        # Blender pose bone.
        "pose_bone",
        # Blender pose matrix.
        "pose_mat",
        # Blender rest matrix (armature space).
        "rest_arm_mat",
        # Blender rest matrix (local space).
        "rest_local_mat",
        # Pose_mat inverted.
        "pose_imat",
        # Rest_arm_mat inverted.
        "rest_arm_imat",
        # Rest_local_mat inverted.
        "rest_local_imat",
        # Last used euler to preserve euler compatibility in between keyframes.
        "prev_euler",
        # Is the bone disconnected to the parent bone?
        "skip_position",
        "rot_order",
        "rot_order_str",
        # Needed for the euler order when converting from a matrix.
        "rot_order_str_reverse",
    )

    # _eul_order_lookup = {
    #     'XYZ': (0, 1, 2),
    #     'XZY': (0, 2, 1),
    #     'YXZ': (1, 0, 2),
    #     'YZX': ,
    #     'ZXY': (2, 0, 1),
    #     'ZYX': (2, 1, 0),
    # }

    def __init__(
        self, 
        bone_name: str, 
        obj: Object, 
        arm: Armature
    ):
        self.name = bone_name
        self.rest_bone = arm.bones[bone_name]
        self.pose_bone = obj.pose.bones[bone_name]

        # TODO improve
        self.rot_order_str = "YZX"
        self.rot_order_str_reverse = self.rot_order_str[::-1] # XZY

        #self.rot_order = DecoratedBone._eul_order_lookup[self.rot_order_str]
        self.rot_order = (1, 2, 0) # YZX

        self.pose_mat = self.pose_bone.matrix

        self.rest_arm_mat = self.rest_bone.matrix_local
        self.rest_local_mat = self.rest_bone.matrix

        # inverted mats
        self.pose_imat = self.pose_mat.inverted()
        self.rest_arm_imat = self.rest_arm_mat.inverted()
        self.rest_local_imat = self.rest_local_mat.inverted()

        self.parent = None
        # self.prev_euler = Euler((0.0, 0.0, 0.0), self.rot_order_str_reverse)
        self.prev_euler = Euler((0.0, 0.0, 0.0), "XZY") # reverse order
        #self.skip_position = ((self.rest_bone.use_connect or root_transform_only) and self.rest_bone.parent)

    def update_posedata(self):
        self.pose_mat = self.pose_bone.matrix
        self.pose_imat = self.pose_mat.inverted()

# end class DecoratedBone


def get_anim_data(obj):
    '''
    Gets animation data from the Blender object.
    '''

    armature = obj.data
    pose = obj.pose
    scene = bpy.context.scene
    prev_scene_frame = scene.frame_current
    
    anim = {}
    anim["rotations"] = []
    anim["positions"] = []
    anim["offsets"] = []    # Vector objects
    anim["parents"] = []
    anim["names"] = []

    # 1. Get bone names, parents and offsets

    for bone in armature.bones:
        anim["names"].append(bone.name)
        if bone.parent:     # If bone has parent
            anim["parents"].append(anim["names"].index(bone.parent.name))
            anim["offsets"].append(bone.head_local - bone.parent.head_local)
        else:
            anim["parents"].append(-1)
            anim["offsets"].append(bone.head_local)

    # 2. Get positions and rotations

    bones_decorated = [DecoratedBone(bone.name, obj, armature) for bone in pose.bones]
    bones_decorated_dict = {dbone.name: dbone for dbone in bones_decorated}
    for dbone in bones_decorated:
        parent = dbone.rest_bone.parent
        if parent:
            dbone.parent = bones_decorated_dict[parent.name]
    del bones_decorated_dict

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        
        r = []  # rotation
        p = []  # position
        
        for dbone in bones_decorated:
            dbone.update_posedata()

        for dbone in bones_decorated:
            trans = Matrix.Translation(dbone.rest_bone.head_local)
            itrans = Matrix.Translation(-dbone.rest_bone.head_local)

            # Compute current bone location - act on parent bone (if exists), current pose and neutral armature pose
            if dbone.parent:
                mat_final = dbone.parent.rest_arm_mat @ dbone.parent.pose_imat @ dbone.pose_mat @ dbone.rest_arm_imat
                mat_final = itrans @ mat_final @ trans
                loc = mat_final.to_translation() + (dbone.rest_bone.head_local - dbone.parent.rest_bone.head_local)
            else:
                mat_final = dbone.pose_mat @ dbone.rest_arm_imat
                mat_final = itrans @ mat_final @ trans
                loc = mat_final.to_translation() + dbone.rest_bone.head
            
            # TODO improve
            loc2 = [loc[0], loc[2], -loc[1]]
            p.append(loc2)

            rot = mat_final.to_euler(dbone.rot_order_str_reverse, dbone.prev_euler)
            
            dbone.prev_euler = rot
            # TODO improve
            rot = [-rot[dbone.rot_order[0]], rot[dbone.rot_order[1]], rot[dbone.rot_order[2]]]
                                                                           
            # TODO delegate to a function
            order = "ZYX"
            mat = np.identity(3)
            
            for idx, axis in enumerate(order):
                angle_radians = rot[idx:idx + 1]

                # shape: (..., 1)
                sin = np.sin(angle_radians)
                cos = np.cos(angle_radians)

                # shape(..., 4)
                rot_mat = np.concatenate([cos, sin, sin, cos], axis=-1)
                # shape(..., 2, 2)
                rot_mat = rot_mat.reshape(*rot_mat.shape[:-1], 2, 2)

                if axis == "X":
                    rot_mat *= np.array([[1, -1], [1, 1]])
                    rot_mat = np.insert(rot_mat, 0, [0, 0], axis=-2)
                    rot_mat = np.insert(rot_mat, 0, [1, 0, 0], axis=-1)
                elif axis == "Y":
                    rot_mat *= np.array([[1, 1], [-1, 1]])
                    rot_mat = np.insert(rot_mat, 1, [0, 0], axis=-2)
                    rot_mat = np.insert(rot_mat, 1, [0, 1, 0], axis=-1)
                else:
                    rot_mat *= np.array([[1, -1], [1, 1]])
                    rot_mat = np.insert(rot_mat, 2, [0, 0], axis=-2)
                    rot_mat = np.insert(rot_mat, 2, [0, 0, 1], axis=-1)
                
                mat = np.matmul(mat, rot_mat)
                        
            r.append(mat)
       
        anim["rotations"].append(r)
        anim["positions"].append(p) 
        
    scene.frame_set(prev_scene_frame)

    # Change arrays to numpy arrays
    anim["rotations"] = np.array(anim["rotations"])
    anim["positions"] = np.array(anim["positions"])
    anim["offsets"] = np.array(anim["offsets"])
    anim["parents"] = np.array(anim["parents"])
    anim["names"] = np.array(anim["names"])
    
    return anim
# end function get_anim_data


def apply_transforms(
    obj, 
    true_original_pos: np.array, 
    true_inferred_pos,
    true_original_rot,
    true_inferred_rot,
    offset: int
) -> None:
    '''
    Apply transforms to the object.
    '''

    scene = bpy.context.scene
    prev_scene_frame = scene.frame_current
        
    num_frame = scene.frame_end - scene.frame_start + 1
    obj_offset = None
    
    pose = obj.pose
    
    action = bpy.data.actions.new(name="action1")
    obj.animation_data.action = action
    
    bone_data = {}
    for bone in pose.bones:
        # TODO improve
        bone.rotation_mode = "ZYX"
        bone.rotation_euler = Euler([0,0,0], bone.rotation_mode)
                
        bone_name = bone.name
        pose_bone = bone
        rest_bone = pose_bone.bone
        bone_rest_matrix = rest_bone.matrix_local.to_3x3()
        
        # TODO improve
        bone_rest_matrix[1], bone_rest_matrix[2] = bone_rest_matrix[2], -bone_rest_matrix[1]
                
        bone_rest_matrix_inv = Matrix(bone_rest_matrix)
        bone_rest_matrix_inv.invert()

        bone_rest_matrix_inv.resize_4x4()
        bone_rest_matrix.resize_4x4()
        
        bone_data[bone_name] = (pose_bone, rest_bone, bone_rest_matrix, bone_rest_matrix_inv)
    
    for i, bvh_node in enumerate(pose.bones):
        pose_bone, bone, bone_rest_matrix, bone_rest_matrix_inv = bone_data[bvh_node.name]
        
        # location
        if i == 0:
            data_path = 'pose.bones["%s"].location' % escape_identifier(pose_bone.name)
            location = [(0.0, 0.0, 0.0)] * num_frame
            for frame_i in range(scene.frame_start-1, scene.frame_end-1):
                if frame_i > offset - 1 and frame_i < offset + len(true_inferred_pos):
                    bvh_loc = true_inferred_pos[frame_i-offset][i].tolist()
                else:
                    bvh_loc = true_original_pos[frame_i][i].tolist()

                bone_translate_matrix = Matrix.Translation(Vector(bvh_loc))
                
                loc = (bone_rest_matrix_inv @ bone_translate_matrix).to_translation()
                 
                if obj_offset is None:
                    obj_offset = loc
                                  
                location[frame_i] = loc - obj_offset            
                
            for axis_i in range(3):
                curve = action.fcurves.new(data_path=data_path, index=axis_i, action_group=bvh_node.name)
                keyframe_points = curve.keyframe_points
                keyframe_points.add(num_frame)

                for frame_i in range(num_frame):
                    keyframe_points[frame_i].co = (
                        frame_i+1,
                        location[frame_i][axis_i],
                    )
        
        # rotation
        rotate = [(0.0, 0.0, 0.0)] * num_frame
        data_path = ('pose.bones["%s"].rotation_euler' % escape_identifier(pose_bone.name))

        prev_euler = Euler((0.0, 0.0, 0.0))
        for frame_i in range(scene.frame_start-1, scene.frame_end-1):
            if frame_i > offset - 1 and frame_i < offset + len(true_inferred_rot):
                bvh_rot = true_inferred_rot[frame_i-offset][i].tolist()                
            else:
                bvh_rot = true_original_rot[frame_i][i].tolist()
            
            # TODO improve
            bvh_rot = [radians(bvh_rot[0]), -radians(bvh_rot[1]), radians(180 - bvh_rot[2])]
            
            euler = Euler(bvh_rot, "XYZ")
            bone_rotation_matrix = euler.to_matrix().to_4x4()
            bone_rotation_matrix = (
                bone_rest_matrix_inv @
                bone_rotation_matrix @
                bone_rest_matrix
            )
            rotate[frame_i] = bone_rotation_matrix.to_euler(pose_bone.rotation_mode, prev_euler)
            prev_euler = rotate[frame_i]
            
        # For each euler angle x, y, z (or quaternion w, x, y, z).
        for axis_i in range(len(rotate[0])):
            curve = action.fcurves.new(data_path=data_path, index=axis_i, action_group=bvh_node.name)
            keyframe_points = curve.keyframe_points
            keyframe_points.add(num_frame)

            for frame_i in range(num_frame):
                keyframe_points[frame_i].co = (
                    frame_i+1,
                    rotate[frame_i][axis_i],
                )
                                            
    scene.frame_set(prev_scene_frame)
    
# end function apply_transforms
