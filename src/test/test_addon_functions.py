from .. import addon_functions as add_f
import numpy as np
from .types import Bone, Object, Armature, Pose
from mathutils import Matrix, Vector
import bpy
import pytest


@pytest.fixture(autouse=True)
def cleanup():
    bpy.ops.wm.read_homefile(app_template="")
    bpy.ops.object.select_all(action='DESELECT')


def test_should_return_skeleton():
    bone1 = Bone("bone1", None)
    bone2 = Bone("bone2", bone1)
    bone3 = Bone("bone3", bone1)
    bone4 = Bone("bone4", bone2)
    assert sorted(add_f.get_object_skeleton(Object(Armature([bone1, bone2, bone3, bone4]), None))) == \
           sorted([("bone1", None), ("bone3", "bone1"), ("bone2", "bone1"), ("bone4", "bone2")])


def test_get_anim_data():
    bpy.context.scene.frame_end = 1
    a_bone1 = Bone("bone1", None, Vector([1, 2, 3]), Vector([-1, -2, -3]),
                   Matrix([[1, 2, 3, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                   Matrix([[1, 0, 0, 0], [1, 2, 3, 4], [0, 0, 1, 0], [0, 0, 0, 1]]))
    a_bone2 = Bone("bone2", a_bone1, Vector([4, 5, 6]), Vector([-4, -5, -6]),
                   Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [1, 2, 3, 4], [0, 0, 0, 1]]),
                   Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 2, 3, 4]]))
    p_bone1 = Bone("bone1", None, Vector([7, 8, 9]), Vector([-7, -8, -9]),
                   Matrix([[5, 6, 7, 8], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                   Matrix([[1, 0, 0, 0], [5, 6, 7, 8], [0, 0, 1, 0], [0, 0, 0, 1]]))
    p_bone2 = Bone("bone2", p_bone1, Vector([1, 2, 3]), Vector([-1, -2, -3]),
                   Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [5, 6, 7, 8], [0, 0, 0, 1]]),
                   Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [5, 6, 7, 8]]))

    anim_data = add_f.get_anim_data(Object(Armature([a_bone1, a_bone2]), Pose([p_bone1, p_bone2])))

    np.testing.assert_allclose(anim_data["rotations"], np.array(
        [
            [
                [
                    [0.97, -0.23, -0.07],
                    [0, 0.28, -0.96],
                    [0.24, 0.93, 0.27]
                ],
                [
                    [-0.8, -0.54, 0.26],
                    [0.26, 0.09, 0.96],
                    [-0.55, 0.83, 0.07]
                ]
            ]
        ]
    ), atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(anim_data["positions"][0], np.array(
        [
            [-4, 3, -6],
            [58.4, 20.5, - 11.65]
        ]
    ), atol=1e-2, rtol=1e-2)
    assert np.array_equal(anim_data["offsets"], [[-1, -2, -3], [-3, -3, -3]])
    assert np.array_equal(anim_data["parents"], [-1, 0])
    assert np.array_equal(anim_data["names"], ["bone1", "bone2"])


def test_apply_transforms():
    bpy.context.scene.frame_end = 3
    bone1 = Bone("bone1", None, None, None, None, Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    object = Object(Armature([bone1]), Pose([bone1]))
    original_pos = [[np.array([1, 2, 3])], [np.array([4, 5, 6])], [np.array([7, 8, 9])]]
    inferred_pos = [[np.array([20, 30, 40])]]
    original_rot = [[np.array([0, 90, 30])], [np.array([90, 20, 0])], [np.array([67, 3, 90])]]
    inferred_rot = [[np.array([180, 240, 0])]]
    add_f.apply_transforms(object, original_pos, inferred_pos, original_rot, inferred_rot, 1)

    out_positions = np.array([
        [object.animation_data.action.fcurves.find("pose.bones[\"bone1\"].location", index=i)
         .keyframe_points[j].co[1] for i in range(3)]for j in range(3)])

    out_rotations = np.array([
        [object.animation_data.action.fcurves.find("pose.bones[\"bone1\"].rotation_euler", index=i)
         .keyframe_points[j].co[1] for i in range(3)] for j in range(3)])

    np.testing.assert_allclose(out_positions, np.array(
        [
            [0, 0, 0],
            [19, -37, 28],
            [6, - 6, 6]
        ]
    ), atol=1e-2, rtol=1e-2)

    np.testing.assert_allclose(out_rotations, np.array(
        [
            [0, -2.61, -1.57],
            [-3.14, -3.14, -2.09],
            [-1.52, - 2.74, -1.57]
        ]
    ), atol=1e-2, rtol=1e-2)
