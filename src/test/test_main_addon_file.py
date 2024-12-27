from unittest.mock import MagicMock
from .. import main_addon_file as maf
from .types import Bone, Object, Armature, Pose
from mathutils import Matrix, Vector
import bpy
import numpy as np

import pytest


@pytest.fixture(autouse=True)
def cleanup():
    bpy.ops.wm.read_homefile(app_template="")
    bpy.ops.object.select_all(action='DESELECT')


def test_generate_anim_when_no_object_selected():
    assert maf.generate_anim(1, 100, MagicMock(), True, False, False) == {'CANCELLED1'}


def test_generate_anim_when_selected_object_not_armature():
    obj = bpy.data.objects.new("object1", bpy.data.meshes.new('mesh1'))
    bpy.context.collection.objects.link(obj)
    obj.select_set(True)
    assert maf.generate_anim(1, 100, MagicMock(), True, False, False) == {'CANCELLED2'}


def test_generate_anim_when_more_than_one_object_selected():
    obj, _ = prepare_object(1)
    obj.select_set(True)
    obj2, _ = prepare_object(2)
    obj2.select_set(True)
    assert maf.generate_anim(1, 100, MagicMock(), True, False, False) == {'CANCELLED3'}


def test_generate_anim_when_wrong_start_frame():
    obj, _ = prepare_object(1)
    obj.select_set(True)
    assert maf.generate_anim(-1, 100, MagicMock(), True, False, False) == {'CANCELLED4'}


def test_generate_anim_when_wrong_end_frame():
    obj, _ = prepare_object(1)
    obj.select_set(True)
    assert maf.generate_anim(1, 3000, MagicMock(), True, False, False) == {'CANCELLED5'}


def test_generate_anim_when_wrong_frame_range():
    obj, _ = prepare_object(1)
    obj.select_set(True)
    assert maf.generate_anim(1, 1, MagicMock(), True, False, False) == {'CANCELLED6'}


def test_generate_anim_when_interface_wrong_frame_range():
    obj, _ = prepare_object(1)
    obj.select_set(True)
    interface = MagicMock()
    interface.check_frame_range.return_value = False, 0
    assert maf.generate_anim(1, 100, interface, True, False, False) == {'CANCELLED7'}


def test_generate_anim_when_interface_not_supported_skeleton():
    obj, _ = prepare_object(1)
    obj.select_set(True)
    interface = MagicMock()
    interface.check_frame_range.return_value = True, 0
    interface.is_skeleton_supported.return_value = False
    assert maf.generate_anim(1, 100, interface, True, False, False) == {'CANCELLED8'}


def test_generate_anim_when_error_loading_anim_data():
    obj, _ = prepare_object(1)
    obj.select_set(True)
    interface = MagicMock()
    interface.check_frame_range.return_value = True, 0
    interface.is_skeleton_supported.return_value = True
    assert maf.generate_anim(1, 100, interface, True, False, False) == {'CANCELLED9'}


def test_generate_anim_when_error_using_model():
    obj, _ = prepare_object(1, True)
    obj.select_set(True)
    interface = MagicMock()
    interface.check_frame_range.return_value = True, 0
    interface.is_skeleton_supported.return_value = True
    assert maf.generate_anim(1, 100, interface, True, False, False) == {'CANCELLED10'}


def test_generate_anim_when_error_applying_animation():
    obj, _ = prepare_object(1, True)
    obj.select_set(True)
    interface = MagicMock()
    interface.check_frame_range.return_value = True, 0
    interface.is_skeleton_supported.return_value = True
    interface.infer_anim.return_value = [[np.array([None, 30, 40])]], [[np.array([180, 240, 0])]]
    assert maf.generate_anim(1, 100, interface, True, False, False) == {'CANCELLED13'}


def test_generate_anim_when_correct():
    obj, _ = prepare_object(1, True)
    obj.select_set(True)
    interface = MagicMock()
    interface.check_frame_range.return_value = True, 0
    interface.is_skeleton_supported.return_value = True
    interface.infer_anim.return_value = [[np.array([20, 30, 40])]], [[np.array([180, 240, 0])]]
    assert maf.generate_anim(1, 100, interface, True, False, False) == {'FINISHED'}


def test_format_bvh_name():
    assert maf.format_bvh_name("test_tEst.bvh") == "Test test"


def prepare_object(id, add_bones = False):
    mesh = bpy.data.meshes.new('mesh' + str(id))
    obj = bpy.data.objects.new("object" + str(id), mesh)
    obj.animation_data_create()
    action = bpy.data.actions.new(name="action" + str(id))
    obj.animation_data.action = action
    bpy.context.collection.objects.link(obj)
    armature = bpy.data.armatures.new('Armature')
    armature_obj = bpy.data.objects.new('ArmatureObj', armature)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    if add_bones:
        bpy.ops.object.mode_set(mode='EDIT')
        bone = armature.edit_bones.new('Bone1')
        bone.head = (0, 0, 0)
        bone.tail = (0, 0, 1)
        bpy.ops.object.mode_set(mode='OBJECT')
    modifier = obj.modifiers.new(name='ArmatureMod', type='ARMATURE')
    modifier.object = armature_obj
    obj.parent = armature_obj
    return obj, armature_obj
