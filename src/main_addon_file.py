# --- TODO improve

# import subprocess
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch'])



# --- Imports 

import sys
import os
import bpy

addon_path = bpy.data.filepath
directory_path = os.path.dirname(addon_path)
sys.path.insert(0, directory_path)

from interface.general_interface import GeneralInterface
from models.motion_inbetweening.motion_inbetweening import MotionInbetweeningInterface

from src.addon_functions import apply_transforms, get_anim_data
from src.utils import copy_object, convert_array_3x3matrix_to_euler_zyx

from bpy.utils import register_class, unregister_class
from bpy.types import PropertyGroup
from bpy.props import IntProperty, PointerProperty, BoolProperty 



# --- Main function

def generate_anim(
    start_frame: int, 
    end_frame: int, 
    post_processing: bool, 
    interface: GeneralInterface = MotionInbetweeningInterface()
) -> set[str]:
    
    '''
    Function doing all stuff including using of model.
    '''

    context = bpy.context
    scene = context.scene  
    scene_start_frame = scene.frame_start
    scene_end_frame = scene.frame_end
    obj = context.object

    # check simple things
    if len(context.selected_objects) != 1: return {"CANCELLED"} 
    if obj is None or obj.type != 'ARMATURE': return {"CANCELLED"}
    if start_frame < scene_start_frame or end_frame > scene_end_frame: return {"CANCELLED"}
    if start_frame >= end_frame: return {"CANCELLED"}
    
    # check if frames are correct in model - function from interface
    if interface.check_frames_range(start_frame, end_frame, scene_start_frame, scene_end_frame) == False: return {"CANCELLED"} 
          
    # load anim data
    anim = get_anim_data(obj)
    
    # infer results from ai model - function from interface
    inferred_pos, inferred_rot = interface.infer_anim(anim, start_frame, end_frame, post_processing)
    
    # copy object
    new_obj = copy_object(obj, context)

    # convert original rotation to ZYX Euler angles
    original_rot = convert_array_3x3matrix_to_euler_zyx(anim["rotations"])

    # apply new transforms
    apply_transforms(
        new_obj,
        true_original_pos=anim["positions"], 
        true_inferred_pos=inferred_pos, 
        true_original_rot=original_rot, 
        true_inferred_rot=inferred_rot, 
        offset=start_frame 
    )
    
    return {"FINISHED"}

# end function generate_anim



# --- Blender window stuff 

bl_info = {
    "name" : "Plugin",
    "author" : "Pluginowcy",
    "version" : (1, 0),
    "blender" : (4, 0, 0),
    "location" : "View3D > N",
    "description" : "",
    "warning" : "",
    "doc_url" : "",
    "category" : "",
} 
                                                            
class GenerationProperties(PropertyGroup):
    start_frame : IntProperty(name = "start frame", default = 0, min = 0)
    end_frame : IntProperty(name = "end frame", default = 0, min = 0)
    post_processing : BoolProperty(name = "post processing", default = False)
    
# end class GenerationProperties
     
class GenerationButtonOperator(bpy.types.Operator):
    bl_idname = "plugin.generation_button"
    bl_label = "plugin"

    def execute(self, context):
        mt = bpy.context.scene.my_tool
        return generate_anim(mt.start_frame, mt.end_frame, mt.post_processing)
         
# end class GenerateButtonOperator
       
class GenerationPanel(bpy.types.Panel):
    bl_label = "Plugin"
    bl_idname = "plugin.generation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "plugin"
    bl_context = "objectmode"   

    def draw(self, context):
        layout = self.layout
        scene = context.scene  
        mytool = scene.my_tool
        
        layout.prop(mytool, "start_frame")
        layout.prop(mytool, "end_frame")
        layout.prop(mytool, "post_processing")
        layout.separator()   
            
        row = layout.row()
        row.operator(GenerationButtonOperator.bl_idname, text="Generate animation", icon='COLORSET_01_VEC')
        
# end class GeneratePanel

generation_window_classes = [GenerationButtonOperator, GenerationPanel, GenerationProperties]

def register():
    for x in generation_window_classes: register_class(x)  
    bpy.types.Scene.my_tool = PointerProperty(type=GenerationProperties)
    
# end function register

def unregister():
    for x in generation_window_classes: unregister_class(x)
    del bpy.types.Scene.my_tool
    
# end function unregister
    
if __name__ == "__main__":
    register()    

# generate_anim(70, 140, False)
