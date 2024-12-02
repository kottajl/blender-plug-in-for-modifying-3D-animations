# --- Function for showing info in blender

import bpy

def show_info(type, msg):
    bpy.context.window_manager.popup_menu(
        lambda self, context: self.layout.label(text=msg), 
        title=type, 
        icon=type
    )

# end show_info



# --- Instaling libraries for addon in blender python instance if not installed

import subprocess
import sys
import importlib
import pathlib
import os
import json
import platform 

directory_path = pathlib.Path(bpy.context.space_data.text.filepath).parent.resolve()
modules_path = str(directory_path.parent.resolve())
sys.path.append(modules_path)

os_name = platform.system()
req_path = str(directory_path) + str(os.sep) + "config" + str(os.sep) + "addon_requirements.txt"

def handle_pytorch3d_installation():
    try:
        import pytorch3d
    except ModuleNotFoundError:
        print("Started installing pytorch3d.")
        if os_name == "Windows":
            pytorch3d_whl_path = str(directory_path).removesuffix("src") + "lib" +  str(os.sep) + "pytorch3d-0.7.8-cp310-cp310-win_amd64.whl"
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', pytorch3d_whl_path])
            except Exception as e:
                print("\033[33m" + str(e) + "\033[0m")
                return False
            else:
                print("Successfully installed pytorch3d.")
                return True
            
        elif os_name == "Darwin": # Mac
            print("To install pytorch3d on MAC you have to install it from source.")
            print("https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#building--installing-from-source")
            print("Install it in Blender Python instance using python -m [commands].")
            return False
    
        else: # Linux
            try:
                import torch
                version_str = "".join([
                    "py38_cu",
                    torch.version.cuda.replace(".",""),
                    "_pyt1110"
                ])
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', "iopath"])
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', "pytorch3d", "-f", f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html"])
            except Exception as e:
                print("\033[33m" + str(e) + "\033[0m")
                return False
            else:
                print("Successfully installed pytorch3d.")
                return True
            
    else:
        print("Pytorch3d already installed.")
        return True

# end handle_pytorch3d_installation

def handle_torch_installation():
    try:
        import torch
        import torchaudio
        import torchvision
    except ModuleNotFoundError:
        pip_parts = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']
        if os_name == "Windows":
            pip_parts.append('--index-url')
            cuda_path = os.getenv('CUDA_PATH')
            if cuda_path: 
                cuda_version = cuda_path[-4] + cuda_path[-3] + cuda_path[-1]
                if cuda_version in ["124", "121"]: url = "https://download.pytorch.org/whl/cu" + cuda_version
                else: url = "https://download.pytorch.org/whl/cu118"
                pip_parts.append(url)
            else:
                pip_parts.append("https://download.pytorch.org/whl/cu118")

        elif os_name == "Linux":
            cuda_present = False
            try:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    cuda_present = True
                    cuda_version_line = result.stdout.splitlines()[0]
                    cuda_version = cuda_version_line.split(",")[1].split()[1]  
                    cuda_version = str(cuda_version)
                    cuda_version = cuda_version[0] + cuda_version[1] + cuda_version[3]
                    if version in ["118", "121"]: 
                        pip_parts.append('--index-url')
                        url = "https://download.pytorch.org/whl/cu" + cuda_version
                        pip_parts.append(url)
            except FileNotFoundError:
                pass

            if not cuda_present:
                try:
                    result = subprocess.run(['rocminfo'], capture_output=True, text=True)
                    if result.returncode == 0:
                        pip_parts.append('--index-url')
                        pip_parts.append("https://download.pytorch.org/whl/rocm6.2")
                except FileNotFoundError:
                    pass
        
        try:
            print("Started installing torch, torchvision and torchaudio.")
            subprocess.check_call(pip_parts)
        except Exception as e:
            print("\033[33m" + str(e) + "\033[0m")
            return False
        else:
            print("Successfully installed torch, torchvision and torchaudio.")
            return True
                
    else:
        print("Torch, torchvision and torchaudio already installed.")
        return True
    
# end handle_torch_installation

installing = False
with open(req_path, 'r', encoding='utf-8') as file:
    for line in file:
        x = line.strip()
        if not x or x.startswith('#'): continue

        pip_parts = [sys.executable, '-m', 'pip', 'install'] + [
            part.replace("==", ">=") if "==" in part else part for part in x.split()
        ]

        lib_name = pip_parts[4]
        if '>=' in lib_name: lib_name = lib_name.split('>=')[0]

        try:
            importlib.import_module(lib_name)
        except ModuleNotFoundError: 
            if not installing: 
                installing = True
                show_info("INFO", "Started installing libraries for addon. For more information refer to terminal.")
                print("--- Updating PIP")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
                print("--- Installing libraries")

            if lib_name in ["torch", "torchvision", "torchaudio"]: handle_torch_installation()
            elif lib_name == "pytorch3d": handle_pytorch3d_installation()     
            else: subprocess.check_call(pip_parts)
                
if installing:
    print("--- Done")
    show_info("INFO", "Done installing libraries for addon.")



# --- Downloading large files for models

import requests
import zipfile

def download_and_extract_zip(target_dir, url):
    zip_name = os.path.basename(url)
    zip_path = os.path.join(target_dir, zip_name)
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(zip_path)
        show_info("INFO", "Successfully ended downloading file.")
        print("Successfully ended downloading file.")
    else:
        show_info("ERROR", "Error while downloading file.")
        print("Error while downloading file.")

# end download_and_extract_zip

target_dir = str(directory_path).removesuffix("src") + "models" +  str(os.sep) + "motion_inbetweening" + str(os.sep) + "datasets"
target_dir += str(os.sep) + "lafan1"
bvh_files = [f for f in os.listdir(target_dir) if f.endswith(".bvh")]
if not bvh_files:
    show_info("INFO", "Started downloading 137MB file.")
    print("Started downloading 137MB file.")
    url = "https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip"
    download_and_extract_zip(target_dir, url)

target_dir = str(directory_path).removesuffix("src") + "models" +  str(os.sep) + "motion_inbetweening" + str(os.sep) + "experiments"
bvh_files = [f for f in os.listdir(target_dir + str(os.sep) + "lafan1_context_model_release") if f.endswith(".pth")]
if not bvh_files:
    show_info("INFO", "Started downloading 204MB file.")
    print("Started downloading 204MB file.")
    url = "https://github.com/victorqin/motion_inbetweening/releases/download/v1.0.0/pre-treained.zip"
    download_and_extract_zip(target_dir, url)

# --- Addon imports 

import torch

from interface.general_interface import GeneralInterface

import src.metrics as metrics
from src.addon_functions import apply_transforms, get_anim_data
from src.utils import copy_object, convert_array_3x3matrix_to_euler_zyx, has_missing_keyframes_between, export_dict_to_file

import bpy_extras
from bpy.utils import register_class, unregister_class
from bpy.types import PropertyGroup
from bpy.props import IntProperty, PointerProperty, BoolProperty, EnumProperty, StringProperty, FloatProperty

sys.path.remove(modules_path)



# --- Main function

def generate_anim(
    start_frame: int, 
    end_frame: int, 
    interface: GeneralInterface,
    create_new: bool,
    calculate_metrics: bool,
    **kwargs: dict
) -> set[str]:
    
    '''
    Function doing all stuff including using of model.
    '''

    # blender things
    context = bpy.context
    scene = context.scene  
    scene_start_frame = scene.frame_start
    scene_end_frame = scene.frame_end
    obj = context.object

    # check simple things
    if obj is None or len(context.selected_objects) == 0:
        show_info("ERROR", "No object selected.")
        return {"CANCELLED"} 

    if obj.type != 'ARMATURE':
        show_info("ERROR", "Selected object must be armature.")
        return {"CANCELLED"} 
    
    if len(context.selected_objects) > 1: 
        show_info("ERROR", "Only 1 object can be selected.")
        return {"CANCELLED"}  

    if start_frame < scene_start_frame or end_frame > scene_end_frame or start_frame + 1 >= end_frame: 
        show_info("ERROR", "Selected frames range is invalid.")
        return {"CANCELLED"}
    
    # check if frames are correct in model - function from interface
    b, s = interface.check_frames_range(start_frame, end_frame, scene_start_frame, scene_end_frame)
    if b == False: 
        show_info("ERROR", f"Selected frame range is invalid for model, {s}")
        return {"CANCELLED"} 
    
    # load anim data
    try: 
        anim = get_anim_data(obj)
    except Exception as e: 
        show_info("ERROR", f"Error with loading animation data, {e}.")
        return {"CANCELLED"} 

    # infer results from ai model - function from interface
    try:
        print("--- Start of model infer anim logs")
        inferred_pos, inferred_rot = interface.infer_anim(anim, start_frame, end_frame, **kwargs)
        print("--- End of model infer anim logs")
    except Exception as e: 
        show_info("ERROR", f"Error with using model, {e}.")
        return {"CANCELLED"} 
    
    # copy object
    try:
        if create_new: new_obj = copy_object(obj, context)
        else: new_obj = obj
    except Exception as e: 
        show_info("ERROR", f"Error while copying object, {e}.")
        return {"CANCELLED"} 

    # convert original rotation to ZYX Euler angles
    try:
        original_rot = convert_array_3x3matrix_to_euler_zyx(anim["rotations"])
    except Exception as e: 
        show_info("ERROR", f"Error with animation data, {e}.")
        return {"CANCELLED"} 

    # apply new transforms
    try:
        apply_transforms(
            new_obj,
            true_original_pos=anim["positions"], 
            true_inferred_pos=inferred_pos, 
            true_original_rot=original_rot, 
            true_inferred_rot=inferred_rot, 
            offset=start_frame 
        )
    except Exception as e: 
        show_info("ERROR", f"Error while applying new animation to object, {e}.")
        return {"CANCELLED"} 
    
    # calculate metrics
    if calculate_metrics:
        global generated_metrics

        # If we are creating a new object and it has no missing keyframes between start and end frame, we can calculate comparative metrics.
        # Otherwise we calculate only non-comparative metrics on the generated object.
        original_obj_param = obj if create_new and not has_missing_keyframes_between(obj, (start_frame, end_frame)) else None

        generated_metrics = metrics.calculate_metrics(
            orginal_obj=original_obj_param, 
            generated_obj=new_obj, 
            start_frame=start_frame, 
            end_frame=end_frame
        )
        bpy.ops.metrics.metrics_output_window('INVOKE_DEFAULT')

    return {"FINISHED"}

# end function generate_anim



# --- Blender window stuff 

bl_info = {
    "name" : "Addon",
    "author" : "Pluginowcy",
    "version" : (1, 0),
    "blender" : (4, 0, 0),
    "location" : "View3D > N",
    "description" : "",
    "warning" : "",
    "doc_url" : "",
    "category" : "",
} 

# Global variables
selected_model : GeneralInterface = None
selected_device : str = None
selected_model_index  : int = 0
addon_model_count : int = 0
current_kwargs_attributes = []
device_arg_name = ""
generated_metrics: dict = {}

with open(directory_path / "config" / "model_paths.json", "r") as file:
    data = json.load(file)
    addon_model_count = data["addon_model_count"]

def select_ai_model(index):  
    try:
        model_path = pathlib.Path(get_ai_models()[index][2])
        model = importlib.import_module(str(model_path.name))
        global selected_model, selected_model_index
        selected_model = model.ModelInterface()
        selected_model_index = index
        bpy.context.scene.my_tool.update_properties(bpy.context)
    except Exception as e:
        select_ai_model(0)
        raise(e)

# end function select_ai_model

def select_ai_model_dropdown(self, context):
    select_ai_model(int(self.model))

# end function select_ai_model_dropdown

def get_ai_models_dropdown(self, context):
    return get_ai_models()

# end function get_ai_models_dropdown

def get_ai_models():
    models = [] 
    i = 0
    for model in json.load(open(directory_path / "config" / "model_paths.json"))["model_paths"]:
        models.append((str(i), model["name"], model["path"]))
        model_path = pathlib.Path(model["path"])
        if not os.path.isabs(model_path):
            model_path = pathlib.Path(modules_path) / model_path
        sys.path.append(str(model_path.parent.resolve()))
        i += 1
    
    return models

# end function get_ai_models

def get_device_list(self, context):
    devices = [("0", "CPU", "cpu")] 
    device_i = 1

    # Cuda for NVIDIA GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            devices.append((str(device_i), f"GPU {i}", f"cuda:{i}"))
            device_i += 1

    # MPS for ARM GPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.append((str(device_i), "MPS", "mps"))
        device_i += 1

    return devices

# end function get_device_list

def select_device(self, context):
    global selected_device
    selected_device = get_device_list(self, context)[int(getattr(self, device_arg_name))][2]

# end function select_device

def format_bvh_name(filename):
    filename = filename.removesuffix(".bvh")
    filename = filename.replace("_", " ").capitalize()
    return filename

# end format_bvh_name

sample_bvh_path = str(directory_path).removesuffix("src") + "sample_bvh_files"
def sample_bvh_files_enum(self, context):
    files = [f for f in os.listdir(sample_bvh_path) if f.endswith(".bvh")]
    return sorted([(f, format_bvh_name(f), f) for f in files], key=lambda x: x[2], reverse=True)
         
# end sample_bvh_files_enum

def get_scene_objects_enum(self, context):
    return [(obj.name, obj.name, obj.name) for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']

# end get_scene_objects_enum
                                                        
class GenerationProperties(PropertyGroup): 
    start_frame : IntProperty(name = "Start frame", default = 0, min = 0)
    end_frame : IntProperty(name = "End frame", default = 0, min = 0)
    model : EnumProperty(
        name="AI model",
        description="Select model for generating frames",
        items=get_ai_models_dropdown,
        update=select_ai_model_dropdown,
        default=0
    )
    bvh_file_name : EnumProperty(
        name="BVH file",
        items=sample_bvh_files_enum,
        description="Select sample BVH file to import",
        default=0
    )
    fbx_export_object : EnumProperty(
        name="Object name",
        description="Select object to export to FBX",
        items=get_scene_objects_enum,
        default=0
    )
    create_new : BoolProperty(name = "  Create new object", default = True)
    calculate_metrics: BoolProperty(name = "  Calculate metrics", default = False)

    def update_properties(self, context): 
        global current_kwargs_attributes
        for x in current_kwargs_attributes: delattr(GenerationProperties, x) 
        current_kwargs_attributes = []
    
        if selected_model is not None:
            kwargs_list = selected_model.get_infer_anim_kwargs()
            device_added = False
            for var_type, var_name, var_desc in kwargs_list:
                if var_type == torch.device and not device_added:
                    setattr(GenerationProperties, var_name, EnumProperty(
                        name=var_name,
                        description=var_desc,
                        items=get_device_list,
                        update=select_device,
                        default=0
                    ))
                    device_added = True
                    global device_arg_name
                    device_arg_name = var_name
                    current_kwargs_attributes.append(var_name)
                elif var_type == bool:
                    setattr(GenerationProperties, var_name, BoolProperty(
                        name= "  " + var_name,
                        description=var_desc,
                        default=False
                    ))
                    current_kwargs_attributes.append(var_name)
                elif var_type == int:
                    setattr(GenerationProperties, var_name, IntProperty(
                        name= var_name,
                        description=var_desc,
                        default=0
                    ))
                    current_kwargs_attributes.append(var_name)
                elif var_type == float:
                    setattr(GenerationProperties, var_name, FloatProperty(
                        name= var_name,
                        description=var_desc,
                        default=0.0
                    ))
                    current_kwargs_attributes.append(var_name)
                elif var_type == str:
                    setattr(GenerationProperties, var_name, StringProperty( 
                        name= var_name,
                        description=var_desc,
                        default=""
                    ))
                    current_kwargs_attributes.append(var_name)

# end class GenerationProperties

def get_selected_keyframes():
    context = bpy.context
    obj = context.object

    if obj is None or len(context.selected_objects) == 0:
        show_info("ERROR", "No object selected.")
        return {"CANCELLED"} 

    if obj.type != 'ARMATURE':
        show_info("ERROR", "Selected object must be armature.")
        return {"CANCELLED"} 
    
    if len(context.selected_objects) > 1: 
        show_info("ERROR", "Only 1 object can be selected.")
        return {"CANCELLED"}  
    
    selected_frames = []
    action = bpy.context.object.animation_data.action  # Active action
    
    if action:
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                if keyframe.select_control_point:  
                    selected_frames.append(int(keyframe.co.x))

    return sorted(set(selected_frames))  

# end function get_selected_keyframes

def process_generation():
    mt = bpy.context.scene.my_tool
    kwargs = dict()

    if selected_model is not None:
        kwargs_list = selected_model.get_infer_anim_kwargs()
        for t, n, d in kwargs_list:
            if t == torch.device: kwargs[n] = selected_device
            else: kwargs[n] = getattr(mt, n) 

    return generate_anim(mt.start_frame, mt.end_frame, selected_model, mt.create_new, mt.calculate_metrics, **kwargs)   

# end process_generation
     
class GenerationButtonOperator(bpy.types.Operator):
    bl_idname = "plugin.generation_button"
    bl_label = "Generate frames"

    def execute(self, context):
       return process_generation()
  
# end class GenerateButtonOperator

class dope_sheet_generation_button(bpy.types.Operator):
    bl_idname = "dope.dope_sheet_generation_button"
    bl_label = "Generate frames"
    
    def execute(self, context):
        selected_frames = get_selected_keyframes()
        if len(selected_frames) != 2 or selected_frames[0] + 1 == selected_frames[1]:
            show_info("ERROR", "Wrong frame range selected.")
            return {"CANCELLED"} 
        else:
            mt = bpy.context.scene.my_tool
            mt.start_frame, mt.end_frame = selected_frames[0], selected_frames[1]
            return process_generation()

# end class dope_sheet_generation_button  

class AddModelButtonOperator(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    bl_idname = "plugin.add_model_button"
    bl_label = "Add model"

    filename_ext = ".py"
    filter_glob: bpy.props.StringProperty(default="*.py", options={'HIDDEN'})

    def execute(self, context):
        with open(directory_path / "config" /  "model_paths.json", "r") as file:
            models = json.load(file)
        try:
            path = pathlib.Path(self.filepath)
            models["model_paths"].append({"name": str(path.stem), "path": str(path.parent.resolve() / path.stem)})
        except Exception as e:
            show_info("ERROR", f"Someting went wrong while adding model, {e}.")
            return {"CANCELLED"} 
        
        with open(directory_path / "config" / "model_paths.json", "w") as file:
            file.write(json.dumps(models, indent=4))
        try:   
            select_ai_model((len(get_ai_models_dropdown(self, context)) - 1))
        except Exception as e:
            with open(directory_path / "config" / "model_paths.json", "r") as file:
                models = json.load(file)
            del models["model_paths"][len(models["model_paths"]) - 1]
            with open(directory_path / "config" / "model_paths.json", "w") as file:
                json.dump(models, file, indent=4)
                show_info("ERROR", f"Someting went wrong while adding model, {e}.")
            return {"CANCELLED"} 
        
        show_info("INFO", "Model added succesfully.")
        return {"FINISHED"}
         
# end class AddModelButtonOperator

class InstallLibrariesOperator(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    bl_idname = "plugin.install_libraries_button"
    bl_label = "Install"

    filename_ext = ".txt"
    filter_glob: bpy.props.StringProperty(default="*.txt", options={'HIDDEN'})

    def execute(self, context):
        show_info("INFO", "Started installing libraries from TXT. ")
        print("--- Updating PIP")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

        print("--- Installing libraries")
        not_installed = []

        with open(self.filepath, 'r', encoding='utf-8') as file:
            for line in file:
                x = line.strip()
                if not x or x.startswith('#'): continue

                pip_parts = [sys.executable, '-m', 'pip', 'install'] + [
                    part.replace("==", ">=") if "==" in part else part for part in x.split()
                ]

                lib_name = pip_parts[4]
                if '>=' in lib_name: lib_name = lib_name.split('>=')[0]

                if lib_name in ["torch", "torchvision", "torchaudio"]: 
                    if handle_torch_installation() == False: not_installed.append(x)
                    continue
                elif lib_name == "pytorch3d": 
                    if handle_pytorch3d_installation() == False: not_installed.append(x)
                    continue

                try:
                    subprocess.check_call(pip_parts)
                except Exception as e:
                    print("\033[33m" + str(e) + "\033[0m")
                    not_installed.append(x)

        if len(not_installed) > 0:
            show_info("ERROR", "Some libraries weren't successfully installed. For more information refer to terminal.")
            print("--- Not installed libraries")
            for x in not_installed: print(x)
            print("--- End")
        else:
            show_info("INFO", "Successfully installed all libraries.")
            print("--- Successfully installed all libraries")

        return {"FINISHED"}
    
# end class InstallLibrariesOperator

class DeleteModelButtonOperator(bpy.types.Operator):
    bl_idname = "plugin.delete_model_button"
    bl_label = "Delete selected model"

    def execute(self, context):
        if selected_model_index < addon_model_count:
            show_info("ERROR", "Can't delete premade models.")
            return {"CANCELLED"} 
        else:
            with open(directory_path / "config" / "model_paths.json", "r") as file:
                models = json.load(file)
            del models["model_paths"][selected_model_index]
            with open(directory_path / "config" / "model_paths.json", "w") as file:
                json.dump(models, file, indent=4)

            select_ai_model(0)
            show_info("INFO", "Deleted model succesfully.")
            return {"FINISHED"}
        
# end class DeleteModelButtonOperator

class ImportBVHButton(bpy.types.Operator):
    bl_idname = "import_scene.bvh_file"
    bl_label = "Import"  

    def execute(self, context):
        mt = bpy.context.scene.my_tool
        selected_file = mt.bvh_file_name
        if selected_file:
            file_path = os.path.join(sample_bvh_path, selected_file)
            try:
                bpy.ops.import_anim.bvh(filepath=file_path)
            except Exception as e:
                show_info("ERROR", f"Error while importing BVH file, {e}")
                return {'CANCELLED'}
            else:
                show_info("INFO", "BVH file imported successfully.")
                return {'FINISHED'}
        else:
            show_info("ERROR", "BVH file to import not selected.")
            return {'CANCELLED'}

# end ImportBVHButton

class OpenBVHIportMenu(bpy.types.Operator):
    bl_idname = "import_scene.import_popup"
    bl_label = "Import sample BVH file"

    def draw(self, context):
        layout = self.layout
        scene = context.scene  
        mytool = scene.my_tool

        layout.prop(mytool, "bvh_file_name")
        layout.operator(ImportBVHButton.bl_idname, text="Import selected BVH file")

    def execute(self, context):
        return context.window_manager.invoke_popup(self)
    
# end OpenBVHIportMenu

class ExportFBXButton(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname = "export_scene.fbx_file"
    bl_label = "Export"
    
    filename_ext = ".fbx"
    filter_glob: StringProperty(default="*.fbx", options={'HIDDEN'})

    def execute(self, context):
        mt = bpy.context.scene.my_tool
        selected_object_name = mt.fbx_export_object

        if selected_object_name:
            selected_object = bpy.context.scene.objects.get(selected_object_name)
            if selected_object:
                original_selection = [obj for obj in bpy.context.selected_objects]
                bpy.ops.object.select_all(action='DESELECT')
                selected_object.select_set(True)
                file_path = self.filepath  
                try:
                    bpy.ops.export_scene.fbx(filepath=file_path)
                    show_info("INFO", f"Successfully exported {selected_object_name} to {file_path}.")
                    return {'FINISHED'}
                except Exception as e:
                    show_info("ERROR", f"Error while exporting FBX file, {e}")
                    return {'CANCELLED'}
                finally:
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in original_selection: obj.select_set(True)
            else:
                show_info("ERROR", "No object selected for export.")
                return {'CANCELLED'}
        else:
            show_info("ERROR", "No object selected for export.")
            return {'CANCELLED'}

# end ExportFBXButton

class OpenFBXExportMenu(bpy.types.Operator):
    bl_idname = "export_scene.export_popup"
    bl_label = "Export FBX File"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool

        layout.prop(mytool, "fbx_export_object")
        layout.operator(ExportFBXButton.bl_idname, text="Export selected object to FBX file")

    def execute(self, context):
        return context.window_manager.invoke_popup(self)
    
# end OpenFBXExportMenu
       
class PLUGIN_PT_GenerationPanel(bpy.types.Panel):
    bl_label = "Addon"
    bl_idname = "PLUGIN_PT_generation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Addon"
    bl_context = ""   

    def draw(self, context):
        layout = self.layout
        scene = context.scene  
        mytool = scene.my_tool

        layout.prop(mytool, "model")
        for x in current_kwargs_attributes: layout.prop(mytool, x)
        
        layout.separator()
        layout.separator()
        layout.separator()

        layout.operator(AddModelButtonOperator.bl_idname, text="Add model from directory")
        layout.operator(DeleteModelButtonOperator.bl_idname, text="Delete selected model")
        layout.operator(InstallLibrariesOperator.bl_idname, text="Install libraries from TXT") 

        layout.separator()
        layout.separator()
        layout.separator()

        layout.operator(OpenBVHIportMenu.bl_idname, text="Import sample BVH file")
        layout.operator(OpenFBXExportMenu.bl_idname, text="Export object to FBX file")

        layout.separator()
        layout.separator()
        layout.separator()
              
        layout.prop(mytool, "start_frame")
        layout.prop(mytool, "end_frame")   
        layout.prop(mytool, "create_new")
        layout.prop(mytool, "calculate_metrics")

        layout.operator(GenerationButtonOperator.bl_idname, text="Generate frames")
        
# end class GeneratePanel

class dope_sheet_options_button(bpy.types.Operator):
    bl_idname = "dope.dope_sheet_options_button"
    bl_label = "Generation options"
    bl_description = "Opens addon generation options"
    
    def execute(self, context):
        return context.window_manager.invoke_popup(self)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
                    
        layout.separator()

        layout.prop(mytool, "model")
        for x in current_kwargs_attributes: layout.prop(mytool, x)

        layout.separator()

        layout.prop(mytool, "create_new")
        layout.prop(mytool, "calculate_metrics")

        layout.separator()
    
        layout.operator(AddModelButtonOperator.bl_idname, text="Add model from directory")
        layout.operator(DeleteModelButtonOperator.bl_idname, text="Delete selected model")
        layout.operator(InstallLibrariesOperator.bl_idname, text="Install libraries from TXT")   
        
        layout.separator()
       
# end class dope_sheet_options_button

class dope_sheet_import_button(bpy.types.Operator):
    bl_idname = "dope.dope_sheet_import_button"
    bl_label = "Import sample BVH"
    bl_description = "Import sample BVH files"
    
    def execute(self, context):
        return context.window_manager.invoke_popup(self)

    def draw(self, context):
        layout = self.layout
        scene = context.scene  
        mytool = scene.my_tool

        layout.separator()  
        
        layout.prop(mytool, "bvh_file_name")
        layout.operator(ImportBVHButton.bl_idname, text="Import selected BVH file")

        layout.separator()  
       
       
# end class dope_sheet_import_button

class dope_sheet_export_button(bpy.types.Operator):
    bl_idname = "dope.dope_sheet_export_button"
    bl_label = "Export to FBX"
    bl_description = "Export object to FBX file"
    
    def execute(self, context):
        return context.window_manager.invoke_popup(self)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool

        layout.separator()  

        layout.prop(mytool, "fbx_export_object")
        layout.operator(ExportFBXButton.bl_idname, text="Export selected object to FBX file")

        layout.separator()  
       
# end class dope_sheet_export_button

class MetricsOutputWindow(bpy.types.Operator):
    bl_idname = "metrics.metrics_output_window"
    bl_label = "Generated metrics"

    def execute(self, context):
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        
        layout.separator()

        global generated_metrics
        for key, value in generated_metrics.items():
            layout.label(text=f"{key}: {value}")

        layout.separator()
        layout.operator(ExportMetricsButton.bl_idname, text="Export to file")

# end class MetricsOutputWindow

class ExportMetricsButton(bpy.types.Operator):
    bl_idname = "metrics.export_metrics_button"
    bl_label = "Export metrics"
    bl_description = "Export metrics data to file"

    file_format: bpy.props.EnumProperty(
        name="File Format",
        description="Choose the file format",
        items=[
            ('TXT', "Text File (.txt)", "Save metrics data as a text file"),
            ('JSON', "JSON File (.json)", "Save metrics data as a JSON file"),
            ('CSV', "CSV File (.csv)", "Save metrics data as a CSV file")
        ],
        default='TXT'
    )

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    default_filename: str = "metrics_output.txt"

    def execute(self, context):
        try:
            global generated_metrics

            # Set the right name and extension
            filename, file_extension = os.path.splitext(self.filepath)
            if file_extension[1:].lower() != self.file_format.lower():
                filename += file_extension

            export_dict_to_file(
                data=generated_metrics, 
                filename=filename, 
                export_type=self.file_format
            )

            show_info("INFO", f"Metrics data successfully exported to file.")

        except Exception as e:
            show_info("ERROR", f"Error while exporting metrics data to file, {e}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
        
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "file_format", text="File Format")
    
    def invoke(self, context, event):
        self.filepath = os.path.join(os.path.dirname(self.filepath), self.default_filename)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

# end class ExportMetricsButton

def draw_buttons_in_dope_sheet(self, context):
    layout = self.layout
    layout.separator()  
    layout.operator(dope_sheet_generation_button.bl_idname, text="Generate frames")
    layout.operator(dope_sheet_options_button.bl_idname, text="Generation options")
    layout.operator(dope_sheet_import_button.bl_idname, text="Import sample BVH")
    layout.operator(dope_sheet_export_button.bl_idname, text="Export object to FBX")

# end function draw_buttons_in_dope_sheet

classes_to_register = [
        InstallLibrariesOperator, 
        AddModelButtonOperator, 
        GenerationButtonOperator, 
        PLUGIN_PT_GenerationPanel, 
        GenerationProperties, 
        DeleteModelButtonOperator,
        dope_sheet_generation_button,
        dope_sheet_options_button,
        ImportBVHButton,
        OpenBVHIportMenu,
        ExportFBXButton,
        OpenFBXExportMenu,
        dope_sheet_import_button,
        dope_sheet_export_button,
        MetricsOutputWindow,
        ExportMetricsButton
    ]

def register():
    for x in classes_to_register: register_class(x)  
    bpy.types.Scene.my_tool = PointerProperty(type=GenerationProperties)
    bpy.types.DOPESHEET_MT_context_menu.append(draw_buttons_in_dope_sheet)

# end function register

def unregister():
    for x in classes_to_register: unregister_class(x)
    del bpy.types.Scene.my_tool
    bpy.types.DOPESHEET_MT_context_menu.remove(draw_buttons_in_dope_sheet)
    
# end function unregister
    
if __name__ == "__main__":
    register()
    select_ai_model(0)
    selected_device = "cpu"

# end main
