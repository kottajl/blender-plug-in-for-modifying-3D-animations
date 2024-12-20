import bpy
from .src import main_addon_file


def register():
    main_addon_file.register()

def unregister():
    main_addon_file.unregister()


if __name__ == "__main__":
    
    if not bpy.app.online_access:
        print("Error: Online access is disabled. Please enable it in Preferences.")
        exit(-1)

    register()
