@echo off

if defined AI_ADDON_BLENDER_EXEC (
    cd ..\src\test
    pytest --blender-executable %AI_ADDON_BLENDER_EXEC% 
    echo Testing done.
    cd ..\..\scripts
) else (
    echo Environment variable AI_ADDON_BLENDER_EXEC is not set.
)
