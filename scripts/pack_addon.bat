@echo off

if defined AI_ADDON_BLENDER_EXEC (
    cd ..
    "%AI_ADDON_BLENDER_EXEC%" --command extension build
    echo Packaging done.
    cd scripts
) else (
    echo Environment variable AI_ADDON_BLENDER_EXEC is not set.
)
