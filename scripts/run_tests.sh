#!/bin/bash

if [[ -n "$AI_ADDON_BLENDER_EXEC" ]]; then
    cd ../src/test || exit 1
    pytest --blender-executable "$AI_ADDON_BLENDER_EXEC"
    echo "Testing done."
    cd ../../scripts || exit 1
else
    echo "Environment variable AI_ADDON_BLENDER_EXEC is not set."
fi
