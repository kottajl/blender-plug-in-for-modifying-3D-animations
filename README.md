# Blender software plug-in for modifying 3D character models’ animations

## Project Structure

- **/interface** -> contains the main interface that models must implement

- **/src** -> contains source files, including main_addon_file.py, which must be loaded as a script

- **/models** -> contains pre-prepared models that can be used. Each model has its own folder, with a file named model_name.py located directly within the folder, which implements our interface 

- **/sample_bvh_files** -> sample motion capture files

- **/lib** -> contains binary files required for the plugin


## Installation

To install the plug-in:

1. Download the file ‘***ai_motion_bridge-<version>.zip***’ from this repository.

2. Open **Blender**.

3. Go to ‘**Preferences**’.

4. On the list on the left, select ‘**Add-ons**’.

5. Press the arrow in the top right corner and select ‘***Install from disc...***’.

6. Select the downloaded zip file.

After selecting the file and confirming the selection with the ‘***Install from disc***’ button, the plug-in may take some time to download the relevant dependencies and files from external repositories. 

This proces may take a while. Progress can be checked from the terminal.


## How to use

### Opening the tool
To start using the tool:

1. Go to the Layout tab (on the top bar).

2. Display the panels with additional options to the right of the view (using the ‘**N**’ key).

3. In the bar that appears, select the ‘***Addon***’ tab. You should see a panel with various options and a ‘**Generate frames**’ button at the bottom.


It is also possible to use the plugin via the context menu on the timeline. To do this:

1. Make sure that you are in the Layout tab and that you can see the expanded timeline view at the bottom of the window. 
    - To expand the timeline view, hover the cursor over the space between the scene and the top edge of the timeline bar. Then press the **left mouse button** and move up the screen as desired.

2. Place the cursor inside the timeline view and open the context menu (using the **right mouse button**). You should see additional items such as '**Generate frames**', '**Generation options**', etc.

### Generating animation

1. While in the Layout tab, select the desired Blender object.

2. In the top left corner, change the mode to ‘**Pose mode**’.

3. Select all bones of the skeleton (most conveniently using the **left mouse button** on the scene).

4. Open the tool options
    - If you are using the context menu, select ‘**Generation options**’.

5. Select the AI model from the list and then configure the model and generation options.

6. Select the start and end frames of the animation to be generated
    - If you are using the context menu, move the cursor outside the options window and, holding down the ‘***Shift***’ key, select both keyframes on the timeline.

7. Press the ‘**Generate frames**’ button.
