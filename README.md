# Blender software plug-in for modifying 3D character models’ animations

## Installation

### Blender version 4.2.0 and above

To install the plug-in:

1. Download **ZIP** file from this repository.

2. Open **Blender**.

3. Go to **Edit** (in top left corner) **> Preferences**.

4. On the list from left side, select **Add-ons**.

5. Press the arrow in the top right corner and select **Install from Disc**.

6. Select downloaded **ZIP** file.

7. Confirm the selection with the **Install from Disc** button. 

### Older Blender versions

Currently there is no option to install addon from **ZIP** file. To install the plug-in:

1. Download whole repository.

2. Open **Blender**.

3. Go to **Scripting** (in top section)

4. Click **Open** button in the middle.

5. Select **main_addon_file.py** from **src** folder in repository.

6. Confirm the selection with the **Open Text** button. 

7. Press **Run Script** (with arrow icon) button.

### While installing

The plug-in may take some time to download the relevant dependencies and files from external repositories. This proces may take a few or more minutes depending on network speed. Progress can be checked from the **terminal**.

## How to use

### Opening the tool

To start using the tool:

1. Go to the **Layout** tab (on the top bar).

2. Display the panels with additional options to the right of the view (using the **N** key).

3. In the bar that appears, select the **AI Animation Bridge** tab. You should see a panel with various options and a **Generate frames** button at the bottom.

It is also possible to use the plugin via the context menu on the timeline. To do this:

1. Make sure that you are in the **Layout** tab and that you can see the expanded timeline view at the bottom of the window. 
    - To expand the timeline view, hover the cursor over the space between the scene and the top edge of the timeline bar. Then press the **left mouse button** and move up the screen as desired.

2. Place the cursor inside the timeline view and open the context menu (using the **right mouse button**). You should see additional items such as **Generate frames**, **Generation options**, etc.

### Generating animation

1. While in the **Layout** tab, select the desired 3D object.

2. In the top left corner, change the mode to **Pose mode**.

3. Select all bones of the skeleton (most conveniently using the **left mouse button** on the scene).

4. Open the tool options
    - If you are using the context menu, select **Generation options**.

5. Select the AI model from the list and then configure the model and generation options.

6. Select the start and end frames of the animation to be generated
    - If you are using the context menu, move the cursor outside the options window and, holding down the **Shift** key, select both keyframes on the timeline.

7. Press the **Generate frames** button.

## Project Structure

- **/interface** -> contains the main interface that models must implement

- **/src** -> contains source files, including main_addon_file.py

- **/models** -> contains pre-prepared models that can be used. Each model has its own folder, with a file named model_name.py located directly within the folder, which implements our interface 

- **/sample_bvh_files** -> sample motion capture files

- **/lib** -> contains binary files required for the plugin


## Running tests
1. Install the plugin and run it in Blender at least once to install the necessary libraries

2. Install **pytest** and **pytest-blender** libraries in your Python installation

3. Run command **pytest --blender-executable <blender-executable-path>** in the **src/test folder**