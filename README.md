# MBS_WS
Step 1
[Install zed sdk in your system](https://www.stereolabs.com/docs/installation/linux) and then [ros2 humble](https://docs.ros.org/en/humble/Installation.html)
Step 2: clone this repository
```
git pull https://github.com/engrosamaali91/MBS_WS
```
Step 3: Built it
```
cd /MBS_WS && colcon build --symlink-install 
```
Step 4. source it 
```
source install/setup.bash
```

Step 5: Run zed node to evoke the nodes and topics 
```
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

Step 6:
```
ros2 run zed2i_pkg zed2ii_node
```

