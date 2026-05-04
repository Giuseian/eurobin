# Deactivate left dagana
ros2 service call /cartesian/dagana_1_base/set_active cartesian_interface_ros/srv/SetTaskActive "{activation_state: false}"

# Deactivate right dagana
ros2 service call /cartesian/dagana_2_base/set_active cartesian_interface_ros/srv/SetTaskActive "{activation_state: false}"

# Activate left dagana
ros2 service call /cartesian/dagana_1_base/set_active cartesian_interface_ros/srv/SetTaskActive "{activation_state: true}"

# Activate right dagana
ros2 service call /cartesian/dagana_2_base/set_active cartesian_interface_ros/srv/SetTaskActive "{activation_state: true}"