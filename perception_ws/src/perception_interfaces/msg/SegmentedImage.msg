# 语义分割结果
std_msgs/Header header
sensor_msgs/Image segmented_image
sensor_msgs/Image color_map
string[] class_names
int32[] class_ids
string segmentation_method  # "cv" or "dnn" 