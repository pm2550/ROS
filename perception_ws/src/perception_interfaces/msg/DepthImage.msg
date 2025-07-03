# 深度图像信息
std_msgs/Header header
sensor_msgs/Image depth_image
sensor_msgs/Image disparity_image
string estimation_method  # "stereo_cv" or "stereo_cnn"
float32 baseline
float32 focal_length
geometry_msgs/Point[] point_cloud 