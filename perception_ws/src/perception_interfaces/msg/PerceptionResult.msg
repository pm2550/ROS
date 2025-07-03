# 完整的感知流水线结果
std_msgs/Header header
sensor_msgs/Image left_image
sensor_msgs/Image right_image
DepthImage depth_result
DetectedObjects detection_result
SegmentedImage segmentation_result
TimingInfo timing_info
string pipeline_mode  # "dnn_tpu" or "cv_cpu" 