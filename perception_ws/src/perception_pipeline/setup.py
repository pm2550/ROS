from setuptools import setup
import os
from glob import glob

package_name = 'perception_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='perception_dev',
    maintainer_email='dev@perception.com',
    description='RPI5 TPU based perception pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_input = perception_pipeline.sensor_input_node:main',
            'image_preprocessor = perception_pipeline.image_preprocessor_node:main',
            'depth_estimator = perception_pipeline.depth_estimator_node:main',
            'object_detector = perception_pipeline.object_detector_node:main',
            'semantic_segmentor = perception_pipeline.semantic_segmentor_node:main',
            'perception_manager = perception_pipeline.perception_manager_node:main',
            'timing_analyzer = perception_pipeline.timing_analyzer_node:main',
        ],
    },
) 