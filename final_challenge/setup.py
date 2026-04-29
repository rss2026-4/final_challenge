import glob
import os
from setuptools import find_packages
from setuptools import setup

package_name = 'final_challenge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch', 'part_a'),
         glob.glob('launch/part_a/*')),
        (os.path.join('share', package_name, 'launch', 'part_b'),
         glob.glob('launch/part_b/*')),
        (os.path.join('share', package_name, 'config', 'part_a'),
         glob.glob('config/part_a/*')),
        (os.path.join('share', package_name, 'config', 'part_b'),
         glob.glob('config/part_b/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='todo@todo.todo',
    description='RSS 2026 Final Challenge ROS2 Package (Snail Race + Boating School)',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector = final_challenge.part_a.lane_detector:main',
            'lane_follower = final_challenge.part_a.lane_follower:main',

            #part b 
            'astar_planner = final_challenge.part_b.astar_planner:main',
            'homography_transform = final_challenge.part_b.homography_transform:main',
            'map_inflator = final_challenge.part_b.map_inflator:main',
            'parking_controller = final_challenge.part_b.parking_controller:main',
            'parking_detector = final_challenge.part_b.parking_detector:main',
            'particle_filter = final_challenge.part_b.particle_filter:main',
            'safety_controller = final_challenge.part_b.safety_controller:main',
            'state_machine = final_challenge.part_b.state_machine:main',
            'traffic_light = final_challenge.part_b.traffic_light:main',
            'trajectory_follower = final_challenge.part_b.trajectory_follower:main',
            'yolo_annotator = final_challenge.part_b.yolo_annotator:main',
        ],
    },
)
