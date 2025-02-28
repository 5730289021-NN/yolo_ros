from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['yolo_ros'],
    package_dir={'': 'src'},
    scripts=['scripts/yolo_node.py', 'scripts/debug_node.py', 'scripts/tracking_node.py', 'scripts/detect_3d_node.py']
)

setup(**d)
