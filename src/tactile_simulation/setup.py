from glob import glob
from setuptools import setup

setup(
    name='tactile_simulation', version='0.1.0', packages=['tactile_simulation'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/tactile_simulation']),
        ('share/tactile_simulation', ['package.xml', 'requirements.txt']),
        ('share/tactile_simulation/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'], tests_require=['pytest'], zip_safe=True,
    maintainer='Tact Foundry', maintainer_email='jeffreyzhang2012@users.noreply.github.com',
    description='Tactile robot physics and camera playground', license='Proprietary',
    entry_points={'console_scripts': [
        'playground = tactile_simulation.node:main',
        'scene_controls = tactile_simulation.controls:main',
    ]},
)
