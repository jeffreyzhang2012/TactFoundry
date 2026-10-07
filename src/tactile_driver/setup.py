from glob import glob
from setuptools import setup

setup(
    name='tactile_driver', version='0.1.0', packages=['tactile_driver'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/tactile_driver']),
        ('share/tactile_driver', ['package.xml']),
        ('share/tactile_driver/launch', glob('launch/*.launch.py')),
        ('share/tactile_driver/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'], tests_require=['pytest'], zip_safe=True,
    maintainer='Tact Foundry maintainers', maintainer_email='maintainers@example.com',
    description='Serial tactile hardware driver', license='Proprietary',
    entry_points={'console_scripts': ['serial_driver = tactile_driver.node:main']},
)
