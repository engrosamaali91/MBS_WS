from setuptools import find_packages, setup

package_name = 'zed2i_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='osama',
    maintainer_email='osama@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = zed2i_pkg.my_node:main',
            'zed2ii_node = zed2i_pkg.zed2ii_node:main',

        ],
    },
)
