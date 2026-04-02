from setuptools import setup, find_packages
import os

# Get the directory where setup.py is located
here = os.path.abspath(os.path.dirname(__file__))

# Dynamically find all asset files
assets_dir = os.path.join(here, 'assistive_gym', 'envs', 'assets')
data_files = [os.path.join(here, 'assistive_gym', 'config.ini')]

if os.path.exists(assets_dir):
    for root, dirs, files in os.walk(assets_dir):
        for fn in files:
            # Get path relative to the package folder
            full_path = os.path.join(root, fn)
            data_files.append(os.path.relpath(full_path, os.path.join(here, 'assistive_gym')))

setup(
    name='fmvp-pb',  # Updated name
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'gym>=0.2.3',
        'pybullet @ git+https://github.com/Zackory/bullet3.git@pybullet_3_0_9#egg=pybullet',
        'numpy',
        'keras==2.3.0',
        'tensorflow',
        'h5py==2.10.0',
        'smplx',
        'trimesh',
        'ray[rllib]',
        'numpngw',
        'tensorflow-probability>=0.7.0',
        'matplotlib',
        'screeninfo>=0.6.1'
    ],
    url='https://github.com/hyhalexis/fmvp_pb',
    author='Alexis',
    license='MIT',
    description='Cleaned release for fmvp_pb simulation',
    long_description_content_type="text/markdown",
    package_data={'assistive_gym': data_files},
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)