from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='rtmpose3d',
    version='1.0.0',
    description='RTMPose3D: Real-Time Multi-Person 3D Pose Estimation - Simple PyTorch interface',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Bahadir Arac',
    author_email='',
    url='https://github.com/mutedeparture/rtmpose3d',
    packages=find_packages(exclude=['examples', 'tests']),
    package_data={
        'rtmpose3d': [
            'configs/*.py',
            'models/*.py',
        ],
    },
    include_package_data=True,
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
        'mmpose>=1.0.0',
        'mmdet>=3.0.0',
        'mmcv>=2.0.0',
        'mmengine>=0.7.0',
        'tqdm>=4.60.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='pose estimation, 3d pose, computer vision, pytorch',
)
