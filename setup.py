"""
Setup script for PageScan.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pagescan',
    version='1.0.0',
    author='PageScan',
    description='Book page dewarping and perspective correction application',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'PyQt6>=6.4.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-image>=0.21.0',
        'Pillow>=10.0.0',
    ],
    extras_require={
        'raw': ['rawpy>=0.18.0'],
        'exif': ['piexif>=1.1.3'],
        'all': [
            'rawpy>=0.18.0',
            'piexif>=1.1.3',
            'imageio>=2.31.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'pagescan=pagescan.__main__:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
    ],
)
