from setuptools import setup, find_packages

setup(
    name="idt-eyetracking",
    version="1.0.0",
    description="I-DT (Dispersion-Threshold) fixation detection for 2-D gaze data.",
    author="Joash Ye",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.5",
        "matplotlib>=3.6",
        "openpyxl>=3.1",
        "xlrd>=2.0",
    ],
    extras_require={"dev": ["pytest>=7.0"]},
    entry_points={
        "console_scripts": [
            "idt-analyse=src.cli:main",
        ],
    },
)
