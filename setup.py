from setuptools import setup, find_packages

setup(
    name='gym-analysis-toolkit',  # Replace with your package name
    version='0.1.0',
    description='A toolkit to make episodic analysis of RL environments faster and easier',
    author='Nigel Swenson',
    packages=find_packages(),  # Automatically finds packages with __init__.py
    include_package_data=True,
    install_requires=[],  # List any dependencies here
    python_requires='>=3.6',
)