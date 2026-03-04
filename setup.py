from setuptools import find_packages, setup

setup(
    name='induction_visualization',
    packages=find_packages(include=['induction_visualization']),
    version='0.0.1',
    description='Visualization of induction response to veer',
    author='Storm Mata',
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)