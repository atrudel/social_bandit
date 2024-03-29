from setuptools import setup, find_packages


setup(
    name='socialbandit',
    version='0.1.0',
    description='Code for the Social Bandit experiments with RNNs',
    author='Amric Trudel',
    url="https://github.com/atrudel/social_bandit",
    python_requires='~=3.10',
    install_requires=[
        "torch",
        "matplotlib",
        "pandas",
        "numpy",
        "scipy",
        "tqdm",
        "torchmetrics",
        "torchtyping",
        "tensorboard",
        "jupyter",
        "seaborn"
    ],
    tests_require=[
        'pytest'
    ],
    packages=find_packages()
)