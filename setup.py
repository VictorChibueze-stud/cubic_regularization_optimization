from setuptools import setup, find_packages

setup(
    name='cubic_regularization',
    version='0.1.0',
    description='Modular implementation of cubic regularization and advanced optimizers for logistic regression',
    author='Victor Okoroafor, Benjamin Duvor',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'cubic-regularization=src.main:main',
        ],
    },
    python_requires='>=3.8',
) 