from setuptools import setup, find_packages

setup(
    name='spiceist',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scanpy',
        'torch',
        'torch_geometric',
        'scipy',
        'scikit-learn',
        'geopandas',
        'libpysal',
        'networkx',
        'scib',
        'pyarrow',
    ],
    author='Sungwoo Bae',
    author_email='sungwoo.bae@portrai.io',
    description='Graph autoencoder framework that integrates subcellular transcript distribution patterns with cell-level gene expression profiles for enhanced cell clustering in imaging-based ST',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/portrai-io/SPICEiST',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
) 