from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='gaussianlda',
    version='0.1',
    description='Implementation of Gaussian LDA topic model, with efficiency tricks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/markgw/gaussianlda',
    author='Mark Granroth-Wilding',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Text Processing',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='gaussian lda topic-model machine-learning',
    package_dir={'': 'src'},
    python_requires='>=3.2',
    install_requires=['numpy', 'scipy', 'Cython', 'choldate', 'progressbar'],
    project_urls={
        'Based on': 'https://github.com/rajarshd/Gaussian_LDA',
        'Funding': 'https://www.newseye.eu/',
    },
)
