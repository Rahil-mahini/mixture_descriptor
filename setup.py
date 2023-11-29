from setuptools import setup, find_packages

    

setup(
    name='Mixture-Descriptors',
    version='1.0.0',
    packages=find_packages( where = "src"),
    description='Development of Numerical Features/Descriptors to Describe Complex Materials for Machine Learning Modeling',
    long_description_content_type='text/markdown',
    author='Rahil Ashtari',
    author_email='Rahil.Ashtarimahini@ndsu.edu',
    url='https://github.com/Rahil-mahini/SOFTX-D',
    license = " GPL-3.0",

    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: SCIENCE/RESEARCH',
        'Topic :: CIENTIFIC/ENGINEERING :: CHEMISTRY',
        'License :: OSI Approved :: GNU General Public License, version 3 ',
        'Programming Language :: Python :: 3.10',
 
      
    ],
    python_requires='>=3.10',
    install_requires = [
        # List your project's dependencies here.
        # They will be installed by pip when your project is installed.
        'numpy', 
        'pandas',



    ]
    
)