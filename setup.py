
import setuptools
from distutils.core import setup

setup(
      name             = 'electricityLoadForecasting',
      version          = '0.1.dev0',
      packages         = setuptools.find_packages(),
      scripts          = ['scripts/mainPreparation.py', './scripts/mainForecasting.py'],
      maintainer       = 'Ben',
      license          = 'MIT License',
      long_description = open('README.txt').read(),
      python_requires  = ">= 3.6",
      install_requires = [
                          'astral',
                          'chardet',
                          'datetime',
                          'h5py',
                          'ipdb',
                          'matplotlib',
                          'numpy',
                          'pandas',
                          #'python-spams',
                          'pytz',
                          #'rpy2',
                          'scikit-learn',
                          'scipy',
                          'seaborn',
                          #'sklearn-contrib-py-earth',
                          'termcolor',
                          'tzlocal',
                          'unidecode',
                          #'xgboost',
                          ],
      )