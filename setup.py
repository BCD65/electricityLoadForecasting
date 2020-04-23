
import setuptools
from distutils.core import setup

setup(
      name             = 'electricityLoadForecasting',
      version          = '0.1.dev0',
      packages         = setuptools.find_packages(),
      scripts          = ['scripts/mainPreparation.py', 'scripts/mainForecasting.py'],
      maintainer       = 'Ben',
      license          = 'MIT License',
      long_description = open('README.txt').read(),
      python_requires  = ">= 3.6",
      install_requires = [
                          'astral==1.2',
                          'chardet',
                          'datetime',
                          'h5py',
                          'ipdb',
                          'matplotlib',
                          'numpy',
                          'pandas',
                          'pytz',
                          'scikit-learn',
                          'scipy',
                          'seaborn',
                          'termcolor',
                          'tzlocal',
                          'unidecode',
                          #'sklearn-contrib-py-earth',
                          #'spams',
                          #'rpy2',
                          #'xgboost',
                          ],
      )