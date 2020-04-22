================================================================

In the folder dataPreparation, the main.py script downloads, formats and saves public data from MeteoFrance and RTE websites. The load data and the weather data are saved in two separate multiindexed pandas dataframes. The objective is to have a public dataset that can then be used as the input of forecasting algorithms. In the dataPreparation/config.py file, the user can choose between the national load and the administrative regions, the years to download and the folder where the data is saved.

================================================================

The package can be used with the following environment :
# Create new env
conda create -n py37 python=3.7 anaconda conda h5py termcolor matplotlib scipy scikit-learn pytz ipdb python-spams pandas spyder seaborn chardet unidecode
# Activate
conda activate py37
# Install requirements from specific channels
conda install -c conda-forge python-spams ipdb xgboost sklearn-contrib-py-earth astral tzlocal
conda install -c r rpy2
conda install -c omnia termcolor
# Additionally, the mgcv library should be available in R to use the GAM

================================================================
