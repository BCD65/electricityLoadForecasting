
================================================================

# It can be installed with :
cd ~/Downloads
git clone https://github.com/BCD65/electricityLoadForecasting.git
cd electricityLoadForecasting
conda create --name elec python=3.7 pip
conda activate elec
pip install -e .
# [Optional] However, the packages xgboost, spams and rpy2 have to be installed separately :
conda install -c conda-forge python-spams xgboost sklearn-contrib-py-earth ipdb 
# [Optional] Also, the mgcv library should be available in R if the GAM are to be tested.
conda install -c r rpy2 r-mgcv

================================================================

The script mainPreprocessing.py downloads, formats and saves public data from MeteoFrance and RTE websites. The load data and the weather data are saved in two separate multiindexed pandas dataframes. The objective is to have a public dataset that can then be used as the input of forecasting algorithms. In preprocessing/eCO2mix/config.py, you can choose either the national database 'France' or the 'administrative_regions'.

Running the script mainForecasting.py launches the whole learning process with the dataset selected in forecasting/hyperparameters/choose_dataset.py : selecting the inputs according to the chosen hyperparameters, computing the features and optimizing the coefficients.

================================================================

