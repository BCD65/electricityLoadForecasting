
================================================================

# It can be installed with :
cd ~/Downloads
git clone https://github.com/BCD65/electricityLoadForecasting.git
cd electricityLoadForecasting
conda create --name elec python=3.7 pip
conda activate elec
pip install -e .
# [Optional] However, the packages xgboost and spams have to be installed separately :
conda install -c conda-forge python-spams xgboost sklearn-contrib-py-earth ipdb openblas
# [Optional] Also, rpy2 in Python and the mgcv library in R should be available if the GAM are to be tested.
pip install rpy2==3.3.1
# conda install -c r r r-mgcv

================================================================

The script preprocessing_main.py downloads, formats and saves public data from MeteoFrance and RTE websites. The load data and the weather data are saved in two separate multiindexed pandas dataframes. The objective is to have a public dataset that can then be used as the input of forecasting algorithms. In preprocessing/eCO2mix/config.py, you can choose either the national database 'France' or the 'administrative_regions'.

Running the script main_forecasting.py launches the whole learning process with the dataset selected in forecasting/hyperparameters/choose_dataset.py : selecting the inputs according to the chosen hyperparameters, computing the features and optimizing the coefficients.

================================================================

