
================================================================

#It can be installed with
cd ~/Downloads
git clone git@github.com:BCD65/electricityLoadForecasting.git
cd electricityLoadForecasting
conda create --name elec
conda activate elec
pip install -e .
# However, the packages xgboost, spams and rpy2 have to be installed separately
conda install -c conda-forge python-spams xgboost sklearn-contrib-py-earth ipdb 
conda install -c r rpy2
# Also, the mgcv library should be available in R to use the GAM

================================================================

The script mainPreparation.py formats and saves public data from MeteoFrance and RTE websites. The load data and the weather data are saved in two separate multiindexed pandas dataframes. The objective is to have a public dataset that can then be used as the input of forecasting algorithms. In dataPreparation/eCO2mix/config.py, you can choose either the country database 'France' or the 'administrative_regions'.

Running the script mainForecasting.py launches the whole learning process : selecting the inputs according to the chosen hyperparameters, computing the features and optimizing the coefficients.

================================================================

