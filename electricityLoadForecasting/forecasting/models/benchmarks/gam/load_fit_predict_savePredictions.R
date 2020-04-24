# Fetch command line arguments
myargs <- commandArgs(trailingOnly = TRUE)

# Get formula
string_formula = myargs[1]
cat(gsub(')+', ')\n', sub('~', '~\n ', string_formula)))

# Load data
path_Rfiles = myargs[2]
load(paste(path_Rfiles, "temp_dat_for_r_train.gzip", sep = ''))
load(paste(path_Rfiles, "temp_dat_for_r_test.gzip",  sep = ''))

# Import
library(mgcv)

# Fit
model = gam(as.formula(string_formula), data = data_train)

# Predict
predictions_train = predict(model, data_train)
predictions_test  = predict(model, data_test )

# Save Predictions 
write.table(predictions_train, file = paste(path_Rfiles, "predictions_from_r_train.gzip", sep = ''))
write.table(predictions_test,  file = paste(path_Rfiles, "predictions_from_r_test.gzip",  sep = ''))

#return 1