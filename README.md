# Predict-Used-Toyota-Corolla-Car-Price-base-on-Specification
The goal is to predict the price of a used Toyota Corolla based on its specifications.

Consider the data on used cars (ToyotaCorolla.csv) with 1436 records and details on 38 attributes,
including Price, Age, KM, HP, and other specifications. 

a. Fit a neural network model to the data. Use a single hidden layer with 2 nodes.
 
*Use predictors Age_08_04, KM, Fuel_Type, HP, Automatic, Doors, Quarterly_Tax,
Mfr_Guarantee, Guarantee_Period, Airco, Automatic_airco, CD_Player,
Powered_Windows, Sport_Model, and Tow_Bar.

*Remember to first scale the numerical predictor and outcome variables to a 0–1
scale and convert categorical predictors to dummies.

b. Record the root mean square (RMS) error for the training data and the validation data. Repeat
the process, changing the number of hidden layers and nodes to {single layer with
5 nodes}, {two layers, 5 nodes in each layer}.
 
*What happens to the RMS error for the training data as the number of layers
and nodes increases?

*What happens to the RMS error for the validation data?

*Comment on the appropriate number of layers and nodes for this application.
