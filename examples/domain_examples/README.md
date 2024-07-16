# Electricity Load Modelling and Forecasting

This tutorial displays an easy-to-follow notebook for novice machine learning (ML) and deep learning (DL) practitioners within the energy buildings space. More specifically, this tutorial demonstrates the use of time-series modelling and forecasting using the Short-Term Electricity Load Forecasting (Panama case study) dataset and introduces to users the benefits of feature engineering (data preprocessing) and the ease of large models that may otherwise seem inapproachable to beginners.

Within this introductory notebook, NeuroMANCER can now bring more advanced state-of-the-art (SOTA) models into the existing library, increasing the breadth of models available to end-users as well as broadening prospective new users.

Specifically, the Transformer model is incorporated into the blocks.py module, allowing for end-users to load an out-of-the-box network using the blocks interface on NeuroMANCER.

Complimentary to the Energy Load Forecasting Notebook using a multi-layered perceptron (MLP), this notebook offers a tutorial for energy-load modelling using historical weather features as inputs to a vanilla Transformer model. This tutorial also allows for the capability of forecasting future timesteps when future weather forecasts are inaccessible or too uncertain.

## Forecasting  
While this notebook does not display the use of forecasting future timesteps without weather inputs, after training the model displayed in the notebook for 175 epochs, one can achieve relatively good results for forecasting future timesteps:

![image.png](Prediction_0.png)
![image.png](Prediction_3.png)
![image.png](Prediction_14.png)

