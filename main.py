import func
import pandas as pd

data = pd.read_csv('data.csv')

y = func.random_variable_output(1.5, data['y'])
x1 = func.random_variable_output(2.5, data['x1'])
x2 = func.random_variable_output(10, data['x2'])

func.correlational_field(y, x1, x2)

parameters = func.parameters_estimation(y, x1, x2)

func.build_regressive_model(parameters)

residuals_variance_estimation = func.residuals_variance_estimation(func.leavings(y, x1, x2))

residuals = func.model_parameters_variances_estimation(residuals_variance_estimation, x1, x2)

func.f_test(y, x1, x2)

func.t_test(residuals, parameters)

func.prediction(y, x1, x2)
