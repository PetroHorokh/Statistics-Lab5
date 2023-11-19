import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def random_variable_output(const, data):
    return list(data + const * (random.uniform(0, 1) - 1 / 2))


def correlational_field(y, x1, x2):
    plt.scatter(x1, y, label='x1', color='green', marker='o')
    plt.scatter(x2, y, label='x2', color='red', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Dependence between x1/x2 and y')
    plt.legend(loc='lower right')
    plt.show()


def parameters_estimation(y, x1, x2):
    x = np.column_stack((np.ones_like(x1), x1, x2))

    beta = np.linalg.lstsq(x, y, rcond=None)[0]

    print("Parameter estimates:")
    print("b0:", beta[0])
    print("b1:", beta[1])
    print("b2:", beta[2], "\n")

    return {
        "b0": beta[0],
        "b1": beta[1],
        "b2": beta[2],
    }


def build_regressive_model(parameters):
    print(f"y = {parameters['b0']} + {parameters['b1']} * x_1 + {parameters['b2']} + x_2", '\n')


def f_test(y, x1, x2):
    x = np.column_stack((np.ones_like(x1), x1, x2))

    beta = np.linalg.lstsq(x, y, rcond=None)[0]

    y_hat = np.dot(x, beta)

    ssr = np.sum((y_hat - np.mean(y)) ** 2)
    sse = np.sum((y - y_hat) ** 2)

    f_statistic = ssr / sse * 15 / 2

    print(f"F-статистика: {f_statistic}")
    critical_value = stats.f.ppf(1 - 0.05, 2, 15)

    if f_statistic > critical_value:
        print("The model is adequate.\n")
    else:
        print("The model may not be adequate.\n")


def leavings(y, x1, x2):
    x = np.column_stack((np.ones_like(x1), x1, x2))

    beta = np.linalg.lstsq(x, y, rcond=None)[0]

    y_pred = x @ beta

    residual_matrix = np.reshape(y - y_pred, (len(y), 1))

    return residual_matrix[0]


def residuals_variance_estimation(leaving):
    return leaving.T @ leaving / 16


def model_parameters_variances_estimation(residual, x1, x2):
    x = np.column_stack((np.ones_like(x1), x1, x2))

    operation_result = residual * np.linalg.inv(x.T @ x)

    result = []

    for i in range(3):
        result.append(operation_result[i][i])

    return result


def t_test(residuals, parameters):
    t0 = parameters['b0'] / residuals[0]
    t1 = parameters['b1'] / residuals[1]
    t2 = parameters['b2'] / residuals[2]

    if 2 * (1 - stats.t.cdf(abs(t0), 15)) < 0.05:
        print("Parameter b0 is significant")
    else:
        print("Parameter b0 is not significant.")

    print(
        f"Confidence interval for b0: [{parameters['b0'] + stats.t.ppf(0.05 / 2, 15) * residuals[0]};{parameters['b0'] - stats.t.ppf(0.05 / 2, 15) * residuals[0]}]")

    if 2 * (1 - stats.t.cdf(abs(t1), 15)) < 0.05:
        print("Parameter b1 is significant")
    else:
        print("Parameter b1 is not significant.")

    print(
        f"Confidence interval for b1: [{parameters['b1'] + stats.t.ppf(0.05 / 2, 15) * residuals[1]};{parameters['b1'] - stats.t.ppf(0.05 / 2, 15) * residuals[1]}]")

    if 2 * (1 - stats.t.cdf(abs(t0), 15)) < 0.05:
        print("Parameter b2 is significant")
    else:
        print("Parameter b2 is not significant.")

    print(
        f"Confidence interval for b2: [{parameters['b2'] + stats.t.ppf(0.05 / 2, 15) * residuals[2]};{parameters['b2'] - stats.t.ppf(0.05 / 2, 15) * residuals[2]}]\n")


def prediction(y, x1, x2):
    new_x1, new_x2 = 2.5, 150.3

    x = np.column_stack((np.ones_like(x1), x1, x2))

    beta = np.linalg.lstsq(x, y, rcond=None)[0]

    new_observation = np.array([1, new_x1, new_x2])
    predicted_value = np.dot(new_observation, beta)

    residuals = y - np.dot(x, beta)
    mse = np.sum(residuals ** 2) / (len(y) - x.shape[1])
    cov_matrix = mse * np.linalg.inv(np.dot(x.T, x))
    se_predicted_value = np.sqrt(np.dot(new_observation, np.dot(cov_matrix, new_observation)))

    df = len(y) - x.shape[1]

    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha / 2, df)

    lower_bound = predicted_value - t_critical * se_predicted_value
    upper_bound = predicted_value + t_critical * se_predicted_value

    print(f"x_1 = {new_x1}")
    print(f"x_2 = {new_x2}")
    print(f"Predicted value: {predicted_value:.4f}")
    print(f"Confidence interval: ({lower_bound:.4f}, {upper_bound:.4f})")
