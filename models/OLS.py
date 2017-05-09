import numpy as np
from scipy.optimize import least_squares

# Implement the OLS class here
class OLS:
    def __init__(self, x, y):
        """Method for independent and dependent variable.
        Parameters
        ----------
        x : ndarray
            The first independent parameter.
        y : ndarray
            The second dependent parameter.
        """

        self.x = x
        self.y = y

    def _cost_function(self):
        def fun(j, x, y):
            return ((j * x) - y)*((j * x) - y)

    def fit(self):
        x0 = np.repeat(1, len(self.x))
        res = least_squares(self._cost_function(), x0, args=(self.x, self.y))
        self.beta = res.x


