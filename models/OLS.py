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

    @staticmethod
    def _cost_function(beta, x, y):
        for row in x:
            for m in row[0:11]:
                return ((beta * m) - y) * ((beta * m) - y)


    def fit(self):
        x0 = np.repeat(1, len(self.x))
        res = least_squares(OLS._cost_function, x0, args=(self.x, self.y))
        self.beta = res.x
