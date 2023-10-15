import numpy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

poly_model_d3 = numpy.poly1d(numpy.polyfit(x, y, 3))
x_coordinates = numpy.linspace(1, 16, 100)


def y_coordinate(x):
    return slope * x + intercept


model = list(map(y_coordinate, x))
print(f"R-squared with SciPy: {r**2:.6f}")
print(f"R-squared: {r2_score(y, model):.6f}")
print(f"R-squared: {r2_score(y, poly_model_d3(x)):.6f}")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.scatter(x, y, label="Original date")
plt.plot(x, model, color="red", label="Fitting line")
# plt.plot(x_coordinates, poly_model_d3(x_coordinates))
plt.legend(loc='upper center')
plt.show()
