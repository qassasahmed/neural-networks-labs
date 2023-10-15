import numpy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

poly_model_d3 = numpy.poly1d(numpy.polyfit(x, y, 3))
poly_model_d5 = numpy.poly1d(numpy.polyfit(x, y, 5))
poly_model_d10 = numpy.poly1d(numpy.polyfit(x, y, 10))
# try polynomial regression of 20th degree!!


x_coordinates = numpy.linspace(1, 22, 100)

slope, intercept, r, p, std_err = stats.linregress(x, y)


def y_coordinate(x):
    return slope * x + intercept


linear_model = list(map(y_coordinate, x))

print(f"R-squared-linear: {r2_score(y, linear_model):.6f}")
print(f"R-squared-3: {r2_score(y, poly_model_d3(x)):.6f}")
print(f"R-squared-5: {r2_score(y, poly_model_d5(x)):.6f}")
print(f"R-squared-10: {r2_score(y, poly_model_d10(x)):.6f}")

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.scatter(x, y, label="Original date")
plt.plot(x, linear_model, label="Linear Fitting")
plt.plot(x_coordinates, poly_model_d3(x_coordinates), label="Poly-3 Fitting")
plt.plot(x_coordinates, poly_model_d5(x_coordinates), label="Poly-5 Fitting")
plt.plot(x_coordinates, poly_model_d10(x_coordinates), label="Poly-10 Fitting")
plt.legend(loc='upper center')
plt.show()
