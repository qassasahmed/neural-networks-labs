import percv2

x, T = percv2.read_file("../lab-3/datasets/data.csv")
p1 = percv2.Perceptron(len(x[0]), 1)
p1.run(x, T)
print(p1.get_activation_info())
