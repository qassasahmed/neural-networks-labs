import perc as p

if __name__ == '__main__':
    x = [[1, 1, 1, 1, 1],
         [-1, 1, -1, -1, 1],
         [1, 1, 1, -1, 1],
         [1, -1, -1, 1, 1]]

    per = p.Perceptron(5)
    print(per.run(x, [1, 1, -1, -1]))
