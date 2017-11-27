import numpy as np
import random as rd


def sig(x, derivative=False):
    if derivative is True:
        return sig(x) * (1 - sig(x))
    return 1 / (1 + np.exp(-x))


np.random.seed(1)
weight0 = 2 * np.random.random((10, 10)) - 1
weight1 = 2 * np.random.random((10, 10)) - 1
rd.seed(1)


def move():
    return rd.randint(0, 9)


num_list = [move() for a in range(10)]

#learning
for count in range(60000):
    # right
    right = move()
    out_true = np.asarray([0 for a in range(10)])
    out_true[right] = 1

    # input
    num = np.asarray(num_list)
    layer0 = num
    layer1 = sig(np.dot(layer0, weight0))
    layer2 = sig(np.dot(layer1, weight1))

    l2_error = out_true - layer2
    l2_delta = l2_error * sig(layer2, derivative=True)

    l1_error = l2_delta.dot(weight1.T)
    l1_delta = l1_error * sig(layer1, derivative=True)

    # weight2 += layer2.T.dot(l3_delta)
    weight1 += layer1.T.dot(l2_delta)
    weight0 += layer0.T.dot(l1_delta)

    num_list.pop(0)
    num_list.append(right)
    #print(l2_error.mean())

#testing
for count in range(1000):
    # right
    right = move()
    out_true = np.asarray([0 for a in range(10)])
    out_true[right] = 1

    # input
    num = np.asarray(num_list)
    layer0 = num
    layer1 = sig(np.dot(layer0, weight0))
    layer2 = sig(np.dot(layer1, weight1))
    print(layer2.argmax())

    num_list.pop(0)
    num_list.append(right)
    print(right)