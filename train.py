#!/usr/bin/env python
import sys
import time

def readdataset(filename):
    result = []
    with open(filename) as input:
        for line in input:
            line = line.strip()
            if line == '':
                continue
            numbers = line.split()
            X = list( float(num) for num in numbers[1:-1] )
            Y = float(numbers[-1])
            result.append([X, Y])
            # print('Y: {Y:.2f} X: {X}'.format(X=X, Y=Y))
    return result


def linear_calculate(X, params):
    assert(len(X) + 1 == len(params))
    i = 0
    H = params[len(X)]
    while i < len(X):
        H += X[i] * params[i]
        i += 1
    return H

def loss(X, Y, params):
    H = linear_calculate(X, params)
    # print('H: {H:.2f}, Y: {Y:.2f}'.format(H=H, Y=Y))
    return ((H - Y) ** 2)

def cost(dataset, params):
    J = 0
    for data in dataset:
        J += loss(data[0], data[1], params)
    return J / len(dataset)

def train(dataset, params, rate = 0.0001):
    dparams = [0] * len(params)
    
    for data in dataset:
        X, Y = data
        tmp = (linear_calculate(X, params) - Y)
        k = 0
        while k < len(X):
            dparams[k] += tmp * X[k]
            k += 1
        dparams[k] += tmp
    
    # real deltaparams = -rate/m * dparams
    # now update params
    k = 0
    while k < len(params):
        params[k] += -(rate/len(dataset)) * dparams[k]
        k += 1

def dump_message(dataset, params, J, n):
    print('train times: {}, cost: {}'.format(n, J))
    k = 0
    while k < len(params):
        print('k[{k}]: {val:.4f}'.format(k=k, val=params[k]))
        k += 1
    print('prediction and real: ')
    for data in dataset:
        X, Y = data
        H = linear_calculate(X, params)
        print('H: {H:.4f} Y: {Y:.4f}'.format(H=H, Y=Y))



train_set = readdataset(sys.argv[1])
X_len = len(train_set[0][0])
params = [0] * (X_len + 1)

last_J = None
n = 0
while True:
    J = cost(train_set, params)
    if last_J is None or int(last_J) != int(J):
        if last_J is None:
            dump_message(train_set, params, J, n)
        else:
            dump_message(train_set, params, last_J, n-1) 
    last_J = J
    train(train_set, params, 0.0003)
    n += 1
    # time.sleep(0.1)