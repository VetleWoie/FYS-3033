
from turtle import color
from neural_net import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
# np.set_printoptions(precision=2)

trainingSet = [
    np.array([[0],[0]]),
    np.array([[1],[0]]),
    np.array([[0],[1]]),
    np.array([[1],[1]]),
]

trainingLabel = [
    np.array([[0]]),
    np.array([[1]]),
    np.array([[1]]),
    np.array([[0]]),
]

nn = Neural_Net(2,
    [
        Dense(2,learning_rate=1,weights=np.array([[0,1],[1,0]]), bias=np.array([[0], [0]])), 
        Dense(1,learning_rate=1,weights=np.array([[1,1]]),bias=np.array([[0]]))
    ])
nn = Neural_Net(2,
[
    Dense(2,learning_rate=1),#,learning_rate=1,weights=np.array([[0,1],[1,0]]), bias=np.array([[0], [0]])), 
    Dense(1,learning_rate=1)#,learning_rate=1,weights=np.array([[1,1]]),bias=np.array([[0]]))
])
print(nn)

fig, ax = plt.subplots(1,2)
for l,(i,j) in zip(trainingLabel,trainingSet):
    ax[0].scatter(i,j, color = "green" if l else "orange")

w = nn.layers[0].weights
b = nn.layers[0].bias

x = np.linspace(-10,10,2)
y0 = (-w[0,0]/(w[0,1]+ 0.000001)*x) - (b[0]/(w[0,1]+0.000001))
y1 = (-w[1,0]/(w[1,1]+ 0.000001)*x) - (b[1]/(w[1,1]+ 0.000001))
ax[0].plot(x,y0, color = "red")
ax[0].plot(x,y1, color = "blue")
ax[0].set_xlim([-0.5, 1.5])
ax[0].set_ylim([-0.5, 1.5])
plt.pause(0.1)

p = np.array([[1,1,0,0],[1,0,1,0]])
label = np.array([[0,1,1,0]])

cf = Quadratic_cost()
cost = []
for i in range(4000):
    out = nn.evaluate(p)
    cost.append(cf(out,label)[0][0]/4)
    nn.back_propagate(label)
    nn.update_weights()
    if(i%100 == 0):
        ax[0].cla()
        ax[1].cla()

        ax[1].plot(np.arange(len(cost)), cost)
        for l,(i,j) in zip(trainingLabel,trainingSet):
            ax[0].scatter(i,j, color = "green" if l else "orange")

        w = nn.layers[0].weights
        b = nn.layers[0].bias

        y0 = (-w[0,0]/(w[0,1]+0.000001)*x) - b[0]/(w[0,1]+0.000001)
        y1 = (-w[1,0]/(w[1,1]+ 0.000001)*x) - b[1]/(w[1,1]+ 0.000001)
        ax[0].plot(x,y0, color = "red")
        ax[0].plot(x,y1, color = "blue")
        ax[0].set_xlim([-0.5, 1.5])
        ax[0].set_ylim([-0.5, 1.5])
        plt.pause(0.1)
    

print(p[:,0].reshape(2,1))
print("Evaluating:")
out = nn.evaluate(p)
print()
print("Final output:")
print(out)
plt.show()
