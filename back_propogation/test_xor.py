
from neural_net import Neural_Net
from layers.basic_layers import Input
from cost.cost_functions import Quadratic_cost
import numpy as np
from layers.dense import Dense
from matplotlib import pyplot as plt

#Training set and labels
p = np.array([[1,1,0,0],[1,0,1,0]])
label = np.array([[0,1,1,0]])

#Setup nerual network
nn = Neural_Net(2,
[
    Dense(2,learning_rate=1),
    Dense(1,learning_rate=1)
])

#Setup plots for animation
fig, ax = plt.subplots(1,2)
#plot training set
for i,l in enumerate(label[0]):
    ax[0].scatter(p[:,i][0],p[:,i][1], color = "green" if l else "orange")

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
        for i,l in enumerate(label[0]):
            ax[0].scatter(p[:,i][0],p[:,i][1], color = "green" if l else "orange")
        w = nn.layers[0].weights
        b = nn.layers[0].bias

        y0 = (-w[0,0]/(w[0,1]+0.000001)*x) - b[0]/(w[0,1]+0.000001)
        y1 = (-w[1,0]/(w[1,1]+ 0.000001)*x) - b[1]/(w[1,1]+ 0.000001)
        ax[0].plot(x,y0, color = "red")
        ax[0].plot(x,y1, color = "blue")
        ax[0].set_xlim([-0.5, 1.5])
        ax[0].set_ylim([-0.5, 1.5])
        plt.pause(0.1)
plt.show()