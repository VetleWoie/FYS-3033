from neural_net import Neural_Net
from layers.basic_layers import Input
from layers.dense import Dense
from cost.cost_functions import Quadratic_cost
from activation.activation_functions import Step
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons

def draw_lines(ax, nn, layer):
    weight = nn.layers[layer].weights
    bias = nn.layers[layer].bias
    y = []
    x = np.linspace(-10,10,2)
    for w,b in zip(weight, bias):
        y.append((-w[0]/(w[1]+0.000001)*x) - b/(w[1]+0.000001))
    for l in y:
        ax.plot(x,l)

def draw_points(ax, set, labels, amount):
    for p,l in zip(set[:amount], labels[:amount]):
        ax.scatter(p[0],p[1],color = "blue" if l else "red")

dataset = make_moons(n_samples=1000, noise = 0.1)
nn = Neural_Net(2,[
    Input(2),
    Dense(3),
    Dense(1)
    ]
)
print(nn)

fig, ax = plt.subplots(1,2)
ax[0].set_xlim([-2, 3])
ax[0].set_ylim([-2, 3])


step = Step()
cost_function = Quadratic_cost()
accuracy = []
current_cost = 0
cost = []

batch_size = 5
num_epochs = 1000

cf = Quadratic_cost()
cost = []
epochs = []

nn.fit(num_epochs, batch_size,training=(dataset[0].T,np.expand_dims(dataset[1],0)))

# for epoch in range(num_epochs):
#     # ax[0].cla()
#     # ax[1].cla()
#     # ax[0].set_xlim([-2, 3])
#     # ax[0].set_ylim([-2, 3])

#     print("Epoch: ", epoch)
#     correct = 0
#     out = nn.evaluate(dataset[0].T)

#     cost.append(cf(out, np.expand_dims(dataset[1],0))[0,0]/len(dataset[0]))
#     epochs.append(epoch)

#     # draw_points(ax[0], dataset[0], dataset[1], amount = 100)
#     # draw_lines(ax[0], nn, 1)
#     # plt.pause(0.1)
#     for i in range(0,len(dataset[0]),batch_size):
#         start = i
#         stop = start+batch_size
#         batch = dataset[0][start:stop]
#         batch_label = np.expand_dims(dataset[1][start:stop],0)
#         out = nn.evaluate(batch.T)
#         nn.back_propagate(batch_label)
#         nn.update_weights()
# ax[0].cla()
# ax[1].cla()
# ax[0].set_xlim([-2, 3])
# ax[0].set_ylim([-2, 3])
# ax[1].plot(epochs, cost)
draw_points(ax[0], dataset[0], dataset[1], amount = 100)
draw_lines(ax[0], nn, 1)
plt.show()