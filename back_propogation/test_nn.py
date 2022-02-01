from neural_net import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons


dataset = make_moons(n_samples=10, noise = 0.1)

for i,(p,l) in enumerate(zip(dataset[0],dataset[1])):
    plt.scatter(p[0],p[1],color = "blue" if l else "red")

nn = Neural_Net(2,[
    Dense(3),
    Dense(2),
    Dense(1)
    ]
)
print(nn)

step = Step()
cost_function = Quadratic_cost()
accuracy = []
current_cost = 0
cost = []

batch_size = 5
num_epochs = 2

print(dataset[0][:5].T.shape)

for epoch in range(num_epochs):
    print("Epoch: ", epoch)
    correct = 0
    for i in range(0,len(dataset[0]),batch_size):
        print(f"{i}:")
        start = i
        stop = start+batch_size
        print(f"{start}->{stop}")
        batch = dataset[0][start:stop]
        print(batch)
        p = np.reshape(p,(2,1))

        out = nn.evaluate(p)
        nn.back_propagate(l)
        current_cost += cost_function(out,l)

