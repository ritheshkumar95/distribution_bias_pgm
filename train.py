import numpy as np
import torch
import modules
import losses
import matplotlib.pyplot as plt


OUTPUT_DIST = 'StandardGaussian'
criterion = eval('losses.'+OUTPUT_DIST+'NLL')
network = eval('modules.'+OUTPUT_DIST+'RNN()').cuda()
opt = torch.optim.Adam(network.parameters())

train_loss = []
for i in xrange(25000):
    train_loss.append(
        modules.process_batch(network, criterion, opt)
        )
    if i % 1000 == 0:
        val_loss = []
        for j in xrange(1000):
            val_loss.append(
                modules.process_batch(network, criterion, None, False)
                )
        print "Validation loss: ", np.asarray(val_loss).mean()
        modules.plot_distributions(network)

    if i % 100 == 0:
        print "Iter {} train_loss: {}".format(i, np.asarray(train_loss)[-100:].mean())

np.save('std_gaussian', train_loss)


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'valid')


def plot_curves(curves, names):
    fig = plt.figure()
    colors = ['red', 'green', 'blue', 'orange', 'violet']
    handles = []
    for i, curve in enumerate(curves):
        curve = movingaverage(curve, 100)
        handles += [plt.plot(curve, c=colors[i], label=names[i])]
    plt.legend()
    fig.savefig('likelihood_curves.png')
