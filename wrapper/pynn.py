
from random import randint

#from libSFCNN import NN

class ParallelNN:
    def __init__(self, config = None, dump = None, n_workers = 4):
        self.master = NN()
        self.n_workers = n_workers
        self.workers = []
        for i in range(n_workers):
            self.workers.append(NN())
        if config:
            self.master.FromConfig(config)
            print "Master OK"
            self.master_neurons = self.master.GetNeurons()
            for worker in self.workers:
                worker.FromConfig(config)
                worker.SetNerons(self.master_neurons)
        elif dump:
            self.master.FromDump(dump)
            for worker in self.workers:
                worker.FromDump(dump)
        else:
            raise Exception()

    def Train(self, X, Y, iters):
        assert len(X) == len(Y)
        n = len(X)
        deltas = []
        for it in xrange(iters):
            if it % n_workers == 0 and deltas:
                sum_deltas = deltas[0]
                for i in range(len(deltas)):
                    cur_deltas = deltas[i]
                    for j in range(len(cur_deltas)):
                        self.master_neurons[j] += cur_deltas[j]
                for worker in self.workers:
                    worker.SetNerons(self.master_neurons)
                deltas = []


            worker = self.workers[it % self.n_workers]

            train_i = randint(0, n-1)

            deltas.append(worker.BackProp(X[train_i], Y[train_i]))



if __name__ == "__main__":
    pnn = ParallelNN(dump = "dump_digits.txt")

    from sklearn.datasets import load_digits
    import numpy

    digits = load_digits()
    for x, y_raw in zip(digits.data, digits.target):
        y = numpy.zeros(10)
        y[y_raw] = 1

        X.append(x)
        Y.append(y)

    pnn.Train(X, Y, 1000000)