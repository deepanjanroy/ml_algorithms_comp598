import numpy as np
import sys

class GradientDescentRunner(object):
    """Runs gradient descent.

    Constructor params:
        gradient: fun inp -> float
            function to calculate gradient from input
        inp_dim: int
            dimension of input variable

    Methods:
        run_once: numpy.array of shape (inp_dim,)
            Runs gradient descent and returns a vector of dimension inp_dim
    """

    def __init__(self, gradient, inp_dim, alpha=1e-11, max_iter=9999999, epsilon=1e-5):
        self.gradient = gradient
        self.inp_dim = inp_dim
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon

    @staticmethod
    def report_iterations(num_iters):
        sys.stdout.write("Iterations complete: %d   \r" % (num_iters,))
        sys.stdout.flush()

    def run_once(self):
        print "Running gradient descent... This may take a while"
        # next = np.random.rand(self.inp_dim)
        next = np.ones(self.inp_dim)
        for iter_count in xrange(self.max_iter):
            prev = next
            next = prev - self.alpha * self.gradient(prev)
            if np.linalg.norm(next - prev) < self.epsilon:
                break
            if iter_count % 1e3 == 0:
                self.report_iterations(iter_count)

        return next
