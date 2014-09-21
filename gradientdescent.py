import numpy as np
import sys

class GradientDescentRunner(object):
    """What does this thing do? It runs gradient descent.
    What is the input of gradient descent?
    We are trying to find the minimum value of an objective function
    Need w, initial objective function
    Need an expression for the gradient with respect to w
    Need hyperparameter \alpha, or find a way to choose it
    need meta parameters like the maximum number of iterations



    How does it do it?
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

            ### DEBUG
            if iter_count < 5:
                print next

        return next
