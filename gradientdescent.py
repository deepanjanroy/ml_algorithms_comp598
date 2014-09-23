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

    show_progress_modulo = 100
    fancy_progress = False


    def __init__(self, gradient, inp_dim, objective, alpha=1e-11, max_iter=9999999, epsilon=1e-8):
        self.gradient = gradient
        self.inp_dim = inp_dim
        self.objective = objective
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.alpha_diminishing_factor = 1
        self.alpha_diminishing_frequency = 10  # diminish after every n iteration
        self.repeats = 1

    @staticmethod
    def report_iterations(num_iters, fancy=fancy_progress):
        if fancy:
            sys.stdout.write("Iterations complete: %d   \r" % (num_iters,))
            sys.stdout.flush()
        else:
            print "Iterations complete: %d   \r" % (num_iters,)

    def run_once(self, max_iter = None):
        print "Running gradient descent... This may take a while"
        next = np.random.rand(self.inp_dim)
        alpha = self.alpha

        if max_iter is None:
            max_iter = self.max_iter

        for iter_count in xrange(max_iter):
            prev = next
            next = prev - alpha * self.gradient(prev)
            if np.linalg.norm(next - prev) < self.epsilon:
                break

            if iter_count % self.show_progress_modulo == 0:
                self.report_iterations(iter_count)
                import warnings

            if iter_count % self.alpha_diminishing_frequency == 1:
                alpha = alpha * self.alpha_diminishing_factor

        print "Gradient descent last error: ", np.linalg.norm(next - prev)
        print "Gradient norm at the end: ", np.linalg.norm(self.gradient(next))
        objective = self.objective(next)
        return objective, next

    def run(self):
        min_val = np.inf
        min_weight = None

        for i in xrange(self.repeats):
            obj, weight = self.run_once()
            print "Try {0}: Received value of objective function {1}".format(i, obj)
            if obj < min_val:
                min_val = obj
                min_weight = weight

        print "Min value of obj: ", min_val

        return min_weight






