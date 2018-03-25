import time
import logging

from tuner.initial_design.init_random_uniform import init_random_uniform
from tuner.parallel_solver.base_parallel_solver import BaseParallelSolver
from tuner.parallel_solver.base_parallel_solver import evaluate_func

logger = logging.getLogger(__name__)


class SyncParallelSolver(BaseParallelSolver):

    def __init__(self, objective_func, lower, upper,
                 acquisition_func, model, maximize_func,
                 initial_design=init_random_uniform,
                 initial_points=3,
                 output_path=None,
                 train_interval=1,
                 n_restarts=1,
                 n_workers=4,
                 rng=None):
        """
        function description.

        Parameters
        ----------
        acquisition_func: BaseAcquisitionFunctionObject
            The acquisition function which will be maximized.
        """
        BaseParallelSolver.__init__(self, acquisition_func=acquisition_func, maximize_func=maximize_func, model=model,
                                    init_points=initial_points, n_restarts=n_restarts, rng=rng,
                                    initial_design=initial_design, lower=lower, upper=upper,
                                    objective_func=objective_func, n_workers=n_workers)

        self.time_func_evals = []
        self.time_overhead = []
        self.train_interval = train_interval
        self.output_path = output_path
        self.time_start = None

    def run(self, num_iterations=10, X=None, y=None):
        """
        The main parallel optimization loop

        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,1)
            Function values of the already evaluated points

        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            (Estimated) function value of the incumbent
        """
        # Save the time where we start the parallel optimization procedure
        self.time_start = time.time()

        if X is None and y is None:
            self.initialize()
        else:
            self.X = X
            self.y = y

        # Main synchronous parallel optimization loop
        for it in range(self.init_points, num_iterations):
            logger.info("Start iteration %d ... ", it)
            self.trial_statistics.clear()

            start_time = time.time()

            if it % self.train_interval == 0:
                do_optimize = True
            else:
                do_optimize = False

            # Choose next point to evaluate
            new_x_batch = self.choose_next_batch(self.X, self.y, do_optimize)

            self.time_overhead.append(time.time() - start_time)
            logger.info("Optimization overhead was %f seconds", self.time_overhead[-1])
            logger.info("Next candidate %s", str(new_x_batch))

            for new_x in new_x_batch:
                self.trial_statistics.append(self.pool.submit(evaluate_func, (self.objective_func, new_x)))

            # wait a batch of trials finish
            self.wait_tasks_finish()

            # get the evaluation statistics
            self.collect()

        logger.info("Return %s as incumbent with error %f ",
                    self.incumbents[-1], self.incumbents_values[-1])

        return self.incumbents[-1], self.incumbents_values[-1]

    def choose_next_batch(self, X=None, y=None, do_optimize=True):
        """
        Suggests a batch of points to evaluate.

        Parameters
        ----------
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,1)
            Function values of the already evaluated points
        do_optimize: bool
            If true the hyperparameters of the model are
            optimized before the acquisition function is
            maximized.
        Returns
        -------
        np.ndarray(batch,D)
            Suggested points
        """

        if X is None and y is None:
            x = self.initial_design(self.lower, self.upper, 1, rng=self.rng)[0, :]

        elif X.shape[0] == 1:
            # We need at least 2 data points to train a GP
            x = self.initial_design(self.lower, self.upper, 1, rng=self.rng)[0, :]

        else:
            try:
                logger.info("Train model...")
                t = time.time()
                self.model.train(X, y, do_optimize=do_optimize)
                logger.info("Time to train the model: %f", (time.time() - t))
            except:
                logger.error("Model could not be trained!")
                raise
            self.acquisition_func.update(self.model)

            logger.info("Maximize acquisition function...")
            t = time.time()
            x = self.maximize_func.maximize(batch_size=self.num_workers)

            logger.info("Time to maximize the acquisition function: %f", (time.time() - t))

        return x
