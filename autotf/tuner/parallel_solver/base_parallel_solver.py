import os

import time
import errno
import logging
import json
import numpy as np
import dill
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


def evaluate_func(params):
    objective_func, x = params
    objective_func = dill.loads(objective_func)
    start_time = time.time()
    return_val = objective_func(x)
    time_overhead = time.time() - start_time
    return return_val, time_overhead, x


class BaseParallelSolver(object):

    def __init__(self, acquisition_func=None, model=None, lower=None, upper=None,
                 maximize_func=None, task=None, save_dir=None, objective_func=None,
                 n_restarts=1, init_points=3, rng=None, initial_design=None, n_workers=4):
        """
        Base class which specifies the interface for solvers. Derive from
        this class if you implement your own solver.

        Parameters
        ----------
        acquisition_func: BaseAcquisitionFunction Object
            The acquisition function which will be maximized.
        model: ModelObject
            Model (i.e. GaussianProcess, RandomForest) that models our current
            believe of the objective function.
        task: TaskObject
            Task object that contains the objective function and additional
            meta information such as the lower and upper bound of the search
            space.
        maximize_func: MaximizerObject
            Optimization method that is used to maximize the acquisition
            function
        save_dir: String
            Output path
        """
        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(100000))
        else:
            self.rng = rng

        self.X = None
        self.y = None
        self.lower = lower
        self.upper = upper
        self.time_func_evals = []
        self.time_overhead = []
        self.output_path = None
        self.start_time = time.time()  # timestamp when the program starts

        self.incumbents = []
        self.incumbents_values = []
        self.runtime = []  # timestamp from the program starts
        self.model = model
        self.acquisition_func = acquisition_func
        self.maximize_func = maximize_func
        self.objective_func = dill.dumps(objective_func)
        self.task = task
        self.save_dir = save_dir
        self.trial_statistics = []
        self.n_restarts = n_restarts
        self.init_points = init_points
        self.initial_design = initial_design

        self.num_workers = n_workers
        self.pool = ProcessPoolExecutor(max_workers=n_workers)

        if self.save_dir is not None:
            self.create_save_dir()

    def initialize(self):
        # Initial design

        start_time_overhead = time.time()
        init = self.initial_design(self.lower,
                                   self.upper,
                                   self.init_points*self.num_workers,
                                   rng=self.rng)
        time_overhead = (time.time() - start_time_overhead) / self.init_points*self.num_workers

        # Run all init config
        for i, x in enumerate(init):
            logger.info("Evaluate: %s", x)
            self.trial_statistics.append(self.pool.submit(evaluate_func, (self.objective_func, x)))
            self.time_overhead.append(time_overhead)

        # Wait all initial trials finish
        self.wait_tasks_finish()

        self.collect()

    def wait_tasks_finish(self):
        all_completed = False
        while not all_completed:
            all_completed = True
            for trial in self.trial_statistics:
                if not trial.done():
                    all_completed = False
                    time.sleep(0.1)
                    break

    def collect(self):
        # Get the evaluation statistics
        trial_id = 0
        while trial_id < len(self.trial_statistics):
            trial = self.trial_statistics[trial_id]
            trial_id += 1
            if trial.done():
                new_y, time_taken, new_x = trial.result()
                self.time_func_evals.append(time_taken)
                logger.info("Configuration achieved a performance of %f ", new_y)
                logger.info("Evaluation of this configuration took %f seconds", self.time_func_evals[-1])

                # Extend the data
                if self.X is None:
                    X = []; y = []
                    X.append(new_x); y.append(new_y)
                    self.X = np.array(X); self.y = np.array(y)
                else:
                    self.X = np.append(self.X, new_x[None, :], axis=0)
                    self.y = np.append(self.y, new_y)

                # Estimate incumbent
                best_idx = np.argmin(self.y)
                incumbent = self.X[best_idx]
                incumbent_value = self.y[best_idx]

                self.incumbents.append(incumbent.tolist())
                self.incumbents_values.append(incumbent_value)
                logger.info("Current incumbent %s with estimated performance %f",
                            str(incumbent), incumbent_value)

                self.runtime.append(time.time() - self.start_time)

                if self.output_path is not None:
                    self.save_output(str(new_x))

                self.trial_statistics.remove(trial)
                trial_id -= 1

    def create_save_dir(self):
        """
        Creates the save directory to store the runs
        """
        try:
            os.makedirs(self.save_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self.output_file = open(os.path.join(self.save_dir, 'results.csv'), 'w')
        self.output_file_json = open(os.path.join(self.save_dir, 'results.json'), 'w')
        self.csv_writer = None
        self.json_writer = None

    def get_observations(self):
        return self.X, self.Y

    def get_model(self):
        if self.model is None:
            logger.info("No model trained yet!")
        return self.model

    def run(self, num_iterations=10, X=None, y=None):
        """
        The main optimization loop

        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,)
            Function values of the already evaluated points

        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            (Estimated) function value of the incumbent
        """
        pass

    def choose_next(self, X=None, y=None):
        """
        Suggests a new point to evaluate.

        Parameters
        ----------
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        y: np.ndarray(N,)
            Function values of the already evaluated points

        Returns
        -------
        np.ndarray(1,D)
            Suggested point
        """
        pass

    def get_json_data(self, it):
        """
        Json getter function

        :return: dict() object
        """

        jsonData = {"optimization_overhead":self.time_overhead[it], "runtime": time.time() - self.time_start,
                    "incumbent": self.incumbent.tolist(),
                    "incumbent_fval": self.incumbent_value.tolist(),
                    "time_func_eval": self.time_func_eval[it],
                    "iteration": it}
        return jsonData

    def save_json(self, it, **kwargs):
        """
        Saves meta information of an iteration in a Json file.
        """
        base_solver_data =self.get_json_data(it)
        base_model_data = self.model.get_json_data()
        base_task_data = self.task.get_json_data()
        base_acquisition_data = self.acquisition_func.get_json_data()

        data = {'Solver': base_solver_data,
                'Model': base_model_data,
                'Task': base_task_data,
                'Acquisiton': base_acquisition_data
                }

        json.dump(data, self.output_file_json)
        self.output_file_json.write('\n')  # Json more readable. Drop it?

    def save_output(self, it):

        data = dict()
        data["optimization_overhead"] = self.time_overhead[it]
        data["runtime"] = self.runtime[it]
        data["incumbent"] = self.incumbents[it]
        data["incumbents_value"] = self.incumbents_values[it]
        data["time_func_eval"] = self.time_func_evals[it]
        data["iteration"] = it

        json.dump(data, open(os.path.join(self.output_path, "tuner_iter_%d.json" % it), "w"))
