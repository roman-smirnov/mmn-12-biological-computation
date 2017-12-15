""" this module is used to bulk run solvers, generate statistics and graph the results """

# graph plotting libs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# numerical computation lib
import numpy as np
# solver
from genetic import CryptarithmeticSolver

# Graph axis annotations
X_LABEL = "Generation"
Y_AVG_LABEL = "Average Evaluation"
Y_MAX_LABEL = "Maximum Evaluation"
Y_MIN_LABEL = "Minimum Evaluation"

# description strings
RUNS_TXT = "Number of Runs = "
POP_SIZE_TXT = "Population Size = "

# cosmetics
LINE_WIDTH = 0.5
PLOT_PIXEL_DENSITY = 150
COLOR_MAP_NAME = 'winter'


def graph_statistics(avg_evals, max_evals, min_evals, max_gens, pop_size, num_runs, desc):
    """
    :param avg_evals: a list of lists containing the average evaluation in each generation of a run
    :param max_evals: a list of lists containing the maxima evaluation in each generation of a run
    :param min_evals: a list of lists containing the minima evaluation in each generation of a run
    :param max_gens: the number of generations in the longest running solver
    :param pop_size: solver genotype population size
    :param num_runs: the number of times the solver was run
    :param desc: the problem description as a string
    """

    # get a color map for plot line colors
    cmap = cm.get_cmap(COLOR_MAP_NAME)

    # init the plot and set description
    plt.figure(PLOT_PIXEL_DENSITY)
    plt.suptitle(desc + "\n" + RUNS_TXT + str(num_runs) + "\n" + POP_SIZE_TXT + str(pop_size), fontsize=10)

    # plot the averages graph
    plt.subplot(221)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_AVG_LABEL)
    for i in range(num_runs):
        y = np.array(avg_evals[i])
        x = np.arange(len(y))
        plt.plot(x, y, color=cmap(i / float(num_runs)), linewidth=LINE_WIDTH, aa=True)

    # plot the maxima graph
    plt.subplot(222)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_MAX_LABEL)
    for i in range(num_runs):
        y = np.array(max_evals[i])
        x = np.arange(len(y))
        plt.plot(x, y, color=cmap(i / float(num_runs)), linewidth=LINE_WIDTH, aa=True)

    # plot the minima graph
    plt.subplot(223)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_MIN_LABEL)
    for i in range(num_runs):
        y = np.array(min_evals[i])
        x = np.arange(len(y))
        plt.plot(x, y, color=cmap(i / float(num_runs)), linewidth=LINE_WIDTH, aa=True)

    # adjust the subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    # display the plot
    plt.show()


def generate_statistics(a, b, c, pop_size, time_lim, eval_func, num_runs):
    """
    :param a: first problem term
    :param b: second problem term
    :param c: result term
    :param pop_size: the size of genotype population to generate
    :param time_lim: the maximum run time limit of a solver
    :param eval_func: the solver evaluation function
    :param num_runs: how many times to run the solver
    :return: the stats
    """
    # lists of lists of generated values
    avg_evals = []
    max_evals = []
    min_evals = []
    # number of generations in the longest running solver
    max_gens = 0

    # run the solvers and collect the stats
    for r in range(num_runs):
        solver = CryptarithmeticSolver(a, b, c, pop_size, time_lim, eval_func)
        solver.solve()
        av, ma, mi, g = solver.retrieve_stats__()
        avg_evals.append(list(av.values()))
        max_evals.append(list(ma.values()))
        min_evals.append(list(mi.values()))
        max_gens = max(max_gens, g)

    return avg_evals, max_evals, min_evals, max_gens, pop_size, num_runs
