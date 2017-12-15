# Cryptarithmetic puzzle solver implemented via a genetic algorithm

# Author:   Roman Smirnov
# Created:  10/12/2017

# import the solver and its evaluation functions
from genetic import CryptarithmeticSolver, add_eval, sub_eval, mult_eval, div_eval
import sys  # used to get command line arguments
# import the statistics generation and graphing functionality
from statistics import generate_statistics, graph_statistics


# default run constants
POP_SIZE = 50
TIME_LIMIT = 1

# constants for the statistics and graph generation demo
NUM_OF_RUNS = 10
A = "SEND"
B = "MORE"
C = "MONEY"

# constants for CLI input parsing
CLI_NUM_OF_STAT_ARGS = 2
CLI_STAT_CMD_STR_ARG = "stats"
CLI_STAT_CMD_ARG_POS = 1
NUM_OF_PROB_ARGS = 6
PROB_TYPE_ARG = 2
FIRST_TERM_ARG = 1
SECOND_TERM_ARG = 3
RESULT_TERM_ARG = 5


def demo_stats():
    """ run the solver a bunch of times and graph averages and extrema """
    desc = A + " + " + B + " = " + C
    stats = generate_statistics(A, B, C, POP_SIZE, TIME_LIMIT, add_eval, NUM_OF_RUNS)
    graph_statistics(*stats, desc)


if __name__ == '__main__':
    """ handle command line input """

    if len(sys.argv) == CLI_NUM_OF_STAT_ARGS and sys.argv[CLI_STAT_CMD_ARG_POS] == CLI_STAT_CMD_STR_ARG:
        # stats generation command - run the solver multiple times, collect statistics and output a plot
        demo_stats()
    elif len(sys.argv) != NUM_OF_PROB_ARGS:
        print(" wrong number of arguments ", len(sys.argv))
    elif sys.argv[PROB_TYPE_ARG] not in "+-*/" or sys.argv[4] != "=":
        print(" no such command", " usage: python main.py a + b = c ")
    elif sys.argv[PROB_TYPE_ARG] == "+":
        # it's an addition problem
        print(sys.argv[FIRST_TERM_ARG:])
        print(CryptarithmeticSolver(sys.argv[FIRST_TERM_ARG], sys.argv[SECOND_TERM_ARG], sys.argv[RESULT_TERM_ARG], POP_SIZE, TIME_LIMIT, add_eval).solve())
    elif sys.argv[PROB_TYPE_ARG] == "-":
        # it's a subtraction problem
        print(sys.argv[FIRST_TERM_ARG:])
        print(CryptarithmeticSolver(sys.argv[FIRST_TERM_ARG], sys.argv[SECOND_TERM_ARG], sys.argv[RESULT_TERM_ARG], POP_SIZE, TIME_LIMIT, sub_eval).solve())
    elif sys.argv[PROB_TYPE_ARG] == "*":
        # it's a multiplication problem
        print(sys.argv[FIRST_TERM_ARG:])
        print(CryptarithmeticSolver(sys.argv[FIRST_TERM_ARG], sys.argv[SECOND_TERM_ARG], sys.argv[RESULT_TERM_ARG], POP_SIZE, TIME_LIMIT, mult_eval).solve())
    elif sys.argv[PROB_TYPE_ARG] == "/":
        # it's a division problem
        print(sys.argv[FIRST_TERM_ARG:])
        print(CryptarithmeticSolver(sys.argv[FIRST_TERM_ARG], sys.argv[SECOND_TERM_ARG], sys.argv[RESULT_TERM_ARG], POP_SIZE, TIME_LIMIT, div_eval).solve())
    else:
        print("a ShouldNotHappenError has occurred")
