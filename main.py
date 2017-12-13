# Cryptarithmetic puzzle solver implemented via a genetic algorithm

# Author:   Roman Smirnov
# Created:  9/12/2017

from genetic import *
import sys  # used to get command line arguments

if __name__ == '__main__':

    if len(sys.argv) != 6:
        print(" wrong number of arguments ", len(sys.argv))
    elif sys.argv[2] not in "+-*/" or sys.argv[4] != "=":
        print(" no such command", " usage: python main.py a + b = c ")
    elif sys.argv[2] == "+":
        print(sys.argv[1:])
        print(CryptarithmeticSolver(sys.argv[1], sys.argv[3], sys.argv[5], 50, 1, addition_evaluation).solve())
    elif sys.argv[2] == "-":
        print(sys.argv[1:])
    elif sys.argv[2] == "*":
        print(sys.argv[1:])
    elif sys.argv[2] == "/":
        print(sys.argv[1:])
    else:
        print(" ShouldNotHappenError ")
