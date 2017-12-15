"""
    The module contains classes which implement genetic algorithms
"""

import numpy as np
from abc import ABC, abstractmethod
import time


#####################################################################
#    Genetic Solver Abstract Template
#####################################################################


class GeneticSolver(ABC):
    """
        Provides a template for implementing various genetic algorithm solvers.
        Designed for inheritance - subclasses are required to implement all abstract methods.
    """

    def __init__(self, problem, population_size, run_time_limit_secs):
        """ initialize the generated genotype population size and solver run time limit """
        self.pop_size = population_size
        self.time_limit = run_time_limit_secs
        self.generation_counter = 0
        self.problem = problem
        # DO NOT CHANGE ORDER
        self.encoding = self.__encoding__()
        self.encoded_evaluation = self.__encode_evaluation__()

    @abstractmethod
    def __encoding__(self):
        """ generate an encoding of the problem into a representation more suitable for computation """
        pass

    @abstractmethod
    def __decode__(self, solution):
        """ return a decoded string representation of a solution"""
        pass

    @abstractmethod
    def __encode_evaluation__(self):
        """ use the generated encoding to encode the problem evaluation"""
        pass

    @abstractmethod
    def __seed__(self):
        """ generate a seed population of genotypes"""
        pass

    @abstractmethod
    def __mutate__(self, population):
        """ mutate the population """
        pass

    @abstractmethod
    def __fitness__(self):
        """ returns a the reproductive fitness of genotypes based on the evaluation """
        pass

    @abstractmethod
    def __evaluation__(self, population):
        """ generate the population evaluation """
        pass

    @abstractmethod
    def __reproduce__(self, population):
        """ produce the next generation population"""
        pass

    @abstractmethod
    def __is_solved__(self, evaluation):
        """ check if the problem is solved """
        pass

    @abstractmethod
    def solve(self):
        """ run the solver and return the solution if found """
        pass


#####################################################################
#    Genetic Cryptarithmetic Solver Implementation
#####################################################################

# noinspection PyMethodOverriding
class CryptarithmeticSolver(GeneticSolver):
    """ an evolutionary solver for various cryptarithmetic problems """

    # value to which the genotype should evaluate to be a valid solution
    SOLUTION_EVALUATION = 0
    # number of possible variables(0-9)
    VARIABLE_VALUE_RANGE = 10
    # denoted an unused value in the encoding
    UNUSED_VALUE_SYMBOL = '±'
    # the axis along which to permute the genotype population
    ROW_AXIS = 1

    # once per how many generations to generate a fresh set of random genotypes
    NICHE_INTERVAL = 10

    def __init__(self, first_term, second_term, result_term, population_size, time_limit, eval_func):
        """
        :param first_term: first problem term
        :param second_term: second  problem term
        :param result_term: result problem term
        :param population_size: how many genotype samples to generate
        :param time_limit: the maximum run time of the solver
        :param eval_func: the function against which to check if a solution was found
        """

        # set the genotype solution evaluation function
        self.evaluation_function = eval_func
        self.generation_avg_eval = {}

        # init statistic collection data structs
        self.generation_max_eval = {}
        self.generation_min_eval = {}

        # init the super type
        super(CryptarithmeticSolver, self).__init__((first_term, second_term, result_term), population_size, time_limit)

    def __encoding__(self):
        """ generate an encoding for the problem """
        a, b, c = self.problem
        prob_enc = list(set(a + b + c))
        prob_enc.sort()
        # if the encoding has less than 10 symbols, fill the rest with the unused value symbol
        prob_enc.extend(self.UNUSED_VALUE_SYMBOL * (self.VARIABLE_VALUE_RANGE - len(prob_enc)))
        return prob_enc

    def __encode_evaluation__(self):
        """ encode the evaluation function into a row vector """
        # encode each term
        enc_f_term, enc_s_term, enc_r_term = [self.__encode_term__(term) for term in self.problem]
        # encode the terms into a single evaluation vector
        return self.evaluation_function(enc_f_term, enc_s_term, enc_r_term)

    def __encode_term__(self, term):
        """ encode the string problem into an evaluation vector """
        # reverse the string
        term = term[::-1]
        # create an empty vector
        eval_vec = np.zeros(self.VARIABLE_VALUE_RANGE, dtype=int)
        # assign the decimal value of each variable according to its position in the term
        for i in range(len(term)):
            # use the encoding to encode the character into a vector position
            j = self.encoding.index(term[i])
            # assign the evaluation value
            eval_vec[j] = 10 ** i
        return eval_vec

    def __seed__(self):
        """ generate a randomized genotype population matrix """

        # populate the genotype matrix with copies of the same genotype
        seed = np.arange(self.VARIABLE_VALUE_RANGE)
        seed = np.tile(seed, self.pop_size).reshape(self.pop_size, -1)
        # shuffle the generated genotypes individually
        seed = np.apply_along_axis(lambda x: np.random.permutation(x), self.ROW_AXIS, seed)
        return seed

    def __rand_pop__(self, pop):
        """ generate a randomized rival population """
        # most of the time returns a mutated copy of the existing population
        # if it's a niche generation - returns a freshly create random population (more expensive computationally)
        return self.__seed__() if self.generation_counter % self.NICHE_INTERVAL == 0 else self.__mutate__(np.copy(pop))

    def __mutate__(self, population):
        """ mutate the population """
        # mutate by swapping two randomly selected column in the whole population matrix

        # randomly select columns
        col1 = np.random.randint(0, len(self.encoding))
        col2 = np.random.randint(0, len(self.encoding))
        # swap the selected columns
        population[:, [col1, col2]] = population[:, [col2, col1]]
        return population

    def __evaluation__(self, population):
        """ create an evaluation matrix for the given genotype population """
        # the evaluation is done by multiplying the population matrix by the evaluation vector
        return np.abs(np.matmul(population, self.encoded_evaluation))

    def __fitness__(self, eval1, eval2):
        """ compare evaluations and generate a selection matrix """
        return np.repeat(np.where(eval1 < eval2, True, False), len(self.encoding))

    def __reproduce__(self, pop, pop_eval, rand_pop, rand_pop_eval):
        """ generated a single unified population by element wise evaluation comparison """
        # generate the selection matrix
        fitness = self.__fitness__(pop_eval, rand_pop_eval).reshape(pop.shape)
        # perform element wise selection according to the fitness matrix
        pop = np.where(fitness, pop, rand_pop)
        # return the unified population
        return pop

    def __is_solved__(self, evaluation):
        """ check if a value which  is a solution exists in the evaluation matrix """
        return np.isin(self.SOLUTION_EVALUATION, evaluation)

    def _stats_string__(self, t, generations):
        """ return a string representation of number of generation and run time """
        return 'Time (miliseconds): ' + str(int(t * 1000)) + ' Generations: ' + str(generations)

    def __solved__(self, pop, pop_eval, t, generations):
        """ find the solution genotype in the population matrix and return it in a decoded form """
        sol = pop[pop_eval == self.SOLUTION_EVALUATION]
        # return a string representation of the solution
        return 'solution: ' + self.__decode__(sol) + '\n' + self._stats_string__(t, generations)

    def __decode__(self, solution):
        """ decode a genotype into a string representation """
        # remove the empty placeholder symbols
        filt_enc = list(filter(lambda x: x != '±', self.encoding))
        return str(
            [filt_enc[i] + ' = ' + str(solution.flatten()[i]) for i in range(len(filt_enc))])

    def __record_stats__(self, evaluation):
        """ keep track of the evluation at each generation """
        self.generation_avg_eval[self.generation_counter] = int(np.average(evaluation))
        self.generation_max_eval[self.generation_counter] = int(np.max(evaluation))
        self.generation_min_eval[self.generation_counter] = int(np.min(evaluation))

    def retrieve_stats__(self):
        """ return the collected evaluation statistics lists """
        return self.generation_avg_eval, self.generation_max_eval, self.generation_min_eval, self.generation_counter

    def solve(self):
        """ run the solver """

        # keep track of the time to avoid an infinite solver loop
        start_time = time.time()
        end_time = start_time + self.time_limit

        # generate a random seed genotype population matrix
        pop = self.__seed__()
        # generate a rival genotype population matrix
        rand_pop = self.__rand_pop__(pop)

        # check if run time is still within bounds
        while time.time() < end_time:
            # evaluate the population
            pop_eval = self.__evaluation__(pop)
            # record the evaluation stats
            self.__record_stats__(pop_eval)
            # evaluate the rival population
            rand_pop_eval = self.__evaluation__(rand_pop)

            # check if a solution exists within the matrix - return a string representation of it if it does
            if self.__is_solved__(pop_eval):
                return self.__solved__(pop, pop_eval, time.time() - start_time, self.generation_counter)
            elif self.__is_solved__(rand_pop_eval):
                return self.__solved__(rand_pop, rand_pop_eval, time.time() - start_time, self.generation_counter)

            # select better evaluated genotypes
            pop = self.__reproduce__(pop, pop_eval, rand_pop, rand_pop_eval)
            # generate a randomized rival population
            rand_pop = self.__rand_pop__(pop)

            self.generation_counter += 1

        # couldn't find a solution within the time limit
        return 'FAILED TO FIND A SOLUTION :('


#####################################################################
#    Cryptarithmetic Solver Evaluation Functions
#####################################################################

def add_eval(first_term, second_term, result_term):
    """ create an addition evaluation vector """
    return result_term - first_term - second_term


def sub_eval(first_term, second_term, result_term):
    """ create a subtraction evaluation vector """
    return result_term - first_term + second_term


def mult_eval(first_term, second_term, result_term):
    """ create an multiplication evaluation vector """
    return result_term - second_term * first_term


def div_eval(first_term, second_term, result_term):
    """ create an division evaluation vector """
    return result_term * second_term - first_term


#####################################################################
#    Cryptarithmetic Solver Tests
#####################################################################

# TODO: impl actual unit tests

# TEST_POPULATION_SIZE = 50
# TEST_MAX_TIME_SECONDS = 1
#
#
# def addition_test():
#     a = "SEND"
#     b = "MORE"
#     c = "MONEY"
#     print("--------------------------")
#     print(a, ' + ', b, ' = ', c)
#     # solution = np.array((7, 5, 1, 6, 0, 8, 9, 2))
#     my_solver = CryptarithmeticSolver(a, b, c, TEST_POPULATION_SIZE, TEST_MAX_TIME_SECONDS, add_eval)
#     print(my_solver.solve())
#
#
# def subtraction_test():
#     a = "COUNT"
#     b = "COIN"
#     c = "SNUB"
#     print("--------------------------")
#     print(a, ' - ', b, ' = ', c)
#     my_solver = CryptarithmeticSolver(a, b, c, TEST_POPULATION_SIZE, TEST_MAX_TIME_SECONDS, sub_eval)
#     print(my_solver.solve())
#
#
# def multiplication_test():
#     a = "AB"
#     b = "AB"
#     c = "ABB"
#     print("--------------------------")
#     print(a, ' * ', b, ' = ', c)
#     my_solver = CryptarithmeticSolver(a, b, c, TEST_POPULATION_SIZE, TEST_MAX_TIME_SECONDS, mult_eval)
#     print(my_solver.solve())
#
#
# def division_test():
#     a = "ABB"
#     b = "AB"
#     c = "AB"
#     print("--------------------------")
#     print(a, ' / ', b, ' = ', c)
#     my_solver = CryptarithmeticSolver(a, b, c, TEST_POPULATION_SIZE, TEST_MAX_TIME_SECONDS, div_eval)
#     print(my_solver.solve())
#
#
# def run_solver_tests():
#     addition_test()
#     subtraction_test()
#     multiplication_test()
#     division_test()
