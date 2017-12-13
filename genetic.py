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
    SOLUTION_EVALUATION = 0
    VARIABLE_VALUE_RANGE = 10
    ROW_AXIS = 1
    UNUSED_VALUE_SYMBOL = '±'

    def __init__(self, first_term, second_term, result_term, population_size, time_limit, eval_func):
        self.evaluation_function = eval_func
        super(CryptarithmeticSolver, self).__init__((first_term, second_term, result_term), population_size, time_limit)

    def __encoding__(self):
        a, b, c = self.problem
        prob_enc = list(set(a + b + c))
        prob_enc.sort()
        prob_enc.extend(self.UNUSED_VALUE_SYMBOL * (self.VARIABLE_VALUE_RANGE - len(prob_enc)))
        return prob_enc

    def __encode_evaluation__(self):
        enc_f_term, enc_s_term, enc_r_term = [self.__encode_term__(term) for term in self.problem]
        return self.evaluation_function(enc_f_term, enc_s_term, enc_r_term)

    def __encode_term__(self, term):
        term = term[::-1]
        eval_vec = np.zeros(self.VARIABLE_VALUE_RANGE, dtype=int)
        for i in range(len(term)):
            j = self.encoding.index(term[i])
            eval_vec[j] = 10 ** i
        return eval_vec

    def __seed__(self):
        seed = np.random.choice(self.VARIABLE_VALUE_RANGE, len(self.encoding), replace=False)
        seed = np.tile(seed, self.pop_size).reshape(self.pop_size, -1)
        seed = np.apply_along_axis(lambda x: np.random.permutation(x), self.ROW_AXIS, seed)
        return seed

    def __rand_pop__(self, pop):
        return self.__seed__() if self.generation_counter % 10 == 0 else self.__mutate__(np.copy(pop))

    def __mutate__(self, population):
        col1 = np.random.randint(0, len(self.encoding))
        col2 = np.random.randint(0, len(self.encoding))
        population[:, [col1, col2]] = population[:, [col2, col1]]
        return population

    def __evaluation__(self, population):
        return np.abs(np.matmul(population, self.encoded_evaluation))

    def __fitness__(self, eval1, eval2):
        return np.repeat(np.where(eval1 < eval2, True, False), len(self.encoding))

    def __reproduce__(self, pop, pop_eval, rand_pop, rand_pop_eval):
        fitness = self.__fitness__(pop_eval, rand_pop_eval).reshape(pop.shape)
        pop = np.where(fitness, pop, rand_pop)
        return pop

    def __is_solved__(self, evaluation):
        return np.isin(self.SOLUTION_EVALUATION, evaluation)

    def __stats__(self, t, generations):
        return 'Time (miliseconds): ' + str(int(t * 1000)) + ' Generations: ' + str(generations)

    def __solved__(self, pop, pop_eval, t, generations):
        sol = pop[pop_eval == self.SOLUTION_EVALUATION]
        return 'solution: ' + self.__decode__(sol) + '\n' + self.__stats__(t, generations)

    def __decode__(self, solution):
        filt_enc = list(filter(lambda x: x != '±', self.encoding))
        return str(
            [filt_enc[i] + ' = ' + str(solution.flatten()[i]) for i in range(len(filt_enc))])

    def solve(self):
        start_time = time.time()
        end_time = start_time + self.time_limit

        pop = self.__seed__()
        rand_pop = self.__rand_pop__(pop)

        while time.time() < end_time:

            pop_eval = self.__evaluation__(pop)
            rand_pop_eval = self.__evaluation__(rand_pop)

            if self.__is_solved__(pop_eval):
                return self.__solved__(pop, pop_eval, time.time() - start_time, self.generation_counter)
            elif self.__is_solved__(rand_pop_eval):
                return self.__solved__(rand_pop, rand_pop_eval, time.time() - start_time, self.generation_counter)

            pop = self.__reproduce__(pop, pop_eval, rand_pop, rand_pop_eval)
            rand_pop = self.__rand_pop__(pop)
            self.generation_counter += 1

        return 'FAILED TO FIND A SOLUTION :('


#####################################################################
#    Cryptarithmetic Solver Evaluation Functions
#####################################################################

def addition_evaluation(first_term, second_term, result_term):
    return result_term - first_term - second_term


def subtraction_evaluation(first_term, second_term, result_term):
    return result_term - first_term + second_term


def multiplication_evaluation(first_term, second_term, result_term):
    return result_term - second_term * first_term


def division_evaluation(first_term, second_term, result_term):
    return result_term * second_term - first_term


#####################################################################
#    Cryptarithmetic Solver Tests
#####################################################################

TEST_POPULATION_SIZE = 50
TEST_MAX_TIME_SECONDS = 1


def addition_test():
    a = "SEND"
    b = "MORE"
    c = "MONEY"
    print("--------------------------")
    print(a, ' + ', b, ' = ', c)
    # solution = np.array((7, 5, 1, 6, 0, 8, 9, 2))
    my_solver = CryptarithmeticSolver(a, b, c, TEST_POPULATION_SIZE, TEST_MAX_TIME_SECONDS, addition_evaluation)
    print(my_solver.solve())


def subtraction_test():
    a = "COUNT"
    b = "COIN"
    c = "SNUB"
    print("--------------------------")
    print(a, ' - ', b, ' = ', c)
    my_solver = CryptarithmeticSolver(a, b, c, TEST_POPULATION_SIZE, TEST_MAX_TIME_SECONDS, subtraction_evaluation)
    print(my_solver.solve())


def multiplication_test():
    a = "AB"
    b = "AB"
    c = "ABB"
    print("--------------------------")
    print(a, ' * ', b, ' = ', c)
    my_solver = CryptarithmeticSolver(a, b, c, TEST_POPULATION_SIZE, TEST_MAX_TIME_SECONDS, multiplication_evaluation)
    print(my_solver.solve())


def division_test():
    a = "ABB"
    b = "AB"
    c = "AB"
    print("--------------------------")
    print(a, ' / ', b, ' = ', c)
    my_solver = CryptarithmeticSolver(a, b, c, TEST_POPULATION_SIZE, TEST_MAX_TIME_SECONDS, division_evaluation)
    print(my_solver.solve())


def run_solver_tests():
    addition_test()
    subtraction_test()
    multiplication_test()
    division_test()

