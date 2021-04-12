import numpy as np
from sympy import *
from sympy.abc import *
import random
import math

def convert_str_to_num(str):
        '''
        This function takes a string and convert it into a number. 
        The string can be a mathematical expression that will be evaluated an the result will be returned
        '''
        if str.find('.') == 1:
            return float(str)
        else:
            return eval(str)

def generate_rv_values(rv_from, rv_to, rv_num):
    '''
    This function generate an array of uniformly distributed random values of theta
    that are in the range of from theta_from to theta_to and the number of values are num_theta.
    All the parameters are strings that will be converted to numbers.
    '''
    _from = convert_str_to_num(rv_from)
    _to = convert_str_to_num(rv_to)
    _num = convert_str_to_num(rv_num) 
    return np.random.uniform(_from, _to, _num)

def generate_the_time_array(time_from, time_to, time_step):
    '''
    This function generates an array of evenly spaced values 
    that are in the range of the open half open interval [time_from, time_to)
    and with a step of time_step
    '''
    _from = convert_str_to_num(time_from)
    _to = convert_str_to_num(time_to)
    _step = convert_str_to_num(time_step)
    return np.arange(start = _from, stop = _to, step = _step)

def evaluate_str_expr(ensemble_, rv, t_):
    '''
    This function take a mathematical expression in the form of a string (ensemble_)
    and the values of the expression variables (theta_, omega_, a_ and t_),
    then it convert it to a sympy expression and finally evaluate the expression 
    and return the value
    '''
    expr = sympify(ensemble_)
    return expr.subs([(theta, rv),(t, t_), (pi, np.pi)])

def calc_x_matrix(file, ensemble, rv_values, time_array):
    '''
    This function takes the empty matrix x and loops over it to calculate the values of x for a the given parameters 
    '''
    for row_i in range(rv_values.shape[0]):
        for col_i in range(time_array.shape[0]):
            file.write(str(evaluate_str_expr(ensemble_ = ensemble, rv = rv_values[row_i], t_ = time_array[col_i])))
            file.write(' ')
        file.write('\n')

def write_array_to_file(file, array):
    for i in array:
        file.write(str(i))
        file.write(' ')
    file.write('\n')

def generate_polar_nrz_process(file, a_, t_b_, n_, time_array, alpha_values):
    _n = convert_str_to_num(n_)
    _a = convert_str_to_num(a_)
    _t_b = convert_str_to_num(t_b_)


    for i in alpha_values:
        number_of_elements_in_row = time_array.shape[0]
        q = np.where(time_array >= i)
        alpha_2_time_elements = q[0][0]
        u = np.where(time_array >= _t_b) 
        t_b_2_num_time_elements = u[0][0]
        for _ in range(alpha_2_time_elements):
            file.write('0 ')
            number_of_elements_in_row -= 1
        for _ in range(_n):
            bit = bool(random.getrandbits(1))
            for _ in range(t_b_2_num_time_elements):
                number_of_elements_in_row -= 1
                if bit == True:
                    file.write(str(_a))
                    file.write(' ')
                else:
                    file.write(str(-_a))
                    file.write(' ')
        for _ in range(number_of_elements_in_row):  
            file.write('0 ')
            number_of_elements_in_row -= 1
        file.write('\n')

def generate_unipolar_nrz_process(file, a_, t_b_, n_, time_array, alpha_values):
    _n = convert_str_to_num(n_)
    _a = convert_str_to_num(a_)
    _t_b = convert_str_to_num(t_b_)

    for i in alpha_values:
        number_of_elements_in_row = time_array.shape[0]
        q = np.where(time_array >= i)
        alpha_2_time_elements = q[0][0]
        u = np.where(time_array >= _t_b) 
        t_b_2_num_time_elements = u[0][0]
        for _ in range(alpha_2_time_elements):
            number_of_elements_in_row -= 1
            file.write('0 ')
        for _ in range(_n):
            bit = bool(random.getrandbits(1))
            for _ in range(t_b_2_num_time_elements):
                number_of_elements_in_row -= 1
                if bit == True:
                    file.write(str(_a))
                    file.write(' ')
                else:
                    file.write('0')
                    file.write(' ') 
        for _ in range(number_of_elements_in_row):  
            file.write('0 ')
            number_of_elements_in_row -= 1
        file.write('\n')

def generate_Manchester_process(file, a_, t_b_, n_, time_array, alpha_values):
    _n = convert_str_to_num(n_)
    _a = convert_str_to_num(a_)
    _t_b = convert_str_to_num(t_b_)

    for i in alpha_values:
        number_of_elements_in_row = time_array.shape[0]
        q = np.where(time_array >= i)
        alpha_2_time_elements = q[0][0]
        u = np.where(time_array >= _t_b) 
        t_b_2_num_time_elements = math.floor(u[0][0] / 2)
        for _ in range(alpha_2_time_elements):
            number_of_elements_in_row -= 1
            file.write('0 ')
        for _ in range(_n):
            bit = bool(random.getrandbits(1))
            for _ in range(t_b_2_num_time_elements):
                number_of_elements_in_row -= 1
                if bit == True:
                    file.write(str(_a))
                    file.write(' ')
                else:
                    file.write(str(-_a))
                    file.write(' ')
            for _ in range(t_b_2_num_time_elements):
                number_of_elements_in_row -= 1
                if bit == True:
                    file.write(str(-_a))
                    file.write(' ')
                else:
                    file.write(str(+_a))
                    file.write(' ') 
        for _ in range(number_of_elements_in_row):  
            file.write('0 ')
            number_of_elements_in_row -= 1
        file.write('\n')      

file_name = input("Enter the file name: ")
f = open(file_name, 'w')
f.write('')
f.close()

file = open(file_name, 'a')
print('Choose the type of random process to be generated by entering the number.\n1 Binary random process\n2 Other random process by analytical expression\n')
process_type = input(' ')
if process_type == '1':
    binary_process_type = input("Enter the type of the number of the binary process type to be generated.\n1 Polar NRZ process\n2 Unipolar NRZ process\n3 Manchester process\n")
    time_step = input("Enter the time step: ")
    a_str = input("Enter the amplitude in volts, A = ")
    t_b_str = input('Enter the bit duration in seconds, T_b = ')
    n_str = input('Enter the number of bits to be generated, N = ')
    print('Enter he number of random variable values to be generated')
    alpha_num = input('num of values: ')
    time_array = np.arange(0, convert_str_to_num(n_str)*convert_str_to_num(t_b_str)+ convert_str_to_num(t_b_str), convert_str_to_num(time_step))
    write_array_to_file(file, time_array)
    alpha_values = generate_rv_values('0', t_b_str, alpha_num)
    if binary_process_type == '1':
        generate_polar_nrz_process(file, a_str, t_b_str, n_str, time_array, alpha_values)
    elif binary_process_type == '2':
        generate_unipolar_nrz_process(file, a_str, t_b_str, n_str, time_array, alpha_values)
    elif binary_process_type == '3':
        generate_Manchester_process(file, a_str, t_b_str, n_str, time_array, alpha_values)

elif process_type == '2':
    print('Enter the range of time as well as the time step')
    time_from = input('from')
    print(type(time_from))
    time_to = input('to')
    time_step = input('step')
    print('Use theta as the random variable\nEnter the range of the random variable')
    theta_from = input('From')
    theta_to = input('to')
    theta_num = input('number of values to be generated')
    expr_str = input('Enter the equation expression X(t) = ')
    time_array = generate_the_time_array(time_from, time_to, time_step)   
    write_array_to_file(file, time_array)
    theta_values = generate_rv_values(theta_from, theta_to, theta_num)
    calc_x_matrix(file, expr_str, theta_values, time_array)

file.close()
