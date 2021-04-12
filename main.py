from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from os import path
import numpy as np
from sympy import *
from sympy.abc import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.fft import fft, ifft

#region Global variable

pi = 3.141592

#endregion

#region UI

#region UI methods
def submit_btn_clicked():
    messagebox.showwarning(title = 'Warning', message = 'It may take some time to show the results depending on the number of elements you entered.\nIf you entered many elements be prepared to wait a while\nIf the power spectral density plot appeared to be empty just run the program again.\nPress Ok to continue.')
    main()

def write_array_to_file(file, array):
    for i in array:
        file.write(str(i))
        file.write(' ')
    file.write('\n')
    
def show_results(results_file_name, ensemble_mean, time_mean,acf_matrix, acf_tao, acf_i_j, n_time_acf, psd, tap):
    f = open(results_file_name, 'w')
    f.write('')
    f.close()

    results_file = open(results_file_name, 'a')

    results_file.write('Ensemble Mean: ')
    write_array_to_file(results_file, ensemble_mean)
    results_file.write("\nThe time mean of the " + n_value.get() + "th sample function: " + str(time_mean))
    results_file.write('\nAutocorrelation function for all possible i and j: ')
    for i in range(acf_matrix.shape[0]):
        write_array_to_file(results_file, acf_matrix[i])
    results_file.write('\nAutocorrelation function in terms of tao: ')
    write_array_to_file(results_file, acf_tao)
    results_file.write('\nAutocorrelation function between ith and jth sample : ' + str(acf_i_j))
    results_file.write('\nTime Autocorrelation function: ')
    write_array_to_file(results_file, n_time_acf)
    results_file.write('\nPower spectral density: ')
    write_array_to_file(results_file, psd)
    results_file.write('\ntotal average power: ' + str(tap))



#endregion

#region Create a Window

window = Tk()
#endregion

#region Set up tabs and tab control

tab_control = ttk.Notebook(window)

tab1 = ttk.Frame(tab_control)
results_tab = ttk.Frame(tab_control)

tab_control.add(tab1, text='Input Data')
tab_control.add(results_tab, text = "Results")


#endregion

#region Input Data Tab

#region UI data variables

sepration_line_txt = '___________________________________________________________'

#endregion

#region Add UI widgets

#region Time Vector input 

# time_from_value = StringVar()

# time_to_value = StringVar()

# time_step_value = StringVar()

# sep_line1 = Label(tab1, text = sepration_line_txt)
# sep_line1.grid(row = 0)

# time_info_lbl = Label(tab1, text = "Enter the time vector range and time step to be generated:")
# time_info_lbl.grid(row = 1)

# time_vector_frame = Frame(tab1)
# time_vector_frame.grid(row = 2)


# time_from_lbl = Label(time_vector_frame, text = "From :")
# time_from_lbl.grid(row = 0, column = 0)
# time_from_entry = Entry(time_vector_frame, textvariable = time_from_value)
# time_from_entry.grid(row = 0, column = 1)

# time_to_lbl = Label(time_vector_frame, text = "To :")
# time_to_lbl.grid(row = 1, column = 0)
# time_to_entry = Entry(time_vector_frame, textvariable = time_to_value)
# time_to_entry.grid(row = 1, column = 1)

# time_step_lbl = Label(time_vector_frame, text = "Step :")
# time_step_lbl.grid(row = 2, column = 0)
# time_step_entry = Entry(time_vector_frame, textvariable = time_step_value)
# time_step_entry.grid(row = 2, column = 1)

#endregion

#region Choose the type of the random process

#* Variables
process_type_value = StringVar()

#* Functions
# def binary_process_type_selected():
#     ensemble_input_type_frame.grid_remove()
#     anal_ensemble_input_frame.grid_remove()

# def other_process_type_selected():
#     ensemble_input_type_frame.grid()
#     numerical_ensemble_input_type_rad.select()
#     numerical_ensemble_input_type_rad.invoke()    

#* UI Widgets
sep_line2 = Label(tab1, text = sepration_line_txt)
sep_line2.grid(row = 3)

process_type_lbl = Label(tab1, text = "Choose the type of the random process")
process_type_lbl.grid(row = 4)

process_type_frame = Frame(tab1)
process_type_frame.grid(row = 5)

binary_process_type_rad = Radiobutton(process_type_frame, text = 'Binary Random Process', value = 'binary', variable = process_type_value)
binary_process_type_rad.grid(row = 0, column = 1)

other_process_type_rad = Radiobutton(process_type_frame, text = 'Other Random Process', value = 'other', variable = process_type_value)
other_process_type_rad.grid(row = 0, column = 0)
other_process_type_rad.select()
#endregion

#region Choose the way to input the ensemble
# #* Variables
# ensemble_input_type_value = StringVar()

# #* Functions 
# def expr_ensemble_input_type_selected():
#     anal_ensemble_input_frame.grid()
#     numerical_ensemble_input_frame.grid_remove()

# def numerical_ensemble_input_type_selected():
#     anal_ensemble_input_frame.grid_remove()
#     numerical_ensemble_input_frame.grid()

# #* UI Widgets
# sep_line3 = Label(tab1, text = sepration_line_txt)
# sep_line3.grid(row = 6)

# ensemble_input_type_lbl = Label(tab1, text = 'Choose the way you prefer to input the ensemble')
# ensemble_input_type_lbl.grid(row = 7)

# ensemble_input_type_frame = Frame(tab1)
# ensemble_input_type_frame.grid(row = 8)

# expr_ensemble_input_type_rad = Radiobutton(ensemble_input_type_frame, text = 'Analytical Expression', value = 'expr', variable = ensemble_input_type_value, command = expr_ensemble_input_type_selected) 
# expr_ensemble_input_type_rad.grid(row = 0, column = 0)

# numerical_ensemble_input_type_rad = Radiobutton(ensemble_input_type_frame, text = 'Numerical Values', value = 'numerical', variable = ensemble_input_type_value, command = numerical_ensemble_input_type_selected)
# numerical_ensemble_input_type_rad.grid(row = 0, column = 1)

# numerical_ensemble_input_type_rad.select()

#endregion

#region Ensemble input in analytical expression form

# #* Variable
# rv_name_value = StringVar()
# rv_from_value = StringVar()
# rv_to_value = StringVar()
# rv_num_value = StringVar()
# ensemble_value = StringVar()

# #region Functions
# def alpha_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "alpha")

# def omega_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "omega")

# def theta_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "theta")

# def cos_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "cos(")

# def sin_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "sin(")

# def tan_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "tan(")

# def plus_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "+")
    
# def minus_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "-")
    
# def multiply_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "*")
    
# def divid_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "/")
    
# def open_bracket_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, "(")
    
# def close_bracket_btn_clicked():
#     anal_ensemble_input_entry.insert(INSERT, ")")
# #endregion

# #* UI Widgets
# sep_line4 = Label(tab1, text = sepration_line_txt)
# sep_line4.grid(row = 9)

# anal_ensemble_input_lbl = Label(tab1, text = 'Enter the ensemble analytical expression and don\'t use any variable names other than the one you chosen from the dropdown menu')
# anal_ensemble_input_lbl.grid(row = 10)

# anal_ensemble_input_frame = Frame(tab1)
# anal_ensemble_input_frame.grid(row = 12)

# theta_rv_radio = Radiobutton(anal_ensemble_input_frame, text = 'theta', value = 'theta', variable = rv_name_value)
# theta_rv_radio.grid(row = 0, column = 0)
# theta_rv_radio.select()

# omega_rv_radio = Radiobutton(anal_ensemble_input_frame, text = 'omega', value = 'omega', variable = rv_name_value)
# omega_rv_radio.grid(row = 0, column = 1)

# alpha_rv_radio = Radiobutton(anal_ensemble_input_frame, text = 'alpha', value = 'alpha', variable = rv_name_value)
# alpha_rv_radio.grid(row = 0, column = 2)

# rv_info_lbl = Label(anal_ensemble_input_frame, text = "Input the range of the random variable in numbers as well as the number of values to be generated.\nFor example rv = unifrnd(from, to, number of rv values).")
# rv_info_lbl.grid(row = 1)

# rv_from_lbl = Label(anal_ensemble_input_frame, text = "From :")
# rv_from_lbl.grid(row = 2, column = 0)
# rv_from_entry = Entry(anal_ensemble_input_frame, textvariable = rv_from_value)
# rv_from_entry.grid(row = 2, column = 1)

# rv_to_lbl = Label(anal_ensemble_input_frame, text = "To :")
# rv_to_lbl.grid(row = 4, column = 0)
# rv_to_entry = Entry(anal_ensemble_input_frame, textvariable = rv_to_value)
# rv_to_entry.grid(row = 4, column = 1)

# rv_num_values_lbl = Label(anal_ensemble_input_frame, text = "Number of RV values :")
# rv_num_values_lbl.grid(row = 5, column = 0)
# rv_num_values_entry = Entry(anal_ensemble_input_frame, textvariable = rv_num_value)
# rv_num_values_entry.grid(row = 5, column = 1)

# anal_ensemble_input_lbl = Label(anal_ensemble_input_frame, text = "X(t) = ")
# anal_ensemble_input_lbl.grid(row = 12, column = 0)
# anal_ensemble_input_entry = Entry(anal_ensemble_input_frame, textvariable = ensemble_value, width = 30)
# anal_ensemble_input_entry.grid(row = 12, column = 1)

# alpha_btn = Button(anal_ensemble_input_frame, command = alpha_btn_clicked, text = "alpha", width = 10)
# omega_btn = Button(anal_ensemble_input_frame, command = omega_btn_clicked, text = "omega", width = 10)
# theta_btn = Button(anal_ensemble_input_frame, command = theta_btn_clicked, text = "theta", width = 10)
# cos_btn = Button(anal_ensemble_input_frame, command = cos_btn_clicked, text = "cos", width = 10)
# sin_btn = Button(anal_ensemble_input_frame, command = sin_btn_clicked, text = "sin", width = 10)
# tan_btn = Button(anal_ensemble_input_frame, command = tan_btn_clicked, text = "tan", width = 10)
# plus_btn = Button(anal_ensemble_input_frame, command = plus_btn_clicked, text = "+", width = 10)
# minus_btn = Button(anal_ensemble_input_frame, command = minus_btn_clicked, text = "-", width = 10)
# multiply_btn = Button(anal_ensemble_input_frame, command = multiply_btn_clicked, text = "*", width = 10)
# divid_btn = Button(anal_ensemble_input_frame, command = divid_btn_clicked, text = "/", width = 10)
# open_bracket_btn = Button(anal_ensemble_input_frame, command = open_bracket_btn_clicked, text = "(", width = 10)
# close_bracket_btn = Button(anal_ensemble_input_frame, command = close_bracket_btn_clicked, text = ")", width = 10)

# alpha_btn.grid(row = 13, column = 0)
# omega_btn.grid(row = 14, column = 0)
# theta_btn.grid(row = 15, column = 0)
# sin_btn.grid(row = 13, column = 1)
# cos_btn.grid(row = 14, column = 1)
# tan_btn.grid(row = 15, column = 1)
# plus_btn.grid(row = 13, column = 3)
# minus_btn.grid(row = 13, column = 4)
# multiply_btn.grid(row = 14, column = 3)
# divid_btn.grid(row = 14, column = 4)
# open_bracket_btn.grid(row = 15, column = 3)
# close_bracket_btn.grid(row = 15, column = 4)

# anal_ensemble_input_frame.grid_remove()

#endregion

#region Numerical ensemble input

file_name_value = StringVar()

#*Functions 

def choose_file_btn_clicked():
    file_name_value.set(filedialog.askopenfilename(filetypes = (("Text files", "*.txt"), ("all files", "*.*")), initialdir = path.dirname(__file__)))
    chosen_file_lbl.config(text = file_name_value.get())

#* UI Widgets
sep_line5 = Label(tab1, text = sepration_line_txt)
sep_line5.grid(row = 13)

numerical_ensemble_input_lbl = Label(tab1, text = 'Choose the file that contains the numerical values of the ensemble.\nThe format of the file should be as follows.')
numerical_ensemble_input_lbl.grid(row = 14)

file_format_lbl = Label(tab1, text = '1 4 5 6 7 8\n1 4 5 6 7 8\n1 4 5 6 7 8\n1 4 5 6 7 8')
file_format_lbl.grid(row = 15)

numerical_ensemble_input_frame = Frame(tab1)
numerical_ensemble_input_frame.grid(row = 16)

choose_file_btn = Button(numerical_ensemble_input_frame, text = 'Choose file', command = choose_file_btn_clicked)
choose_file_btn.grid(row = 0, column = 0)

chosen_file_lbl = Label(numerical_ensemble_input_frame, text = '')
chosen_file_lbl.grid(row = 0, column = 1)

#endregion

#region M, n, i, and j input fields

#* Variables
m_value = StringVar()

n_value = StringVar()

i_value = StringVar()

j_value = StringVar()

#* UI Widgets

m_n_i_j_input_lbl = Label(tab1, text = 'Enter the values for M, n, i, and j')
m_n_i_j_input_lbl.grid(row = 17, column = 0)

m_n_i_j_input_frame = Frame(tab1)
m_n_i_j_input_frame.grid(row = 18, column = 0)

m_input_label = Label(m_n_i_j_input_frame, text = 'M :')
m_input_label.grid(row = 0, column = 0 )
m_input_entry = Entry(m_n_i_j_input_frame, textvariable = m_value)
m_input_entry.grid(row = 0, column = 1)
m_input_entry.insert(INSERT, '0')

n_input_lbl = Label(m_n_i_j_input_frame, text = "n :")
n_input_lbl.grid(row = 1, column = 0)
n_input_entry = Entry(m_n_i_j_input_frame, textvariable = n_value)
n_input_entry.grid(row = 1, column = 1)

i_input_lbl = Label(m_n_i_j_input_frame, text = "i :")
i_input_lbl.grid(row = 2, column = 0)
i_input_entry = Entry(m_n_i_j_input_frame, textvariable = i_value)
i_input_entry.grid(row = 2, column = 1)

j_input_lbl = Label(m_n_i_j_input_frame, text = "j :")
j_input_lbl.grid(row = 3, column = 0)
j_input_entry = Entry(m_n_i_j_input_frame, textvariable = j_value)
j_input_entry.grid(row = 3, column = 1)

#endregion

resutls_file_name_value = StringVar()
resutls_file_name_value.set('results.txt')

results_file_lbl = Label(tab1, text = 'Enter the name of the file to store the results')
results_file_lbl.grid(row = 20, column = 0)
results_file_entry = Entry(tab1, textvariable = resutls_file_name_value)
results_file_entry.grid(row = 20, column = 1)

submit_btn = Button(tab1, text = "Submit values", command = submit_btn_clicked)
submit_btn.grid(row = 21)


#endregion

#endregion

#region Results Tab

#region UI Widgets

# ensemble_mean_lbl = Label(results_tab, text = 'Ensemble mean: ')
# ensemble_mean_lbl.grid(row = 0, column = 0)
# ensemble_mean_result = Label(results_tab, text = ' ')
# ensemble_mean_result.grid(row = 0, column = 1)

# time_mean_lbl = Label(results_tab, text = "The time mean of the nth sample function: ")
# time_mean_lbl.grid(row = 1, column = 0)
# time_mean_result = Label(results_tab, text = ' ')
# time_mean_result.grid(row = 1, column = 1)

# acf_i_j_lbl = Label(results_tab, text = 'The autocorrelation function between ith sample and jth sample')
# acf_i_j_lbl.grid(row = 2, column = 0)
# acf_i_j_result = Label(results_tab, text = ' ')
# acf_i_j_result.grid(row = 2, column = 1)

# time_acf_lbl = Label(results_tab, text = "The time autocorrelation function of the nth sample function")
# time_acf_lbl.grid(row = 3, column = 0)
# time_acf_result = Label(results_tab, text = ' ')
# time_acf_result.grid(row = 3, column = 1)

# psd_lbl = Label(results_tab, text = 'The power spectral density')
# psd_lbl.grid(row = 4, column = 0)
# psd_result = Label(results_tab, text = ' ')
# psd_result.grid(row =4, column = 1)

# tap_lbl = Label(results_tab, text = 'The total average power')
# tap_lbl.grid(row = 5, column = 0)
# tap_result = Label(results_tab, text = " ")
# tap_result.grid(row = 6, column = 1)

#endregion

#endregion

#endregion

#region Core Logic

#region Logic functions
def convert_str_to_num(str):
        '''
        This function takes a string and convert it into a number. 
        The string can be a mathematical expression that will be evaluated an the result will be returned
        '''
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

def evaluate_str_expr(ensemble_, rv_name, rv, t_):
    '''
    This function take a mathematical expression in the form of a string (ensemble_)
    and the values of the expression variables (theta_, omega_, a_ and t_),
    then it convert it to a sympy expression and finally evaluate the expression 
    and return the value
    '''
    expr = sympify(ensemble_)
    if rv_name == 'theta':
        return expr.subs([(theta, rv),(t, t_), (pi, 3.14)])
    elif rv_name == 'omega': 
        return expr.subs([(omega, rv),(t, t_), (pi, 3.14)])
    elif rv_name == 'alpha':        
        return expr.subs([(alpha, rv),(t, t_), (pi, 3.14)])

def calc_x_matrix(x, ensemble, rv_values, rv_name, time_array):
    '''
    This function takes the empty matrix x and loops over it to calculate the values of x for a the given parameters 
    '''
    for row_i in range(x.shape[0]):
        for col_i in range (x.shape[1]):
            x[row_i][col_i] = evaluate_str_expr(ensemble_ = ensemble, rv = rv_values[row_i], rv_name = rv_name, t_ = time_array[col_i])

def plot_m_sample_functions(x, m_, time_array, process_type):
    '''
    This function takes the matrix of values of x and the number of sample functions to be plotted (m_)
    and plot m_ sample function randomly
    '''
    for _ in range(m_):
        rand_i = random.randrange(start = 0, stop = x.shape[0])
        plt.figure()
        if process_type == 'binary':
            plt.step(time_array, x[rand_i])
        else:
            plt.plot(time_array, x[rand_i])
        plt.xlabel('time (t)')
        plt.ylabel('x(t)')
        plt.title("Plot of the sample function number: " + str(rand_i + 1))

def calc_and_plot_ensemble_mean(x, time_array):
    '''
    This function takes the ensemble matrix and the time array
    and calculate the mean of the process and plot it.
    It also return the mean array to be used later.
    '''
    mean = np.mean(x, axis = 0)
    plt.figure()
    plt.plot(time_array, mean)
    plt.xlabel("time (t)")
    plt.ylabel("mean of x")
    plt.title("Plot of ensemble mean")
    return mean

def calc_time_mean_of_n(x, n_):
    mean = np.mean(x[n_])
    return mean

def calc_ACF_for_i_and_j(x, i, j):
    acf_value = 0
    for row_i in range(x.shape[0]):
        acf_value += x[row_i][i] * x[row_i][j]
    acf_value = acf_value/x.shape[0]
    return acf_value

def calc_acf_matrix_for_all_i_and_j(x, acf_matrix):
    for row_i in range(acf_matrix.shape[0]):
        for col_i in range(acf_matrix.shape[0]):
            acf_matrix[row_i][col_i] = calc_ACF_for_i_and_j(x, row_i, col_i)

def calc_acf_in_term_of_tao(acf_matrix):
    acf_tao = np.empty(acf_matrix.shape[0])
    acf_tao_element_count = np.empty(acf_tao.shape[0])
    for i in range(acf_matrix.shape[0]):
        for j in range(acf_matrix.shape[1]):
            acf_tao[np.absolute(i-j)] += acf_matrix[i][j]
            acf_tao_element_count[np.absolute(i-j)] += 1
    for i in range(acf_tao.shape[0]):
        acf_tao[i] = acf_tao[i] / acf_tao_element_count[i]
    return acf_tao


def plot_acf_3d(acf_matrix, time_array):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(time_array, time_array, acf_matrix, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Auto-correlation function')

def calc_time_acf(sample_func):
    acf_array = []
    for i in range(sample_func.shape[0]):
        acf_value = 0
        count = 0
        for j in range(sample_func.shape[0]):
            if j+i >= sample_func.shape[0]:
                break
            else:
                count += 1
                acf_value += sample_func[j]*sample_func[j+i]
        acf_value = acf_value/count
        acf_array.append(acf_value)
    return acf_array

# def plot_time_acf(acf_array, time_array, n):

#     plt.figure()
#     plt.plot(time_array, acf_array)
#     plt.xlabel("time (t)")
#     plt.ylabel("autocorrelation value")
#     plt.title("Plot of the time auto correlation function of the "+ str(n) + " sample function")

def calc_and_plot_psd(acf_tao, time_array):
    psd = fft(acf_tao)
    omega_array = fft(time_array)

    plt.figure()    
    plt.plot(psd)
    plt.xlabel('frequency Omega')
    plt.ylabel('power spectral density')
    plt.title('Plot of the power spectral density')

    return psd

def calc_total_average_power(acf_matrix):
    s = 0
    c = 0
    for i in range(acf_matrix.shape[0]):
        for j in range(acf_matrix.shape[1]):
            if i == j:
                s += acf_matrix[i][j]
                c += 1
    s = s / c
    return s

def main():
    '''
    This is the main function responsible for all the stochastic process calculation and displaying the results
    '''
    x = []
    time_array = []
    # if ensemble_input_type_value.get() == 'expr':
    #     time_array = generate_the_time_array(time_from = time_from_value.get(), time_to = time_to_value.get(), time_step = time_step_value.get())
    #     rv_values = generate_rv_values(rv_from = rv_from_value.get(), rv_to = rv_to_value.get(), rv_num = rv_num_value.get())
    #     ensemble = ensemble_value.get()
    #     anal_x = np.empty([rv_values.shape[0], time_array.shape[0]])
    #     calc_x_matrix(anal_x, ensemble, rv_values = rv_values, rv_name = rv_name_value.get(), time_array = time_array)
    #     x = anal_x

    # elif ensemble_input_type_value.get() == 'numerical':

    with open(file_name_value.get()) as f:
        time = [float(x) for x in next(f).split()]
        row_i = 0
        array = []
        for line in f:
            array.append([float(n) for n in line.split()])
        x = np.array(array)
        time_array = np.array(time)
    _m = convert_str_to_num(m_value.get())
    _n = convert_str_to_num(n_value.get())
    _i = convert_str_to_num(i_value.get())
    _j = convert_str_to_num(j_value.get())

    plot_m_sample_functions(x = x, m_ = _m, time_array = time_array, process_type= process_type_value.get())

    ensemble_mean = calc_and_plot_ensemble_mean(x, time_array)

    time_mean = calc_time_mean_of_n(x, _n)

    acf_matrix = np.empty([time_array.shape[0], time_array.shape[0]])

    calc_acf_matrix_for_all_i_and_j(x, acf_matrix)

    acf_tao = calc_acf_in_term_of_tao(acf_matrix)

    acf_i_j = acf_matrix[_i][_j]

    plot_acf_3d(acf_matrix, time_array)

    n_time_acf = calc_time_acf(x[_n])

    psd = calc_and_plot_psd(acf_tao, time_array)

    total_average_power = calc_total_average_power(acf_matrix)

    show_results(results_file_name = resutls_file_name_value.get(),ensemble_mean = ensemble_mean, time_mean = time_mean, acf_matrix = acf_matrix, acf_tao = acf_tao, acf_i_j = acf_i_j, n_time_acf = n_time_acf, psd = psd, tap = total_average_power) #Todo: pass the rest of the results to the function
    
    plt.show()

    
#endregion

#endregion



tab_control.pack(expand=1, fill='both')
window.mainloop()