###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################### FILE NAME: CommonMethods.py ###################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Common methods used during fluid data computation and processing steps.                    |
#                                                                                             |
# Version: 3.1.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np
import cantera as ct


def ComputeLewisNumber(flame:ct.Solution):
    Le_species = flame.thermal_conductivity/flame.cp_mass/flame.density_mass/(flame.mix_diff_coeffs+1e-15)
    return Le_species

def avg_Le_start_end(Le_sp:np.ndarray):
    Le_av = 0.5*(Le_sp[0] + Le_sp[-1])
    return Le_av

def avg_Le_arythmic(Le_sp:np.ndarray):
    Le_av = np.average(Le_sp)
    return Le_av

def avg_Le_min_max(Le_sp:np.ndarray):
    Le_av = 0.5*(np.min(Le_sp)+np.max(Le_sp))
    return Le_av

def avg_Le_unity(Le_sp:np.ndarray):
    Le_av = np.ones(np.shape(Le_sp))
    return Le_av

def avg_Le_const(Le_sp:np.ndarray, Le_const:float):
    Le_av = Le_const * np.ones(np.shape(Le_sp))
    return Le_av

def avg_Le_local(Le_sp:np.ndarray):
    return Le_sp


def GetReferenceData(dataset_file:str, x_vars:list[str], train_variables:list[str],dtype=np.float32):
    """Read csv file and collect input-output pairs into a numpy array of a set data type.

    :param dataset_file: file path name to csv file
    :type dataset_file: str
    :param x_vars: controlling variables
    :type x_vars: list[str]
    :param train_variables: output variables
    :type train_variables: list[str]
    :param dtype: data type by which to output data arrays, defaults to np.float32
    :type dtype: dtype, optional
    :return: data arrays for controlling variable and dependent variable data.
    :rtype: np.ndarray[float],np.ndarray[float]
    """

    # Open data file and get variable names from the first line
    fid = open(dataset_file, 'r')
    line = fid.readline()
    fid.close()
    line = line.strip()
    line_split = line.split(',')
    if(line_split[0][0] == '"'):
        varnames = [s[1:-1] for s in line_split]
    else:
        varnames = line_split

    # Get indices of controlling and train variables
    iVar_x = [varnames.index(v) for v in x_vars]
    iVar_y = [varnames.index(v) for v in train_variables]

    # Retrieve respective data from data set
    D = np.loadtxt(dataset_file, delimiter=',', skiprows=1, dtype=dtype)
    X_data = D[:, iVar_x]
    Y_data = D[:, iVar_y]

    return X_data, Y_data

def write_SU2_MLP(file_out:str, weights:list[np.ndarray], biases:list[np.ndarray],activation_function_name:str,train_vars:list[str], controlling_vars:list[str], scaler_function:str,scaler_function_vals_in:list[list[float]],scaler_function_vals_out:list[float],additional_header_info_function=None):
    """Write ASCII file that can be loaded into SU2 through the MLPCpp submodule containing the network weights and biases.

    :param file_out: MLP output file name
    :type file_out: str
    :param weights: list with network weight values
    :type weights: list[np.ndarray]
    :param biases: list with network bias values
    :type biases: list[np.ndarray]
    :param activation_function_name: hidden layer activation function name
    :type activation_function_name: str
    :param train_vars: dependent variable names
    :type train_vars: list[str]
    :param controlling_vars: controlling variable names
    :type controlling_vars: list[str]
    :param scaler_function: scaler function by which the dependent and controlling data are scaled
    :type scaler_function: str
    :param scaler_function_vals_in: linear scaling parameters for controlling variable data
    :type scaler_function_vals_in: list[list[float]]
    :param scaler_function_vals_out: linear scaling parameters for dependent variable data
    :type scaler_function_vals_out: list[float]
    :param additional_header_info_function: function writing an optional message in the MLP file header, defaults to None
    :type additional_header_info_function: function, optional
    """
    n_layers = len(weights)+1

    weights_for_output = weights
    biases_for_output = biases

    # Opening output file
    fid = open(file_out+'.mlp', 'w+')
    fid.write("<header>\n\n")

    if additional_header_info_function:
        additional_header_info_function(fid)

    # Writing number of neurons per layer
    fid.write('[number of layers]\n%i\n\n' % n_layers)
    fid.write('[neurons per layer]\n')
    activation_functions = []

    for iLayer in range(n_layers-1):
        if iLayer == 0:
            activation_functions.append('linear')
        else:
            activation_functions.append(activation_function_name)
        n_neurons = np.shape(weights_for_output[iLayer])[0]
        fid.write('%i\n' % n_neurons)
    fid.write('%i\n' % len(train_vars))

    activation_functions.append('linear')

    # Writing the activation function for each layer
    fid.write('\n[activation function]\n')
    for iLayer in range(n_layers):
        fid.write(activation_functions[iLayer] + '\n')

    # Writing the input and output names
    fid.write('\n[input names]\n')
    for input in controlling_vars:
            fid.write(input + '\n')

    fid.write('\n[input regularization method]\n%s\n' % scaler_function)

    fid.write('\n[input normalization]\n')
    for i in range(len(controlling_vars)):
        fid.write('%+.16e\t%+.16e\n' % (scaler_function_vals_in[i][0], scaler_function_vals_in[i][1]))

    fid.write('\n[output names]\n')
    for output in train_vars:
        fid.write(output+'\n')

    fid.write('\n[output regularization method]\n%s\n' % scaler_function)

    fid.write('\n[output normalization]\n')
    for i in range(len(train_vars)):
        fid.write('%+.16e\t%+.16e\n' % (scaler_function_vals_out[i][0], scaler_function_vals_out[i][1]))
    fid.write("\n</header>\n")
    # Writing the weights of each layer
    fid.write('\n[weights per layer]\n')
    for W in weights_for_output:
        fid.write("<layer>\n")
        for i in range(np.shape(W)[0]):
            fid.write("\t".join("%+.16e" % float(w) for w in W[i, :]) + "\n")
        fid.write("</layer>\n")

    # Writing the biases of each layer
    fid.write('\n[biases per layer]\n')

    # Input layer biases are set to zero
    fid.write("\t".join("%+.16e" % 0 for _ in controlling_vars) + "\n")

    for B in biases_for_output:
        fid.write("\t".join("%+.16e" % float(b) for b in B) + "\n")

    fid.close()
    return

def shoelace(XY:np.ndarray[float]):
    """Shoelace algorithm for area computations

    :param XY: hull node coordinates
    :type XY: np.ndarray[float]
    :return: area of concave hull
    :rtype: float
    """
    x = XY[:,0]
    y = XY[:,1]
    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))

    area = .5*np.absolute(S1 - S2)
    return area

    
def FiniteDifferenceDerivative(y:np.ndarray[float], x:np.ndarray[float]):
    """Calculate second-order accurate, one-dimensional finite-difference derivatives of y with respect to x.

    :param y: data to calculate the finite-differences for.
    :type y: np.ndarray[float]
    :param x: axial coordinates.
    :type x: np.ndarray[float]
    :return: finite-difference derivatives of y with respect to x.
    :rtype: np.ndarray[float]
    """
    Np = len(x)
    dydx = np.zeros(Np)
    for i in range(1, Np-1):
        y_m = y[i-1]
        y_p = y[i+1]
        y_0 = y[i]
        x_m = x[i-1]
        x_p = x[i+1]
        x_0 = x[i]
        dx_1 = x_p - x_0
        dx_2 = x_0 - x_m
        dx2_1 = dx_1*dx_1
        dx2_2 = dx_2*dx_2
        if (dx_1==0) or (dx_2==0):
            dydx[i] = 0.0
        else:
            dydx[i] = (dx2_2 * y_p + (dx2_1 - dx2_2)*y_0 - dx2_1*y_m)/(dx_1*dx_2*(dx_1+dx_2))
    dx_1 = x[1] - x[0]
    dx_2 = x[2] - x[0]
    dx2_1 = dx_1*dx_1
    dx2_2 = dx_2*dx_2
    y_0 = y[0]
    y_p = y[1]
    y_pp = y[2]
    if (dx_1==0) or (dx_2==0):
        dydx[0] = 0.0
    else:
        dydx[0] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))

    dx_1 = x[-2] - x[-1]
    dx_2 = x[-3] - x[-1]
    dx2_1 = dx_1*dx_1
    dx2_2 = dx_2*dx_2
    y_0 = y[-1]
    y_p = y[-2]
    y_pp = y[-3]
    if (dx_1==0) or (dx_2==0):
        dydx[-1] = 0.0
    else:
        dydx[-1] = (dx2_1 * y_pp + (dx2_2 - dx2_1)*y_0 - dx2_2*y_p)/(dx_1*dx_2*(dx_1 - dx_2))
    return dydx
