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

def avg_Le_const(from_flamelet_solution:np.ndarray, replacment_value:float):
    """Replace local solution of the species Lewis number with a constant value

    :param from_flamelet_solution: species Lewis number from flamelet solution
    :type from_flamelet_solution: np.ndarray
    :param replacment_value: constant species Lewis number
    :type replacment_value: float
    :return: array with constant species Lewis numbers
    :rtype: np.ndarray[float]
    """
    const_species_lewis_number = replacment_value * np.ones(np.shape(from_flamelet_solution))
    return const_species_lewis_number

class readStateDataFromFile:
    __dataset_filepath:str=""
    __controlling_variable_names:list[str]=[]
    __state_variable_names:list[str]=[]
    __dtype:type=np.float32
    __varnames_from_file:list[str]=None

    def __new__(self, dataset_filepath:str, controlling_variables:list[str], state_variables:list[str],dtype=np.float32):
        """Load data for a set of controlling variables and for a set of state variables from a csv file.

        :param dataset_filepath: _description_
        :type dataset_filepath: str
        :param controlling_variables: _description_
        :type controlling_variables: list[str]
        :param state_variables: _description_
        :type state_variables: list[str]
        :param dtype: _description_, defaults to np.float32
        :type dtype: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        self.__dataset_filepath=dataset_filepath
        self.__controlling_variable_names = controlling_variables.copy()
        self.__state_variable_names = state_variables.copy()
        self.__dtype=dtype
        
        self.__varnames_from_file = self.__getLabelsFromCSVFile(self)
        self.__checkIfVarInFile(self, self.__controlling_variable_names)
        self.__checkIfVarInFile(self, self.__state_variable_names)
        
        return self.__loadDataFromFile(self)
    
    def __getLabelsFromCSVFile(self):
        fid = open(self.__dataset_filepath, 'r')
        first_line = fid.readline()
        fid.close()
        line_split = first_line.strip().split(',')
        varnames = [varname.strip("\"").strip("\'") for varname in line_split]
        return varnames
    
    def __checkIfVarInFile(self, varnames_to_check:list[str]):
        unsupported_vars = []
        for var in varnames_to_check:
            if var not in self.__varnames_from_file:
                unsupported_vars.append(var)
        if any(unsupported_vars):
            raise Exception("The variables %s were not found in the file" % (",".join(var for var in unsupported_vars)))
        return
    
    def __loadDataFromFile(self):
        index_controlling_variables = [self.__varnames_from_file.index(v) for v in self.__controlling_variable_names]
        index_state_variables = [self.__varnames_from_file.index(v) for v in self.__state_variable_names]

        # Retrieve respective data from data set
        data_from_csv_file = np.loadtxt(self.__dataset_filepath, delimiter=',', skiprows=1, dtype=self.__dtype)
        if data_from_csv_file.shape[1] != len(self.__varnames_from_file):
            raise Exception("Number of labels is not equal to the number of data entries in %s" % self.__dataset_filepath)
        
        data_controlling_variables = data_from_csv_file[:, index_controlling_variables]
        data_state_variables = data_from_csv_file[:, index_state_variables]
        return data_controlling_variables, data_state_variables

    

def writeMLPForSU2(file_out:str, weights:list[np.ndarray], biases:list[np.ndarray],activation_function_name:str,train_vars:list[str], controlling_vars:list[str], scaler_function:str,scaler_function_vals_in:list[list[float]],scaler_function_vals_out:list[float],additional_header_info_function=None):
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