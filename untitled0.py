# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:41:20 2023

@author: Rahil
"""
import numpy as np
import pandas as pd
import operator
import os
from itertools import groupby
from operator import itemgetter
#from sympy import *
    




# function reads the descriptors and concentration files without the header and first column
def load_csv_data(descriptors_file_path , concentrations_file_path):
    try:
        # Load data from the first CSV file, exclude the header
        df1 = pd.read_csv(descriptors_file_path, sep=',')

        # Load data from the second CSV file, exclude the header
        df2 = pd.read_csv(concentrations_file_path , sep=',')

        # Convert the dataframes to numpy 2D arrays
        descriptors = df1.values
        concentrations = df2.values
        
        # exclude the first column 
        df1 = df1.iloc[:, 1:]
        df2 = df2.iloc[:, 1:]

        return descriptors, concentrations 

    except Exception as e:
        print("Error occurred while loading CSV data:", e)
        
# loads the concentration files  while keeping the header and exclude first column        
def load_csv_concentrations(concentrations_file_path):  
    # read concentration file including the header
    df = pd.read_csv(concentrations_file_path , sep=',' , header = None)  
    # exclude the first colum in concenteration matrix    
    df = df.iloc[:, 1:]
    concentrations = df.values
    return concentrations
    
# loads the descriptors files  while excluding the header and keeping the first column
def load_csv_descriptors(descriptors_file_path): 
    # read concentration file exclude the header
    df = pd.read_csv(descriptors_file_path, sep=',' )    
    descriptors = df.values
    return descriptors

        
        
# Returns the numpy array of header of the descriptors matrix (1, 2025) and the first column (18,1) of the concentration matrix    
def get_header_firstcoulmn (descriptors_file_path , concentrations_file_path):
     
    # Load data from the first CSV file 
    df1 = pd.read_csv(descriptors_file_path)
 
    # Load data from the second CSV file
    df2 = pd.read_csv(concentrations_file_path)
    
    # store header of descriptor names as list size of (num_descriptors +1)
    header_descriptor = df1.columns.tolist()
    #convert list to numpy array
    header_descriptor_ndarray = np.array (header_descriptor)
    
    # store mixture name column as pandas series
    first_column_mixture = df2.iloc[:, 0]
    
    # convert pandas series to numpy array 
    first_column_mixture_ndarray = first_column_mixture.values
    
    num_descriptors = df1.shape[1]
    num_mixture = df2.shape[0]     
 
    # reshape the header_descriptor_ndarray, first_column_mixture_ndarray  for ex from (2026, ) and (18, ) to (1, 2026) and (18, 1)
    header_descriptors_reshape = np.reshape(header_descriptor_ndarray, (1, num_descriptors ))
    column_mixtures_reshape = np.reshape(first_column_mixture_ndarray, (num_mixture, 1))

    
    return header_descriptors_reshape, column_mixtures_reshape      
  
# Sort concentrations matrix based on components
def sort_concentration (concentrations_file_path):
    # read concentration file including the header
    concentrations = load_csv_concentrations(concentrations_file_path)
    # sort concentrations matrix based on component name in first column
    sorted_concentrations = concentrations[:, np.argsort(concentrations[0])] 
    # exclude the first row (componnet name) of concentration matrix after sorting
    sorted_concentrations_ex  = sorted_concentrations [1:, :]
    sorted_concentrations_float = sorted_concentrations_ex.astype(np.float64)
    
    return sorted_concentrations_float
                      
# Sort descriptors matrix based on components                      
def sort_descriptors (descriptors_file_path) :
    
    # read concentration file exclude the header
    descriptors = load_csv_descriptors(descriptors_file_path)     
    # sort descriptors matrix based on component name in first column 
    sorted_descriptors = descriptors[np.argsort(descriptors[:, 0])]
    # exclude the first column (component name) of Descriptors matrix after sorting
    sorted_descriptors_ex = sorted_descriptors [:, 1:]
    sorted_descriptors_float = sorted_descriptors_ex.astype(np.float64)
    
    return sorted_descriptors_float

# Sort descriptors and concentrations matrices based on components  
def sort_components (descriptors_file_path , concentrations_file_path) :
     
    try:
        # read descriptors file exclude the header
        df1 = pd.read_csv(descriptors_file_path, sep=',' )
    
        # read concentration file including the header
        df2 = pd.read_csv(concentrations_file_path , sep=',' , header = None)  
    
        # exclude the first colum in concenteration matrix    
        df2_ex = df2.iloc[:, 1:]

        # Convert the dataframes to  numpy 2D arrays
        descriptors = df1.values
        concentrations = df2_ex.values
    
        # sort descriptors matrix based on component name in first column 
        sorted_descriptors = descriptors[np.argsort(descriptors[:, 0])]
    
        # exclude the first column (component name) of Descriptors matrix after sorting
        sorted_descriptors_ex = sorted_descriptors [:, 1:]
        # convert the numpy array of object to numpy array of float
        sorted_descriptors_float = sorted_descriptors_ex.astype(np.float64)
    
        # sort concentrations matrix based on component name in first column
        sorted_concentrations = concentrations[:, np.argsort(concentrations[0])] 
    
         # exclude the first row (componnet name) of concentration matrix after sorting
        sorted_concentrations_ex  = sorted_concentrations [1:, :]
         # convert the numpy array of object to numpy array of float
        sorted_concentrations_float = sorted_concentrations_ex.astype(np.float64)
    
        return sorted_descriptors_float, sorted_concentrations_float
    
    except Exception as e:
        print("Error occurred while loading CSV data:", e)


        
def write_matrices_to_csv(tabels_dict, output_path):
    
    average = tabels_dict["centroid"] 
    sqrdiff = tabels_dict["sqr_diff"]  
    absdiff = tabels_dict["abs_diff"]  
    fmolsum = tabels_dict["fmol_sum"]
    fmoldiff = tabels_dict["fmol_diff"] 
    sqrfmol = tabels_dict["sqr_fmol"] 
    rootfmol = tabels_dict["root_fmol"] 
    sqrfmolsum = tabels_dict["sqr_fmol_sum"] 
    normcont = tabels_dict["norm_cont"] 
    moldev = tabels_dict["mol_dev"] 
    sqrmoldev = tabels_dict["sqr_mol_dev"] 
    moldevsqr = tabels_dict["mol_dev_sqr"] 
    linearcombo = tabels_dict["linear_combination"]
    combinatorial = tabels_dict["combinatorial"]
    
    
    # convert the  numpy array to dataframe
    average_df = pd.DataFrame(average)
    sqrdiff_df = pd.DataFrame(sqrdiff)
    absdiff_df = pd.DataFrame(absdiff)
    fmolsum_df = pd.DataFrame(fmolsum)
    fmoldiff_df = pd.DataFrame(fmoldiff)
    sqrfmol_df = pd.DataFrame(sqrfmol)
    rootfmol_df = pd.DataFrame(rootfmol)
    sqrfmolsum_df = pd.DataFrame(sqrfmolsum)
    normcont_df = pd.DataFrame(normcont)
    moldev_df = pd.DataFrame(moldev)
    sqrmoldev_df = pd.DataFrame(sqrmoldev)
    moldevsqr_df = pd.DataFrame(moldevsqr)
    linearcombo_df = pd.DataFrame(linearcombo)
    combinatorial_df = pd.DataFrame(combinatorial)

    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Generate file names
        file_name1 = 'centroid.csv'
        file_name2 = 'sqr_diff.csv'
        file_name3 = 'abs_diff.csv'
        file_name4 = 'fmol_sum.csv'
        file_name5 = 'fmol_diff.csv'
        file_name6 = 'sqr_fmol.csv'
        file_name7 = 'root_fmol.csv'
        file_name8 = 'sqr_fmol_sum.csv'
        file_name9 = 'norm_cont.csv'
        file_name10 = 'mol_dev.csv'
        file_name11 = 'sqr_mol_dev.csv'
        file_name12 = 'mol_dev_sqr.csv'
        file_name13 = 'linear_combination.csv'
        file_name14 = 'combinatorial'

        # Save matrix1-12 as CSV
        file_path1 = os.path.join(output_path, file_name1)
        average_df.to_csv(file_path1, sep=',' , header=False, index=False)
      
        file_path2 = os.path.join(output_path, file_name2)
        sqrdiff_df.to_csv(file_path2, sep=',' , header=False, index=False)
        
        file_path3 = os.path.join(output_path, file_name3)
        absdiff_df.to_csv(file_path3, sep=',' , header=False, index=False)
        
        file_path4 = os.path.join(output_path, file_name4)
        fmolsum_df.to_csv(file_path4, sep=',' , header=False, index=False)
        
        file_path5 = os.path.join(output_path, file_name5)
        fmoldiff_df.to_csv(file_path5, sep=',' , header=False, index=False)
        
        file_path6 = os.path.join(output_path, file_name6)
        sqrfmol_df.to_csv(file_path6, sep=',' , header=False, index=False)

        file_path7 = os.path.join(output_path, file_name7)
        rootfmol_df.to_csv(file_path7, sep=',' , header=False, index=False)

        file_path8 = os.path.join(output_path, file_name8)
        sqrfmolsum_df.to_csv(file_path8, sep=',' , header=False, index=False)
        
        file_path9 = os.path.join(output_path, file_name9)
        normcont_df.to_csv(file_path9, sep=',' , header=False, index=False)
        
        file_path10 = os.path.join(output_path, file_name10)
        moldev_df.to_csv(file_path10, sep=',' , header=False, index=False)
        
        file_path11 = os.path.join(output_path, file_name11)
        sqrmoldev_df.to_csv(file_path11, sep=',' , header=False, index=False)
        
        file_path12 = os.path.join(output_path, file_name12)
        moldevsqr_df.to_csv(file_path12, sep=',' , header=False, index=False)
        
        file_path13 = os.path.join(output_path, file_name13)
        linearcombo_df.to_csv(file_path13, sep=',' , header=False, index=False)
        
        file_path14 = os.path.join(output_path, file_name14)
        combinatorial_df.to_csv(file_path14, sep=',' , header=False, index=False)

        file_path_dict = {
        'centroid': file_path1,
        'sqr_diff': file_path2,
        'abs_diff': file_path3,
        'fmol_sum': file_path4,
        'fmol_diff': file_path5,
        'sqr_fmol': file_path6,
        'root_fmol': file_path7,
        'sqr_fmol_sum': file_path8,
        'norm_cont': file_path9,
        'mol_dev': file_path10,
        'sqr_mol_dev': file_path11,
        'mol_dev_sqr': file_path12, 
        'linear_combination' :  file_path13,         
        'combinatorial' : file_path14
        }

        # Return the file paths
        return file_path_dict

    except Exception as e:
        print("Error occurred while writing matrices to CSV:", e)
          
        
        

def mask_concentration(concentrations):
    mask = np.where(concentrations == 0, 0, 1)
    return mask



def centroid(descriptors, concentrations): 
#    averages = []    
    num_mixtures = concentrations.shape[0]
#    num_descriptors = descriptors.shape[1]
    mask = mask_concentration(concentrations)
    print (mask.shape)
    sum_descriptors = np.dot(mask, descriptors)  
#    print ("sum descriptors ", sum_descriptors.shape)   # (18, 2025) 
# create an array of nonzero components  
    num_nonzero_components = np.sum(mask, axis = 1)
    print (num_nonzero_components.shape)
    num_nonzero_components_reshaped = np.reshape(num_nonzero_components, (num_mixtures, 1))
#    print ("num_nonzero_components",  num_nonzero_components.shape)
#    for i in range(num_mixtures):
#            num_nonzero_components = np.sum(mask[i])
#    average = sum_descriptors[:, np.newaxis, :] / num_nonzero_components[np.newaxis, :, :]          
#    num_nonzero_components_reshaped = np.squeeze(num_nonzero_components) 
#    num_nonzero_components_reshaped = np.reshape(num_nonzero_components_sq, (num_mixtures, num_descriptors))
    average = np.divide(sum_descriptors, num_nonzero_components_reshaped)   
#    averages_np = np.array(averages) 
       
    print ("average ", average.shape)
#    averages_csv = averages.to_csv(r'C:\Users\Rahil\Documents\File path\centroid.xlsx', index=False)
    return average


# return the difference of descriptors of componenets of each mixture  Da-Db
def diff (descriptors, concentrations): 
    num_mixtures = concentrations.shape[0]
    num_components = concentrations.shape[1]
    num_descriptors = descriptors.shape[1]
    mask = mask_concentration(concentrations)
    mult = mask[:, np.newaxis, :] * descriptors.T[np.newaxis, :, :]
    mult_reshape = np.reshape(mult, (num_mixtures, num_descriptors, num_components))
    diff = np.subtract.reduce(mult_reshape , axis= 2)
    #diffs = []
    #num_mixtures = concentrations.shape[0]
    #num_descriptors = descriptors.shape[1]
    #for i in range(num_mixtures ): 
        #for j in range(num_descriptors ):
            #concentration_value = concentrations[i, :]
            #descriptor_value = descriptors[:, j]
        #descriptors_values.append(descriptor_value)
        #column_values = descriptors[:, col_index][concentrations[:, col_index] != 0]
            #diff = np.subtract.reduce(descriptor_value [concentration_value != 0] , axis= 0)
            #diffs.append(diff)
    #diffs = np.reshape(diffs, (num_mixtures,num_descriptors))      
    return diff
    
    

def sqr_diff(descriptors, concentrations):   
    num_mixtures = concentrations.shape[0]
    num_descriptors = descriptors.shape[1] 
    diffs = diff (descriptors, concentrations)    
    squared_diff = np.square(diffs)
    squared_diff = np.reshape(squared_diff, (num_mixtures,num_descriptors))
    print ("squared_diff ", squared_diff.shape)
    #descriptors_averages = descriptors_averages.to_csvl('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//centroid.xlsx', index=False)
    return squared_diff 


def abs_diff(descriptors, concentrations):  
    num_mixtures = concentrations.shape[0]
    num_descriptors = descriptors.shape[1] 
    diffs = diff (descriptors, concentrations)    
    absolute_diff = np.abs (diffs)
    absolute_diff = np.reshape(absolute_diff, (num_mixtures,num_descriptors))
    print ("absolute_diff" ,absolute_diff.shape)
    #descriptors_averages = descriptors_averages.to_excel('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//centroid.xlsx', index=False)
    return absolute_diff
        

def fmol_sum(descriptors, concentrations):

    dot_product = np.dot(concentrations, descriptors)
    print ("dot_product" ,dot_product.shape)
    #dot_product = dot_product.to_csv('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//fmol_sum.xlsx', index=False)
    return dot_product


def fmol_diff (descriptors, concentrations):
    num_mixtures = concentrations.shape[0]
    num_components = concentrations.shape[1]
    num_descriptors = descriptors.shape[1] 
    mult = concentrations[:, np.newaxis, :] * descriptors.T[np.newaxis, :, :]
    mult_reshape = np.reshape(mult, (num_mixtures, num_descriptors, num_components))
    fmoldiff = np.subtract.reduce(mult_reshape , axis= 2)
    print("fmol_diff", fmoldiff.shape)
     #print(np.ndim (mult))
    #result = pd.DataFrame (result., A.T.reshape(2, -1), columns=cols)
    #result = result.to_csv('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//fmol_diff.xlsx', index=False)
    return fmoldiff



def sqr_fmol(descriptors, concentrations):
    squar_conc = np.square(concentrations)
    sqrfmol = np.dot(squar_conc, descriptors)
    print ("sqr_fmol", sqrfmol.shape)
    #dot_product = dot_product.to_csv('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//fmol_sum.xlsx', index=False)
    return sqrfmol



def root_fmol(descriptors, concentrations):
    fmol_root = np.sqrt(concentrations)
    fmolroot = np.dot(fmol_root, descriptors)
    print ("root_fmol", fmolroot.shape)
    #dot_product = dot_product.to_csv('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//fmol_sum.xlsx', index=False)
    return fmolroot




def sqr_fmol_sum (descriptors, concentrations):
    dot_product = np.dot(concentrations, descriptors)
    sqrfmolsum = np.square(dot_product)
    print ("sqr_fmol_sum", sqrfmolsum.shape)
    #dot_product = dot_product.to_excel('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//fmol_sum.xlsx', index=False)
    return sqrfmolsum


# Returns √((Xa Da)^2+ (Xb Db)^2 )
def norm_cont (descriptors, concentrations):
    num_mixtures = concentrations.shape[0]
    num_components = concentrations.shape[1]
    num_descriptors = descriptors.shape[1]
    mult = concentrations[:, np.newaxis, :] * descriptors.T[np.newaxis, :, :]
    mult_reshape = np.reshape(mult, (num_mixtures, num_descriptors, num_components))
    square_mult = np.square(mult_reshape)
    sum_square_mult = np.sum(square_mult , axis= 2)
    normcont = np.sqrt(sum_square_mult)
    print ("normcont", normcont.shape)
    return normcont


# return the difference of the components concentration of each mixture  Xa -Xb
def diff_concentration (concentrations):
    diff = np.subtract.reduce(concentrations , axis= 1)
    return diff

# returns | Da - Db| * [1- | (Xa -Xb)]  
def mol_dev (descriptors, concentrations):
    abs_diff_desc = abs_diff (descriptors, concentrations)
    diff_concent = diff_concentration (concentrations)
    abs_diff_conc = 1- (np.abs(diff_concent))
    moldev = abs_diff_desc.T * abs_diff_conc
    moldev = moldev.T
    print ("moldev", moldev.shape)
    return moldev
    

# returns | Da - Db| * [1- | (Xa^2 -Xb^2)]  
def sq_mol_dev (descriptors, concentrations):
    abs_diff_desc = abs_diff (descriptors, concentrations)
    square_conc = np.square(concentrations)
    diff_concent = diff_concentration (square_conc)
    abs_diff_conc_subtract1 = 1- (np.abs(diff_concent))    
    sqrmoldev = abs_diff_desc.T * abs_diff_conc_subtract1
    sqrmoldev = sqrmoldev.T
    print ("sqrmoldev", sqrmoldev.shape)
    return sqrmoldev
    


# returns | Da - Db| * [1- | (Xa -Xb)]^2  
def mol_dev_sqr (descriptors, concentrations):
    abs_diff_desc = abs_diff (descriptors, concentrations)
    diff_concent = diff_concentration (concentrations)
    abs_diff_conc_subtract1 = 1- (np.abs(diff_concent))
    square_conc = np.square(abs_diff_conc_subtract1)
    moldevsqr = abs_diff_desc.T * square_conc
    moldevsqr = moldevsqr.T
    print (" sqrmoldev", moldevsqr.shape)
    return moldevsqr    

# returns the following mixture discriptors for each mixture [X1*D1+ X2*D2+ X3*D3+ X4*D4,    X1*D2+ X2*D3+ X3*D4+ X4*D1,    X1*D3+X2*D4+ X3*D1+ X3*D2]
def linear_combiantion (descriptors, concentrations):
    
    num_components = descriptors.shape[0]
    num_descriptors = descriptors.shape[1] 
    num_element_maindiagonal = min(num_components, num_descriptors) 
    # create the list of arrays that includes the diagonals of the descriptors matrix
    diagonals = [descriptors.diagonal(offset= i) for i in range( - num_element_maindiagonal + 1, num_descriptors)]
    
    # main diagonals list includes the  subarrays of diagonals that have the same length as main diagonals( which is equal to  min(num_components, num_descriptors))
    main_diagonals =[]
    
    # ragged_matrix_lower list includes the subarrays of diagonals that are  smaller then main diagpnal in lower triangle
    ragged_matrix_lower  = []
    
    # ragged_matrix_lower list includes the subarrays of diagonals that are  smaller then main diagpnal in upper triangle
    ragged_matrix_upper  = []
    
    # populate 3 list of arrays including 1- main_diagonals list  2-ragged_matrix_upper  and 3 ragged_matrix_lower
    for index , subarr in enumerate(diagonals):
        # if subarry is not empty
        if subarr.size > 0:
            
            # if index of the diagonals list of arrays will be less than the number of the component -1 (which is the diagonal number in lower triangles)
            if index < num_components -1:
                ragged_matrix_lower.append(subarr)
            # if the subarray size is equal to the number of the main diagonal (which is equal to smallest dimension of the descriptors matrix)       
            if subarr.size == num_element_maindiagonal:
                main_diagonals.append(subarr)    
             # if the size of sub array is less than the number of the main diagonal and index is in the last num_descriptors -1 element of the lists 
            if  subarr.size < num_element_maindiagonal and index > (len(diagonals)- (num_descriptors -1) ):
                ragged_matrix_upper.append(subarr) 
                
        else:
            print("subarray is empty")
            
     # Concatenate ragged_matrix_upper and ragged_matrix_lower list of arrays
    concat_up_low = []
    for arr1, arr2 in zip(ragged_matrix_upper, ragged_matrix_lower):
        concatenated = np.concatenate((arr1, arr2))
        concat_up_low.append(concatenated)
        
    # Convert two list of main_diagonals and concat_up_low_ to numpy array    
    main_diagonals_arr= np.array(main_diagonals)  
    concat_up_low_arr= np.array(concat_up_low)
    
    # Transpose main_diagonals_arr and concat_up_low_arr and then concatenate them
    concat_horizontal = np.concatenate((main_diagonals_arr.T, concat_up_low_arr.T), axis=1)
    
    # dot product of the concentrations and concat_horizontal
    lin_combo = np.dot(concentrations, concat_horizontal)
    #print("lincombo", lin_combo.shape)
    
    return lin_combo


# Return the numpy array of cartesian product of  descriptors 
def cartesian_product (descriptors): 

    # exclude the first column
    descriptors = descriptors [:, 1:]
    # Create all possible combinations of elements using cartesian product
    cart_product = pd.MultiIndex.from_product(descriptors).tolist()
    cart_product_ndarr= np.array(cart_product).T
    return cart_product_ndarr

#Return the combinatorial of diescriptors 
#  calculate [X1D1+X2D2 , X1D1+X2D2,  X1D1+X2D3, X1D2+X2D1 , X1D2+X2D2 , X1D2+X2D3, X1D3+X2D1, X1D3+X2D2, X1D3+X2D3] for 2 component (X1, X2) and 3 descriptors (D1, D2, D3)  for each mixture.      
def combinatorial (descriptors, concentrations):
    
    # exclude the header 
    concentrations = concentrations[1:, :]
    # convert string to float
    concentrations = concentrations.astype(np.float64)
    # run the cartesian_product function
    cart_ndarr = cartesian_product (descriptors)  
    # convert string to float
    cart_ndarr= cart_ndarr.astype(np.float64)
    # dot product of concentrations numpuy array with numpy array of cartisian product
    mult = np.dot(concentrations, cart_ndarr)
    return mult

# Return the combination of the descriptors as header 
def combinatorial_descriptors(descriptors, concentrations ):

    # Convert numpy array to dataframe
    df = pd.DataFrame (descriptors)
    
    # exclude the first column
    df = df.iloc[:, 1:]
    num_components= df.shape[0]
    
    # store header of descriptor names as list size of (num_descriptors)
    header_descriptor = df.columns.tolist()
    
    # convert dataframe to numpy array
    header_descriptor = np.array(header_descriptor)
    
    # reapeat the header_descriptor vector num_components times in row axis
    header_descriptor = np.repeat(header_descriptor[np.newaxis, :], num_components, axis=0)
    
    # convert numpy array to dataframe 
    header_descriptor_df = pd.DataFrame (header_descriptor)
   
    # combiantion of the strings in header_descriptor   
    header_descriptor_product = pd.MultiIndex.from_product(header_descriptor_df.values).tolist()

    # convert list to numpy array
    combination_descriptors = np.array (header_descriptor_product)
    
    # apply a lambda function in each row of amtrix which join each row into a single comma-separated string.
    combination_descriptors = np.apply_along_axis(lambda row: ','.join(row), axis=1, arr=combination_descriptors)
    
    # reshape the (len(header_descriptor_product) , ) to ( 1, len(header_descriptor_product))
    combination_descriptors = combination_descriptors.reshape(1, len(header_descriptor_product) )
    
    # run the combinatorial functioin 
    combination = combinatorial (descriptors, concentrations)
    
    # concatenate the  combination_descriptors to combinatorial  
    combinatorialheader= np.concatenate((combination_descriptors, combination), axis = 0)
    
    return combinatorialheader



def mixture_descriptors (descriptors_file_path , concentrations_file_path):
    
    descriptors = load_csv_descriptors(descriptors_file_path)
    concentrations  = load_csv_concentrations(concentrations_file_path)
    #concentrations = sort_concentration (concentrations_file_path)
    #descriptors = sort_descriptors (descriptors_file_path)
    average = centroid(descriptors, concentrations)
    sqrdiff = sqr_diff(descriptors, concentrations)
    absdiff = abs_diff(descriptors, concentrations)
    fmolsum = fmol_sum(descriptors, concentrations)
    fmoldiff = fmol_diff (descriptors, concentrations)
    sqrfmol = sqr_fmol(descriptors, concentrations)
    rootfmol = root_fmol(descriptors, concentrations)
    sqrfmolsum = sqr_fmol_sum (descriptors, concentrations)
    normcont = norm_cont (descriptors, concentrations)
    moldev = mol_dev (descriptors, concentrations)
    sqrmoldev = sq_mol_dev (descriptors, concentrations)
    moldevsqr= mol_dev_sqr (descriptors, concentrations)
    #lin_combo = linear_combiantion (descriptors, concentrations)
    combinatorial = combinatorial_descriptors(descriptors, concentrations )
    
    
    # create the dictionary
    tabels_dict = {
        'centroid': np.array(average),
        'sqr_diff': np.array(sqrdiff),
        'abs_diff': np.array(absdiff),
        'fmol_sum': np.array(fmolsum),
        'fmol_diff': np.array(fmoldiff),
        'sqr_fmol': np.array(sqrfmol),
        'root_fmol': np.array(rootfmol),
        'sqr_fmol_sum': np.array(sqrfmolsum),
        'norm_cont': np.array(normcont),
        'mol_dev': np.array(moldev),
        'sqr_mol_dev': np.array(sqrmoldev),
        'mol_dev_sqr': np.array(moldevsqr), 
        #'linear_combination' : np.array(lin_combo)      
        'combinatorial' : np.array(combinatorial)
        }
    
    return tabels_dict 

# This function run mixture_descriptors function, gets theader_descriptors, column_mixtures, and concatenate to the output csv files
# Then write_matrices_to_csv function,    
# This function retun the dictionary of 13  mixture descriptores csv files output path 
def mixture_descriptors_to_csv (descriptors_file_path , concentrations_file_path, output_path ):
    
    tabels_dict = mixture_descriptors (descriptors_file_path , concentrations_file_path)
    header_descriptors, column_mixtures = get_header_firstcoulmn (descriptors_file_path , concentrations_file_path)

    
    # Add the header_descriptors, column_mixtures to the first row and first column of all dataframes in the dictionary
    # create new dictionary to store concatenated ones
    concatenated_dict = {}
    
    for key, arr in tabels_dict.items():
        
        # Concatenate the column to the beginning of the DataFrame
        concatenated = np.concatenate((column_mixtures, arr), axis=1)
        
        if key is not 'combinatorial':
        # Concatenate the header for ex (1, num_descriptor+1) to the beginning of the DataFrame of (num_mixtures, num_descriptor+1), to get matrix of ex (num_mixtures +1, num_descriptor+1)
            concatenated = np.concatenate((header_descriptors, concatenated), axis=0)

        concatenated_dict[key] = concatenated
    
    output_path_dict= write_matrices_to_csv(concatenated_dict, output_path)

    return output_path_dict






# Return two array of dictionaries from concentration and descriptors, ordered_dict_concentrations is ordered based on MW and oncentration amount
# dict_descripors is ordered only based on MW      
def component_order (descriptors, concentrations):
    #df1= pd.read_csv(descriptors_file_path , sep=',' , header = None)  
    #df2 = pd.read_csv(concentrations_file_path , sep=',' , header = None)  
    #descriptors = df1.values
    #concentrations = df2.values
    
    # Sort Descriotors based on the MW (descending)
    column_index = np.where(descriptors[0] == 'MW')[0][0]
    #  argsort() on the corresponding column values (matrix[1:, column_index]) obtains the indices that would sort the column in ascending order.
    #  [::-1] gets the descending order
    # Adding 1 to the resulting indices since we exclude the first row
    sorted_descriptors = descriptors[descriptors[1:,  column_index].argsort() + 1][::-1]
    
    # exclude the first column of concentration matrix 
    concentrations = concentrations[:, 1:]
    # get the indices of the concentration matrix first row based on the equivalent elements' indices in sorted_descriptors matrix first column
    column_order =  np.where(concentrations[0, :] [:, np.newaxis] == sorted_descriptors[:, 0])[1]
    # Sort Concentrations rows based on the components in the sorted descriptors
    sorted_concentrations = concentrations[:,  column_order ]   
    
    # create an array of dictionaries form the sorted_concentrations numpy array
    #list comprehension [...] iterates over each row in sorted_concentrations[1:]. 
    #For each row, the dictionary comprehension {...} creates a dictionary where the keys come from the first row and the values come from the corresponding elements in the current row.
    dictionary_concent =  [{col: float(val) for col, val in zip(sorted_concentrations[0, :], row[:])} for row in sorted_concentrations[1:]]
    
    # Sort each dictinary of dictionary_concent based on the value descending
    # list comprehension [...] iterates over each dictionary and sorted() function sorts its key-value pairs based on the values. 
    #The key=operator.itemgetter(1) specifies that sorting should be done based on the second element of each item (the values).
    ordered_dict_concentrations = [dict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)) for dictionary in dictionary_concent]

    # create an array of dictionaries form the sorted_descriptors
    #list comprehension [...] iterates over each column in sorted_descriptors[:, 1:].T.
    #For each column, the dictionary comprehension {...} creates a dictionary where the keys come from the first element of each row and the values come from the corresponding elements in the current column.
    dict_descripors = [{row[0]: float(val) for val, row in zip(values, sorted_descriptors[:])} for values in sorted_descriptors[:, 1:].T]
   
    return ordered_dict_concentrations, dict_descripors


# output a 3D array of the descriptors with the size of depth of num_mixtures, each 2D matrix of tensor has the same order as component order of each mixture
# also convert the values of ordered_dict_concentrations to array and output that  
# this function mean to capture the order of component for each mixture in 3D tensor    
def ordered_3Ddesc_2Dconcent (descriptors, concentrations):
    
    ordered_dict_concentrations, dict_descripors = component_order (descriptors, concentrations)
                      
    ##sorted_dict_descripors =[[{key: dict1[key] for key in dict2.keys()}  for dict2 in ordered_dict_concentrations]
         ##for dict1 in dict_descripors]  
         
    # iterates over each dictionary in ordered_dict_concentrations and retrieves the keys in the order they appear then 
    #  for each dictionary in dict_descripors, it creates a new dictionary with the same keys and corresponding values and append the array of ordered dictionaries to the sorted_dict_descripors.
    # create array of arrays of dictionaries from the dict_descripors which each array would be based on the order of keys in each dictionary of ordered_dict_concentrations
    sorted_dict_descripors = []
    for dict2 in ordered_dict_concentrations:
        ordered_dicts = []
        keys = list(dict2.keys())
        for dict1 in dict_descripors:
            ordered_dict = {key: dict1[key] for key in keys}
            ordered_dicts.append(ordered_dict)
        sorted_dict_descripors.append(ordered_dicts)
        
        #Convert the array of arrays of dictionaries to a 3D array
        descriptor_tensor = [[[value for value in dictionary.values()] for dictionary in subarray] for subarray in sorted_dict_descripors]
        
        # convert the values of ordered_dict_concentrations to 3D array
        concentrations_array = [[value for value in dictionary.values()] for dictionary in ordered_dict_concentrations] 
          
    return concentrations_array, descriptor_tensor

 
    


# Example usage
    
desc_file_path = r'C:\Users\Rahil\Documents\File path\Des.csv'
concent_file_path = r'C:\Users\Rahil\Documents\File path\Conc.csv'
outputpath = r'C:\Users\Rahil\Documents\File path'

#desc, conc=  sort_components (desc_file_path , concent_file_path)
#print(desc, conc)

#dic = mixture_descriptors_to_csv(desc_file_path, concent_file_path, outputpath)
#print (dic)


descriptors = np.array([['descriptor','MW', 'AMW', 'sv'],
                    ['a22', 7, 5, 1],
                    ['b31', 3, 6, 4],
                    ['c25', 5, 9, 2], 
                    ['d18', 4, 3, 11]])

concentrations = np.array([['Mix', 'b31', 'c25', 'd18', 'a22'],
                           ['A', 0.4, 0.2, 0.4, 0],
                           ['B', 0.7, 0, 0.2, 0.1],
                           ['C', 0.5, 0, 0, 0.5]])
    
 
s = ordered_3Ddesc_2Dconcent (descriptors, concentrations)   
print(s) 
#perm = cartesian_product (matrix1)
#car = combinatorial (desc_file_path, concent_file_path)
#header = combinatorial_descriptors(desc_file_path, car )
#print(perm)
#print(car)
#print (header)
#print(perm) 
#print (car)   
#file1_path = r'C:\Users\Rahil\Documents\File path\cor.csv'
#file2_path =  r'C:\Users\Rahil\Documents\File path\correlation_pw_noglm.csv'   

#diff = diff (matrix1, matrix2) 
#diff_abs = abs_diff(matrix1, matrix2)  
#diff_sqr = sqr_diff(matrix1, matrix2)  
#average = centroid(matrix1, matrix2) 
#print (type(average))
#fsum = fmol_sum( matrix1, matrix2)
#fdiff = fmol_diff( matrix1, matrix2)
#sqrfmol = sqr_fmol( matrix1, matrix2)
#fmolroot = root_fmol( matrix1, matrix2)
#sqrfmolsum = sqr_fmol_sum( matrix1, matrix2)
#normcont = norm_cont( matrix1, matrix2)
#diffconcet = diff_concentration( matrix2)
#moldev = mol_dev( matrix1, matrix2)
#sqrmoldev = sq_mol_dev( matrix1, matrix2)
#moldevsqr = mol_dev_sqr( matrix1, matrix2)
#load = load_csv_data(file1_path, file2_path)
#print(load) 
#print(moldevsqr) 
#print(sqrmoldev) 
#print(moldev) 
#print(diffconcet)    
#print(normcont)
#print(sqrfmolsum)
#print(fmolroot)
#print(sqrfmol)    
#print(fdiff)
#print(fsum) 
#print("Column average:", average)
#print("Column diff_abs:", diff_abs)
#print("diff_sqr:", diff_sqr)
#print("diff:", diff)


#################################################################################################


#concent_df = pd.read_excel('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//Concentrations.xlsx',  sheet_name='Sheet2') # pandas dataframe
#descrip_df = pd.read_excel('C://Users//Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//Descriptors.xlsx',  sheet_name='Sheet2') # pandas dataframe
#concentrations = concent_df.values  # convert datafarame to numpy array
#descriptors= descrip_df.values  # convert datafarame to numpy array
    

#def create_mask_matrix(matrix):
#    mask = np.where(matrix == 0, 0, 1)
#    return mask
    
#def centroid(descriptors, concentrations):
#    averages = []
#    num_columns = descriptors.shape[1]
    
#    for col_index in range(num_columns):
#        column_values = descriptors[:, col_index][concentrations[:, col_index] != 0]
#        average = np.mean(column_values)
#        averages.append(average)
    #descriptors_averages = descriptors_averages.to_excel('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//centroid.xlsx', index=False)
#    return averages


#def sqr_diff(descriptors, concentrations):
#     diffs = []
#     num_columns = descriptors.shape[1]
  
#     for col_index in range(num_columns):
#       column_values = descriptors[:, col_index][concentrations[:, col_index] != 0]
#       diff = np.subtract.reduce(column_values)        
#        diffs.append(diff)
   
#    squared_diff = np.square(diffs)
#    descriptors_averages = descriptors_averages.to_excel('C://Users/Rahil//OneDrive - North Dakota University System//CSCI Course//Summer 2023//ChemInformatic//centroid.xlsx', index=False)
#     return squared_diff





