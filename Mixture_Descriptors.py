
import pandas as pd
import numpy as np
import itertools
import os


def load_csv_data(descriptors_file_path , concentrations_file_path):
    
    try:
        # Load data from the first CSV file which exclude the header 
        df1 = pd.read_csv(descriptors_file_path, sep=',' )

        # Load data from the second CSV file which exclude the header
        df2 = pd.read_csv(concentrations_file_path, sep=',')
        
        # exclude the first column 
        df1 = df1.iloc[:, 1:]
        df2 = df2.iloc[:, 1:]

        # Convert the dataframes to  numpy 2D arrays
        descriptors = df1.values
        concentrations = df2.values

        return descriptors, concentrations 

    except Exception as e:
        print("Error occurred while loading CSV data:", e)
        
# load data from the CSV concentrations file 
def load_csv_concentration(concentrations_file_path):  
    # read concentration file including the header
    df = pd.read_csv(concentrations_file_path , sep=',' , header = None)  
    # exclude the first colum in concenteration matrix    
    df = df.iloc[:, 1:]
    concentrations = df.values
    return concentrations
    
# load data from the CSV descriptors file 
def load_csv_descriptors(descriptors_file_path): 
    # read concentration file exclude the header
    df = pd.read_csv(descriptors_file_path, sep=',' )    
    descriptors = df.values
    return descriptors

# Sort concentrations matrix based on components
def sort_concentration (concentrations_file_path):
    # read concentration file including the header
    concentrations = load_csv_concentration(concentrations_file_path)
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
    

# Returns the numpy array of header of the descriptors matrix (1, num_descriptors) and the first column (num_mixtures,1) of the concentration matrix     
def get_header_firstcolumn (descriptors_file_path , concentrations_file_path):
    
    try:
        # Load data from the first CSV file as pandas dataframe
        df1 = pd.read_csv(descriptors_file_path)
 
        # Load data from the second CSV file as pandas dataframe
        df2 = pd.read_csv(concentrations_file_path)
    
        # store header of descriptor names as list size of (num_descriptors + 1)
        header_descriptor = df1.columns.tolist()
        #convert list to numpy array
        header_descriptor_ndarray = np.array (header_descriptor)
    
        # store mixture name column as pandas series
        first_column_mixture = df2.iloc[:, 0]
    
        # convert pandas series to numpy array 
        first_column_mixture_ndarray = first_column_mixture.values
    
        num_descriptors = df1.shape[1]
        num_mixture = df2.shape[0]     
 
        # reshape the header_descriptor_ndarray, first_column_mixture_ndarray from (num_descriptors + 1, ) and (num_mixtures, ) to (1, num_descriptors + 1) and (num_mixtures, 1)
        header_descriptors_reshape = np.reshape(header_descriptor_ndarray, (1, num_descriptors ))
        column_mixtures_reshape = np.reshape(first_column_mixture_ndarray, (num_mixture, 1))
    
        return header_descriptors_reshape, column_mixtures_reshape
    
    except Exception as e:
        print("Error occurred while loading CSV data:", e)        
        
def permutation (descriptors):        
    # Get the indices for creating combinations of elements in each row
    indices = np.indices(descriptors.shape).reshape(descriptors.ndim, -1).T

    # Create all possible combinations of elements from each row
    permutations = np.array(np.meshgrid(descriptors[indices[:, 0], indices[:, 1]]))
    return permutations


     
# This function, get the dictionary of 13 matrices, and return the dictionary of 13 csv files path       
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
    
    # convert the numpy array to dataframe
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

        # Save matrix1 to matrix 13 as CSV
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
        'linear_combination' : file_path13
        }

        # Return the file paths dictionary
        return file_path_dict

    except Exception as e:
        print("Error occurred while writing matrices to CSV:", e)


#  return the matrix (numpy array) which has 0 for any zero value and 1 for non-szero values of the input matrix
def mask_concentration(concentrations):
    
    mask = np.where(concentrations == 0, 0, 1)
    
    return mask


#  return the matrix (numpy array) of differences of descriptors of all components (nonzero values) of each mixture  (Da-Db)
def diff (descriptors, concentrations): 
    
    num_mixtures = concentrations.shape[0]
    num_components = concentrations.shape[1]
    num_descriptors = descriptors.shape[1]
    mask = mask_concentration(concentrations)
    mult = mask[:, np.newaxis, :] * descriptors.T[np.newaxis, :, :]
    mult_reshape = np.reshape(mult, (num_mixtures, num_descriptors, num_components))
    diff = np.subtract.reduce(mult_reshape , axis= 2) 
    
    return diff

# Return the mean ( (Da + Db)/ 2 ) of all pure components descriptors of each mixtures
def centroid(descriptors, concentrations):
    
    num_mixtures = concentrations.shape[0]
    mask = mask_concentration(concentrations)
    sum_descriptors = np.dot(mask, descriptors)   
    num_nonzero_components = np.sum(mask, axis = 1)
    # In order to broadcast, one of the dimension should be 1, then reshape the  (num_mixtures, ) to (num_mixtures, 1) dimension 
    num_nonzero_components_reshaped = np.reshape(num_nonzero_components, (num_mixtures, 1))
    average = np.divide(sum_descriptors, num_nonzero_components_reshaped)  
    
    return average


# Return the square of difference of all  descriptors of pure components of each mixtures ( (Da - Db)^2)
def sqr_diff(descriptors, concentrations): 
    
    num_mixtures = concentrations.shape[0]
    num_descriptors = descriptors.shape[1] 
    diffs = diff (descriptors, concentrations)    
    squared_diff = np.square(diffs)
    squared_diff = np.reshape(squared_diff, (num_mixtures,num_descriptors))
    
    return squared_diff     

# Return the absolute of difference of all descriptors of pure components of each mixtures (|Da - Db|)
def abs_diff(descriptors, concentrations): 
    
    num_mixtures = concentrations.shape[0]
    num_descriptors = descriptors.shape[1] 
    diffs = diff (descriptors, concentrations)    
    absolute_diff = np.abs (diffs)
    absolute_diff = np.reshape(absolute_diff, (num_mixtures,num_descriptors))
    
    return absolute_diff
        

# Return the sum of weighted descriptors of pure components of each mixtures (Xa*Da + Xb*Db)
def fmol_sum(descriptors, concentrations):
    
    dot_product = np.dot(concentrations, descriptors)
    
    return dot_product

# Return the difference of weighted descriptors of pure components of each mixtures by mol fractions  (Xa*Da - Xb*Db)
def fmol_diff (descriptors, concentrations):
    
    num_mixtures = concentrations.shape[0]
    num_components = concentrations.shape[1]
    num_descriptors = descriptors.shape[1] 
    mult = concentrations[:, np.newaxis, :] * descriptors.T[np.newaxis, :, :]
    mult_reshape = np.reshape(mult, (num_mixtures, num_descriptors, num_components))
    fdiff = np.subtract.reduce(mult_reshape , axis= 2)
    
    return fdiff


# Return the  sum  of weighted descriptors of pure components of each mixtures by square of mol fractions (Xa^2*Da + Xb^2*Db)
def sqr_fmol(descriptors, concentrations):
    
    squar_conc = np.square(concentrations)
    sqrfmol = np.dot(squar_conc, descriptors)
    
    return sqrfmol


# Return the sum  of weighted descriptors of pure components of each mixtures by root of mol fractions (√Xa*Da + √Xb*Db)
def root_fmol(descriptors, concentrations):
    
    fmol_root = np.sqrt(concentrations)
    fmolroot = np.dot(fmol_root, descriptors)
    
    return fmolroot

# Return the square sum  of weighted descriptors of pure components of each mixtures by mol fractions (Xa*Da + Xb*Db)^2
def sqr_fmol_sum (descriptors, concentrations):
    
    dot_product = np.dot(concentrations, descriptors)
    sqrfmolsum = np.square(dot_product)
    
    return sqrfmolsum

# returns √((Xa*Da)^2+ (Xb*Db)^2 )
def norm_cont (descriptors, concentrations):
    
    num_mixtures = concentrations.shape[0]
    num_components = concentrations.shape[1]
    num_descriptors = descriptors.shape[1]
    mult = concentrations[:, np.newaxis, :] * descriptors.T[np.newaxis, :, :]
    mult_reshape = np.reshape(mult, (num_mixtures, num_descriptors, num_components))
    square_mult = np.square(mult_reshape)
    sum_square_mult = np.sum(square_mult , axis= 2)
    normcont = np.sqrt(sum_square_mult)
    
    return normcont


# return the difference of the pure components concentration of each mixture  (Xa -Xb)
def diff_concentration (concentrations):
    
    diff = np.subtract.reduce(concentrations , axis= 1)
    
    return diff

# return | Da - Db| * [1- | (Xa -Xb)]  
def mol_dev (descriptors, concentrations):
    
    abs_diff_desc = abs_diff (descriptors, concentrations)
    diff_concent = diff_concentration (concentrations)
    abs_diff_conc = 1- (np.abs(diff_concent))
    moldev = abs_diff_desc.T * abs_diff_conc
    moldev = moldev.T
    
    return moldev
    

# returns |Da - Db| * [1-|(Xa^2 -Xb^2)|)]  
def sq_mol_dev (descriptors, concentrations):
    
    square_conc = np.square(concentrations)
    diff_concent = diff_concentration (square_conc)
    abs_diff_conc = 1- (np.abs(diff_concent))
    abs_diff_desc = abs_diff (descriptors, concentrations)
    sqrmoldev = abs_diff_desc.T * abs_diff_conc
    sqrmoldev = sqrmoldev.T
    
    return sqrmoldev


# returns |Da - Db| * [1- |(Xa -Xb)|]^2  
def mol_dev_sqr (descriptors, concentrations):
    
    abs_diff_desc = abs_diff (descriptors, concentrations)
    diff_concent = diff_concentration (concentrations)
    abs_diff_conc_subtract1 = 1- (np.abs(diff_concent))
    square_conc = np.square(abs_diff_conc_subtract1)
    moldevsqr = abs_diff_desc.T * square_conc
    moldevsqr = moldevsqr.T
    
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
        
    # Convert two list of main_diagonals and concat_up_low_arr to numpy array    
    main_diagonals_arr= np.array(main_diagonals)  
    concat_up_low_arr= np.array(concat_up_low)
    
    # Transpose main_diagonals_arr and concat_up_low_arr and then concatenate them
    concat_horizontal = np.concatenate((main_diagonals_arr.T, concat_up_low_arr.T), axis=1)
    
    # dot product of the concentrations and concat_horizontal
    lin_combo = np.dot(concentrations, concat_horizontal)
    
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
    combinatorial_header= np.concatenate((combination_descriptors, combination), axis = 0)
    
    return combinatorial_header
         

# This function load the input files and run all the formulation of mixture descriptors 
# This function retun the dictionary of 13 mixture descriptors numpy arrays   
def mixture_descriptors (descriptors_file_path , concentrations_file_path):
    
    #descriptors, concentrations  = load_csv_data (descriptors_file_path , concentrations_file_path)
    concentrations = sort_concentration (concentrations_file_path)
    descriptors = sort_descriptors (descriptors_file_path)
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
    lin_combo = linear_combiantion (descriptors, concentrations)
    
    tabels_dict = {
        'centroid': average,
        'sqr_diff': sqrdiff,
        'abs_diff': absdiff,
        'fmol_sum': fmolsum,
        'fmol_diff': fmoldiff,
        'sqr_fmol': sqrfmol,
        'root_fmol': rootfmol,
        'sqr_fmol_sum': sqrfmolsum,
        'norm_cont': normcont,
        'mol_dev': moldev,
        'sqr_mol_dev': sqrmoldev,
        'mol_dev_sqr': moldevsqr, 
        'linear_combination' : lin_combo
        }
    
    return  tabels_dict 


# This function run mixture_descriptors function, gets header_descriptors, column_mixtures, and concatenate them to the output csv files
# Then run the write_matrices_to_csv function,    
# Retun the dictionary of 13  mixture descriptores csv files output path  
def mixture_descriptors_to_csv (descriptors_file_path , concentrations_file_path, output_path ):
    
    tabels_dict = mixture_descriptors (descriptors_file_path , concentrations_file_path)
    header_descriptors, column_mixtures = get_header_firstcolumn (descriptors_file_path , concentrations_file_path)
    
    # Add the header_descriptors, column_mixtures to the first row and first column of all dataframes in the dictionary
    # create new dictionary to store concatenated ones
    concatenated_dict = {}
    
    for key, arr in tabels_dict.items():

        # Concatenate the column_mixtures to the begining of the DataFrame
        concatenated = np.concatenate((column_mixtures, arr), axis=1)
        
        # Concatenate the header_descriptors (1, num_descriptor + 1) to the begining of the DataFrame of (num_mixtures, num_descriptor + 1), to get matrix of (num_mixtures + 1, num_descriptor + 1)
        concatenated = np.concatenate((header_descriptors, concatenated), axis=0)

        concatenated_dict[key] = concatenated
    
    output_path_dict= write_matrices_to_csv(concatenated_dict, output_path)
    
    return output_path_dict


###################################################################################################################################
# Example usage
# Change the file path of descriptors and concentrations files and output files directory accordingly 
# Make sure the components name order are the same in both files    

desc_file_path = r'C:\Users\Rahil\Documents\File path\Descriptors.csv'
concent_file_path = r'C:\Users\Rahil\Documents\File path\Concentrations.csv'
output_path = r'C:\Users\Rahil\Documents\File path'

result = mixture_descriptors_to_csv(desc_file_path, concent_file_path, output_path)
print (result)



