# run.py
from  Mixture-Descriptors import mixture_descriptors_to_csv


if __name__ == '__main__':  
    desc_file_path = r'Descriptors.csv'
    concent_file_path = r'Components.csv'
    output_path = r'Output'
    
    
    result = mixture_descriptors_to_csv(desc_file_path, concent_file_path, output_path)
    print (result)


