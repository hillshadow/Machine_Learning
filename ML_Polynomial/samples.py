import math
import random
import numpy as np

# in order to produce the data for  training and  test
class Samples:
    def __init__(self,sample_file,test_file,size):
        self.sample_file = sample_file
        self.test_file = test_file
        self.size = size

    def calculate_data(self):
        temp_number_sample = 0
        temp_number_test = 0
        f = open(self.sample_file,"w")
        for i in range(0,self.size):
            temp_number_sample = temp_number_sample+0.1*random.random() # To produce sorted data
            temp_noise = random.gauss(0.5,1) # To add the Gauss noise to the data
            # Put the data into the file in a standard font
            f.write("%.3f" %  temp_number_sample+ ":   " + "%.3f\n" % (math.sin(2*math.pi* temp_number_sample+0.5)+temp_noise))
        f.close()
        f_test = open(self.test_file,"w")
        for i in range(0,self.size):
            temp_number_test = temp_number_test+0.3*random.random()
            temp_noise = random.gauss(0.5,1)
            f_test.write("%.3f" % temp_number_test + ":   " + "%.3f\n" % (math.sin(2*math.pi*temp_number_test+0.5)+temp_noise))
        f_test.close()

    # read the data in the file
    def file_dir(self,filename):
        f = open(filename, "r")
        sample_dict = {}
        for line in f:
            line = line.strip()
            line_elements = line.split(":   ")
            sample_dict[float(line_elements[0])] = float(line_elements[1])
        f.close()
        return sample_dict