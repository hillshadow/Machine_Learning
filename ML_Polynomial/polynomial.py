import math
import matplotlib.pyplot as mp
import numpy as np
import samples

# To get the analitical solution for the sample file

class Analytical_solution:
    x_matrix = []
    w_matrix = []
    y_matrix = []
    loss = 0 # To store the loss without punish
    loss_punish = 0  # To store the loss with punish

    def __init__(self,matrix_size,filename_1,filename_2,size):
        test = samples.Samples(filename_1, filename_2,size)
        dirc = test.file_dir(test.sample_file)
        self.x_list = []
        self.y_list = []
        self.x_container = []
        self.y_container = []
        self.y_data_list = []
        for x_element in dirc.keys():
            temp_list = []
            self.x_list.append(x_element)
            self.y_list.append(math.sin(2*math.pi*x_element+0.5))
            self.y_container.append([dirc[x_element]])
            for i in range(0,matrix_size):
                temp_list.append(math.pow(x_element, i))
            self.x_container.append(temp_list)
        self.x_matrix = np.array( self.x_container) # self.x_matrix is used to store all the information about x
        self.y_matrix = np.array(self.y_container) # self.x_matrix is used to store all the information about y
        self.matrix_size = matrix_size  # show the dimension of the w_matrix,the form is (matrix_size,1)
        self.size = size # show the number of the samples

    #  Calculate the solution to the data without punish
    def N_solution(self):
        temp_matrix_1 = np.matmul(self.x_matrix.T,self.x_matrix)
        temp_matrix_2 = np.matrix(temp_matrix_1)
        temp_matrix_3 = np.matmul(temp_matrix_2.I,self.x_matrix.T)
        self.w_matrix = np.matmul(temp_matrix_3,self.y_matrix)
        temp_loss = np.matmul(self.x_matrix, self.w_matrix)-self.y_matrix
        self.loss = np.matmul(temp_loss.T,temp_loss)/self.size
        # To store the consequence to the given x_matrix and the calculated w_matrix
        self.y_data_list = np.matmul(self.x_matrix, self.w_matrix)

    #  Calculate the solution to the data without punish
    def Y_solution(self,punish):
         temp_matrix_0 = np.eye(self.matrix_size)
         temp_matrix_1 = np.matmul(self.x_matrix.T, self.x_matrix) + punish*temp_matrix_0
         temp_matrix_2 = np.matrix(temp_matrix_1)
         temp_matrix_3 = np.matmul(temp_matrix_2.I, self.x_matrix.T)
         self.w_matrix = np.matmul(temp_matrix_3, self.y_matrix)
         temp_loss = np.matmul(self.x_matrix, self.w_matrix) - self.y_matrix
         self.loss_punish = ((np.matmul(temp_loss.T, temp_loss)+punish/2*np.matmul(self.w_matrix.T,self.w_matrix)))


