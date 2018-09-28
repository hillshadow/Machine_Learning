import math
import numpy as np
import matplotlib.pyplot as mp
import polynomial as poly
import samples


class NewtonApproach:
    def __init__(self,filename_1,filename_2,size,matrix_degree):
        test = samples.Samples(filename_1, filename_2,size)
        # test.calculate_data()
        dirc = test.file_dir(test.sample_file)
        self.x_list = []
        self.y_list = []
        self.x_container = []
        self.y_container = []
        self.y_data_list = []
        for x_element in dirc.keys():
            temp_list = []
            self.x_list.append(x_element)
            self.y_list.append(math.sin(2 * math.pi * x_element + 0.5))
            self.y_container.append([dirc[x_element]])
            for i in range(0,matrix_degree):
                temp_list.append(math.pow(x_element, i))
            self.x_container.append(temp_list)
        self.x_matrix = np.matrix(self.x_container)  # self.x_matrix is used to store all the information about x
        self.y_matrix = np.matrix(self.y_container)  # self.x_matrix is used to store all the information about y
        self.matrix_degree = matrix_degree  # show the dimension of the w_matrix,the form is (matrix_size,1)
        self.size = size  # show the number of the samples
        temp_list_1 = []
        for i in range(0, matrix_degree):
            temp_list_1.append(1)
        self.w_matrix = np.asmatrix(temp_list_1).T

    def calculate_w_matrix(self,precision):
        temp_w = self.w_matrix
        iterator = 0
        while(iterator<100):
            loss =self.calculate_loss(temp_w)
            temp_w = temp_w -np.matmul(self.calculate_derivative(temp_w).I,self.calculate_value(temp_w))
            temp_loss = self.calculate_loss(temp_w)
            if(abs(loss-temp_loss)<precision):
                break
            iterator+=1
            print(temp_loss)
        print(temp_loss)



    def calculate_derivative(self,w_matrix):
        temp_matrix_1 = np.matmul(self.x_matrix.T,self.x_matrix)
        return 2*temp_matrix_1

    def calculate_value(self,w_matrix):
        temp_matrix_1 = np.matmul(self.x_matrix,w_matrix)
        temp_matrix_2 = np.matmul(self.x_matrix.T,temp_matrix_1)
        temp_matrix_3 = np.matmul(self.x_matrix.T,self.y_matrix)
        return temp_matrix_2-temp_matrix_3

    def calculate_loss(self,w_matrix):
        temp_loss = np.matmul(self.x_matrix, w_matrix) - self.y_matrix
        temp_loss_1 = np.matmul(temp_loss.T, temp_loss)
        return temp_loss_1.max()


newton = NewtonApproach("sample.txt","test.txt",10,5)
newton.calculate_w_matrix(0.00000001)