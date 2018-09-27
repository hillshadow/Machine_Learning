import math
import matplotlib.pyplot as mp
import numpy as np
import samples

# calculate the loss with the given w_matrix
class Test_Module:
    def __init__(self, matrix_size, filename_1, filename_2, size,punish):
        test = samples.Samples(filename_1, filename_2, size)
        dirc = test.file_dir(test.test_file)
        self.x_list =[]
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
        self.x_matrix = np.array( self.x_container)
        self.y_matrix = np.array(self.y_container)
        self.matrix_size = matrix_size
        self.size = size
        self.loss = 0
        self.punish = punish

    def Sample_data(self,w_matrix):
        self.y_data_list = np.matmul(self.x_matrix, w_matrix)

    def cal_Loss(self,w_matrix):
        temp_loss = np.matmul(self.x_matrix, w_matrix) - self.y_matrix
        self.loss= ((np.matmul(temp_loss.T, temp_loss) + 0.5 * self.punish * np.matmul(w_matrix.T, w_matrix)))
        return self.loss





