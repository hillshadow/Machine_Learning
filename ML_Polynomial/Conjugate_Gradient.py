import numpy as np
import math
import polynomial as poly
import matplotlib.pyplot as mp
import test_Module as tm

# Calculate the w_matrix by the way named Conjuate_gradient
class Conjuate_gradient:
    def __init__(self,x_matrix,y_matrix,matrix_size,max_iter,precision,size, punish):
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.max_iter = max_iter
        self.precision = precision
        self.A_matrix = np.matmul(x_matrix.T,x_matrix)+punish*np.eye(matrix_size)
        self.y_data_list = []
        temp_list = []
        for i in range(0, matrix_size):
            temp_list.append(1)
        self.w_matrix = np.asmatrix(temp_list).T
        self.size = size
        self.loss = 0
        self.punish = punish

    # calculate the w_matrix with the given algorithm
    def calculate_w(self):
        temp_b_matrix = np.matmul(self.x_matrix.T,self.y_matrix)
        temp_a_matrix = np.matmul(self.x_matrix.T,self.x_matrix)
        r0_matrix = temp_b_matrix-np.matmul(temp_a_matrix,self.w_matrix)
        p_matrix = r0_matrix
        w_matrix = self.w_matrix
        r_matrix = r0_matrix
        iter = 0
        while(iter<self.max_iter):
            print(self.calculate_loss(w_matrix))
            temp_Alpha = self.calculete_Alpha(r_matrix,p_matrix) # temp_Alpha matrix is used to calculate the residual
            temp_w  = w_matrix + temp_Alpha*p_matrix
            temp_r = r_matrix - temp_Alpha*np.matmul(self.A_matrix,p_matrix)
            if(abs(self.calculate_loss(w_matrix)-self.calculate_loss(temp_w)) < self.precision):
                break
            temp_Beta = self.calculate_Beta(r_matrix,temp_r) # temp_Beta is used to calculate the p_matrix, which is the direction of the move
            p_matrix = temp_r + temp_Beta*p_matrix
            w_matrix = temp_w
            r_matrix = temp_r
            iter+=1
        self.w_matrix = w_matrix
        self.y_data_list = np.matmul(self.x_matrix,self.w_matrix) #To store the data for the gram
        self.loss=self.calculate_loss(self.w_matrix)
        return self.loss





    def calculete_Alpha(self,r_matrix,p_matrix):
        temp_matrix_1 = np.matmul(r_matrix.T,r_matrix).max()
        temp_matrix_2 = np.matmul(p_matrix.T,self.A_matrix)
        temp_matrix_3 = np.matmul(temp_matrix_2,p_matrix).max()
        return temp_matrix_1/temp_matrix_3



    # r_matrix_advanced = k,r_matrix ==k+1
    def calculate_Beta(self,r_matrix_advanced,r_matrix):
        temp_matrix_1 = np.matmul(r_matrix.T,r_matrix).max()
        temp_matrix_2 = np.matmul(r_matrix_advanced.T,r_matrix_advanced).max()
        return temp_matrix_1/temp_matrix_2

    def calculate_loss(self,w_matrix):
        temp_loss = np.matmul(self.x_matrix, w_matrix) - self.y_matrix
        loss_punish = ((np.matmul(temp_loss.T, temp_loss) + 0.5 * self.punish * np.matmul(w_matrix.T, w_matrix)))
        return loss_punish


ana = poly.Analytical_solution(20,"sample.txt","test.txt",100)
test = tm.Test_Module(ana.matrix_size,"sample.txt","test.txt",ana.size,0.1)
gd = Conjuate_gradient(ana.x_matrix,ana.y_matrix,ana.matrix_size,1000, 0.0000000001, ana.size,test.punish)
gd.calculate_w()
test.Sample_data(gd.w_matrix)
print()
print(gd.loss)
print(test.cal_Loss(gd.w_matrix))
mp.plot(test.x_list,test.y_matrix,'b',ana.x_list,ana.y_matrix,'y',test.x_list,test.y_data_list,'--')
mp.show()