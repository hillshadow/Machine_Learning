import numpy as np
import polynomial as poly
import matplotlib.pyplot as mp
import test_Module as tm

# Calculate the w_matrix by iteration
class gradient_Descent:
    def __init__(self,step_size,x_matrix,y_matrix,precision,matrix_size,size,punish):
        self.step_size = step_size
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.y_data_list = []
        self.precision = precision  # The condition for the loop to stop
        self.matrix_size = matrix_size
        temp_list = []
        for i in range(0,matrix_size):
            temp_list.append(1)
        self.w_matrix = np.asmatrix(temp_list).T
        self.loss = 0
        self.size = size
        self.punish = punish

    def gradient_start(self,max_iter):
        iter = 0
        w_temp = self.w_matrix
        loss = self.calculate_loss(w_temp)
        gradient= self.calculate_gradient(w_temp)
        while(iter < max_iter): # The max_iter if one of the conditions to stop the loop
            # self.step_size is a given number to control the move distance,
            # gradient is the direction where the number move
            w_temp = w_temp - self.step_size*gradient
            loss_temp = self.calculate_loss(w_temp)
            print(loss_temp)
            if(abs(loss-loss_temp)<0.000001):
                break;
            loss = loss_temp
            print(loss_temp.max())
            gradient = self.calculate_gradient(w_temp)
            iter+=1
        self.w_matrix = w_temp
        self.y_data_list = np.matmul(self.x_matrix,self.w_matrix)        #store the data to draw the gram
        self.loss = loss
        print(iter)

    # Calculate the gradient with the given w_matrix
    def calculate_gradient(self,w_matrix):
        x_matrix = self.x_matrix
        y_matrix = self.y_matrix
        temp_matrix = np.matmul(x_matrix.T,x_matrix)
        temp_matrix_0 = np.matmul(temp_matrix,w_matrix)
        temp_matrix_1 = temp_matrix_0+self.punish*w_matrix
        temp_matrix_2 = np.matmul(x_matrix.T,y_matrix)
        gradient =temp_matrix_1-temp_matrix_2
        return gradient

    def calculate_loss(self,w_matrix):
        temp_loss = np.matmul(self.x_matrix, w_matrix) - self.y_matrix
        loss_punish = ((np.matmul(temp_loss.T, temp_loss) + 0.5*self.punish * np.matmul(w_matrix.T,w_matrix)))
        return loss_punish


# 9    0.011
# 5    0.011
matrix_size = int(input())
sample_size = int(input())
ana = poly.Analytical_solution(matrix_size,"sample.txt","test.txt",sample_size)
test = tm.Test_Module(ana.matrix_size,"sample.txt","test.txt",ana.size,0)
gd = gradient_Descent(0.01,ana.x_matrix,ana.y_matrix,0.0000001,ana.matrix_size,ana.size,test.punish)
gd.gradient_start(500000)
test.Sample_data(gd.w_matrix)
print(gd.loss)
print(test.cal_Loss(gd.w_matrix))
# show the gram
mp.plot(test.x_list,test.y_matrix,'ro',test.x_list,test.y_data_list,'--')
mp.show()


