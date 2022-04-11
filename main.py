# Lets import the neccesary libraries to complete the assignment 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #to plot with matplotlib
import unittest #for unit testing
import os
#for mathematical operations
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#to plot with Bokeh
from bokeh.io import output_file
from bokeh.plotting import figure,show
from bokeh.layouts import gridplot
from scipy.optimize import curve_fit #curve fitting
from pylab import * 
from sqlalchemy import *

Train_data = pd.read_csv('train.csv') 
Ideal_data = pd.read_csv('ideal.csv') 
Test_data = pd.read_csv('test.csv')
#assigning X and Y
Train_X = Train_data['x']
Ideal_X = Ideal_data['x']
Train_Y1 = Train_data['y1']
Train_Y2 = Train_data['y2']
Train_Y3 = Train_data['y3']
Train_Y4 = Train_data['y4']

#Using matplotlib to plot the training data
fig, axs = plt.subplots(2,2, figsize=(15,15)) # generating the different plots (subplots) at once

#'tight_layout' helps adjusting subplots so they fit into the figure area, 'pad' controls the padding around the figure border & subplots
fig.tight_layout(pad=3)

# Plotting 4 figures, the 'axs[value,value]' allows to place graphs in the desire position
axs[0,0].plot(Train_X, Train_Y1)  
axs[0,0].set_title('Train_X Vs Train_Y1',)
axs[1,0].plot(Train_X, Train_Y2, 'tab:red')
axs[1,0].set_title('Train_X Vs Train_Y2')
axs[0, 1].plot(Train_X, Train_Y3, 'tab:orange')
axs[0, 1].set_title('Train_X Vs Train_Y3')
axs[1,1 ].plot(Train_X, Train_Y4, 'tab:green')
axs[1, 1].set_title('Train_X Vs Train_Y4')
plt.show()

#using matplotlib to plot the test data

plt.scatter(Test_data['x'], Test_data['y'], color='b')
plt.suptitle('Test X vs Test Y')
plt.show()

#Using Bokeh to plot the training data

from bokeh.io import curdoc # importing curdoc to change the theme colour 
from bokeh.plotting import figure,show
curdoc().theme = 'night_sky'

# 4 plots will be display
b_graph1=figure(width=600, height=600,title='X vs Train_Y1 ') #1st graph for X vs Y1
b_graph1.triangle(Train_X,Train_Y1, size=12, color='cyan', alpha=0.5)
b_graph2=figure(width=600, height=600,title='X vs Train_Y3 ')#2nd graph for X vs Y3
b_graph2.circle(Train_X,Train_Y3,size=12, color='yellow', alpha=0.5)
b_graph3=figure(width=600, height=600,title='X vs Train_Y2')#3rd graph for X vs Y2
b_graph3.circle(Train_X,Train_Y2,size=12, color='red', alpha=0.5)
b_graph4=figure(width=600, height=600,title='X vs Train_Y4 ')#4th graph for X vs Y4
b_graph4.circle(Train_X,Train_Y4,size=12, color='green', alpha=0.5)

#grouping all plots in a gridformat following 'p = gridplot([[s1, s2], [s3, s3]])'
p= gridplot([[b_graph1,b_graph2],[b_graph3,b_graph4]])
output_file("foo.html")
#display plots
show(p)

#The function 'np.polyfit()' is use to find the least square polynomial fit
Model_A = np.polyfit(Train_X, Train_Y1, 3)
Model_B = np.polyfit(Train_X, Train_Y2, 3) 
Model_C = np.polyfit(Train_X, Train_Y3, 3) 
Model_D = np.polyfit(Train_X, Train_Y4, 3)


#using 'np.poly1d', one-dimentional class 
Prediction_1 = np.poly1d(Model_A)
Prediction_2 = np.poly1d(Model_B)
Prediction_3 = np.poly1d(Model_C)
Prediction_4 = np.poly1d(Model_D)

#setting models variables 
y_model_1= Prediction_1(Ideal_X)
y_model_2 = Prediction_2(Ideal_X)
y_model_3 = Prediction_3(Ideal_X)
y_model_4 = Prediction_4(Ideal_X)


# to meassure erro and or performance it is important that we use these:

# MAE=mean absolute error
mae_y1 = mean_absolute_error(Train_Y1, Prediction_1(Train_X))
mae_y2 = mean_absolute_error(Train_Y2, Prediction_2(Train_X))
mae_y3 = mean_absolute_error(Train_Y3, Prediction_3(Train_X))
mae_y4 = mean_absolute_error(Train_Y4, Prediction_4(Train_X))


print("Mean Absolute Error (MAE)")
print("Model 1 :", mae_y1)
print("Model 2 :", mae_y2)
print("Model 3 :", mae_y3)
print("Model 4 :", mae_y4)
print("--------------------------------------------")

# MSE= mean squared error
mse_y1 = mean_squared_error(Train_Y1, Prediction_1(Train_X))
mse_y2 = mean_squared_error(Train_Y2, Prediction_2(Train_X))
mse_y3 = mean_squared_error(Train_Y3, Prediction_3(Train_X))
mse_y4 = mean_squared_error(Train_Y4, Prediction_4(Train_X))

print("Mean Squared Error (MSE)")
print("Model 1 :", mse_y1)
print("Model 2 :", mse_y2)
print("Model 3 :", mse_y3)
print("Model 4 :", mse_y4)

# R2 Scores= Coefficient of determination
r2_y1 = r2_score(Train_Y1, Prediction_1(Train_X))
r2_y2 = r2_score(Train_Y2, Prediction_2(Train_X))
r2_y3 = r2_score(Train_Y3, Prediction_3(Train_X))
r2_y4 = r2_score(Train_Y4, Prediction_4(Train_X))
print("--------------------------------------------")
print("R2 Score for models is as follows")
print("Model 1 :", r2_y1)
print("Model 2 :", r2_y2)
print("Model 3 :", r2_y3)
print("Model 4 :", r2_y4)



# to meassure erro and or performance it is important that we use these:

# MAE=mean absolute error
mae_y1 = mean_absolute_error(Train_Y1, Prediction_1(Train_X))
mae_y2 = mean_absolute_error(Train_Y2, Prediction_2(Train_X))
mae_y3 = mean_absolute_error(Train_Y3, Prediction_3(Train_X))
mae_y4 = mean_absolute_error(Train_Y4, Prediction_4(Train_X))


print("Mean Absolute Error (MAE)")
print("Model 1 :", mae_y1)
print("Model 2 :", mae_y2)
print("Model 3 :", mae_y3)
print("Model 4 :", mae_y4)
print("--------------------------------------------")

# MSE= mean squared error
mse_y1 = mean_squared_error(Train_Y1, Prediction_1(Train_X))
mse_y2 = mean_squared_error(Train_Y2, Prediction_2(Train_X))
mse_y3 = mean_squared_error(Train_Y3, Prediction_3(Train_X))
mse_y4 = mean_squared_error(Train_Y4, Prediction_4(Train_X))

print("Mean Squared Error (MSE)")
print("Model 1 :", mse_y1)
print("Model 2 :", mse_y2)
print("Model 3 :", mse_y3)
print("Model 4 :", mse_y4)

# R2 Scores= Coefficient of determination
r2_y1 = r2_score(Train_Y1, Prediction_1(Train_X))
r2_y2 = r2_score(Train_Y2, Prediction_2(Train_X))
r2_y3 = r2_score(Train_Y3, Prediction_3(Train_X))
r2_y4 = r2_score(Train_Y4, Prediction_4(Train_X))
print("--------------------------------------------")
print("R2 Score for models is as follows")
print("Model 1 :", r2_y1)
print("Model 2 :", r2_y2)
print("Model 3 :", r2_y3)
print("Model 4 :", r2_y4)

#Finging the ideal functions 
def ideal_function(train_data, ideal_data):
    
    if not isinstance(train_data, pd.Series):
        raise MyException(train_data, "Exception raised! {} Must be a Pandas series".format(train_data))
        
    squared_sum = []
    for j in range(1, len(ideal_data.columns)):
        squared_sum.append((j, sum(abs(train_data - ideal_data['y'+str(j)].values))))
    squared_sum.sort(key = lambda x: x[1]) 
    return squared_sum[0]

Ideal_Y1  = ideal_function(Train_Y1, Ideal_data)
Ideal_Y2  = ideal_function(Train_Y2, Ideal_data)
Ideal_Y3  = ideal_function(Train_Y3, Ideal_data)
Ideal_Y4  = ideal_function(Train_Y4, Ideal_data)


print("Ideal Function (Y1) is:", "y" + str(Ideal_Y1[0]),"," ' Ideal Function (Y2) is:',"y" + str(Ideal_Y2[0]))
print("Ideal Function (Y3) is:", "y" + str(Ideal_Y3[0]), "," ' Ideal Function (Y4) is:', "y" + str(Ideal_Y4[0]))


#find maximum deviation aka criteria ii
def maximum_deviation(train_data, ideal_data):
   
    dev = abs(train_data.values - ideal_data.values)
    dev.sort()
    return dev[-1:][0]

maximum_dev1  = maximum_deviation(Train_Y1, Ideal_data['y'+str(Ideal_Y1[0])])
maximum_dev2  = maximum_deviation(Train_Y2, Ideal_data['y'+str(Ideal_Y2[0])])
maximum_dev3  = maximum_deviation(Train_Y3, Ideal_data['y'+str(Ideal_Y3[0])])
maximum_dev4  = maximum_deviation(Train_Y4, Ideal_data['y'+str(Ideal_Y4[0])])
print('Maximum_dev1:',maximum_dev1, 'Maximum_dev2:',maximum_dev2)
print('Maximum_dev3:',maximum_dev3, 'Maximum_dev4:',maximum_dev4)


Test_data['Delta Y'] = pd.Series(dtype=float)
Test_data['No of Ideal Y'] = pd.Series(dtype=object)
#mapping the test data to the ideal functions ii

for x in range(0, len(Test_data)):
    for i in range(0, len(Ideal_data)):
        # check for first ideal
        if abs(Test_data['y'][x] - Ideal_data['y' + str(Ideal_Y1[0])][i]) <= maximum_dev1 * np.sqrt(2):
            Test_data['No of Ideal Y'][x] = 'y' + str(Ideal_Y1[0])
            Test_data['Delta Y'][x] = abs(Test_data['y'][x] - Ideal_data['y' + str(Ideal_Y1[0])][i])
        
        # check for second ideal
        if abs(Test_data['y'][x] - Ideal_data['y' + str(Ideal_Y2[0])][i]) <= maximum_dev2 * np.sqrt(2):
            Test_data['No of Ideal Y'][x] = 'y' + str(Ideal_Y2[0])
            Test_data['Delta Y'][x] = abs(Test_data['y'][x] - Ideal_data['y' + str(Ideal_Y2[0])][i])
            
        # check for third ideal
        if abs(Test_data['y'][x] - Ideal_data['y' + str(Ideal_Y3[0])][i]) <= maximum_dev3 * np.sqrt(2):
            Test_data['No of Ideal Y'][x] = 'y' + str(Ideal_Y3[0])
            Test_data['Delta Y'][x] = abs(Test_data['y'][x] - Ideal_data['y' + str(Ideal_Y3[0])][i])
             
        # check for fourth ideal
        if abs(Test_data['y'][x] - Ideal_data['y' + str(Ideal_Y4[0])][i]) <= maximum_dev4 * np.sqrt(2):
            Test_data['No of Ideal Y'][x] = 'y' + str(Ideal_Y4[0])
            Test_data['Delta Y'][x] = abs(Test_data['y'][x] - Ideal_data['y' + str(Ideal_Y4[0])][i])

# To calculate the number of test data points assigned

number_assigned = len(Test_data) - Test_data.count()
print("The number of x-y-pair values that can be assigned to the four chosen ideal functions :", len(Test_data) - number_assigned.values[3])

#User deined exception and construciton of Exception class
class Exception1(Exception):
    
    def __init___(self, exception_parameter, exception_message):
        
        super().__init__(self, exception_parameter, exception_message)

#Class for database connection and saving the training data to the database

class databaseconnection:
    
    engine = create_engine('sqlite:///assignment.db', echo=True)
    meta = MetaData()

    def save_trainingdata(self):
       
        training = pd.read_csv('train.csv')
        sqlite_table = "training_data"
        try:
            sqlite_connection = self.engine.connect()
            training.to_sql(sqlite_table, sqlite_connection, if_exists='fail')
        except:
            print('ops! The conneciton to the database has failed')

#Class for database conneciton of the ideal functions
class databaseidealconnection(databaseconnection):

    def save_idealfunctions(self):
    
        ideal= pd.read_csv('ideal.csv')     #saving ideal functions data to SQLite file
        sqlite_table = "ideal_functions"
        try:
            sqlite_connection = self.engine.connect()
            ideal.to_sql(sqlite_table, sqlite_connection, if_exists='fail')
        except:
            print('ops! The conneciton to the database has failed')
            
    def save_testdata(self, test_data):  
        
        sqlite_table = "test_data_table"         #saving updated test data to SQLite file
        try:
            sqlite_connection = self.engine.connect()
            test_data.to_sql(sqlite_table, sqlite_connection, if_exists='fail')
        except:
            print('ops! The conneciton to the database has failed')


#unit test to check quality of 
class InsertionTestCheck(unittest.TestCase):
    
    def test_inserted_data(self):
        
        try:
            os.remove("assignment.db")
        except OSError:
            print("Database File does not exist")

        obj = databaseidealconnection()
        obj.save_trainingdata()
        obj.save_idealfunctions()
        obj.save_testdata(Test_data)

        db = create_engine('sqlite:///assignment.db')
        metadata = MetaData(db)

        test_data_table = Table('test_data_table', metadata, autoload=True)
        query = test_data_table.select()

        df_test = pd.DataFrame(query.execute())
        df_test.drop(0, axis=1, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        df_test.columns = Test_data.columns
        self.assertEqual(len(df_test.columns), len(Test_data.columns))
        pd.testing.assert_frame_equal(df_test, Test_data)
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


#Unit test to check the quality of ideal functions code

class TestFunctions(unittest.TestCase):
    
    def shouldReturnIdealFunctions(self):
        
        expectedIdealY1 = 'y33'
        expectedIdealY2 = 'y10'
        expectedIdealY3 = 'y48'
        expectedIdealY4 = 'y18'
        
        Ideal_Y1  = ideal_function(Train_Y1, Ideal_data)
        Ideal_Y2  = ideal_function(Train_Y2, Ideal_data)
        Ideal_Y3  = ideal_function(Train_Y3, Ideal_data)
        Ideal_Y4  = ideal_function(Train_Y4, Ideal_data)
        
        self.assertEqual(expectedIdealY1, Ideal_Y1)
        self.assertEqual(expectedIdealY1, Ideal_Y2)
        self.assertEqual(expectedIdealY1, Ideal_Y3)
        self.assertEqual(expectedIdealY1, Ideal_Y4)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

#4 plots:

i_graph1 = figure(width=600, plot_height=600,title="X vs Y1")
i_graph1.triangle_pin(Ideal_X, Ideal_data['y' + str(Ideal_Y1[0])], size=10, color='cyan', alpha=0.5)
i_graph1.outline_line_color='yellow' #adding colour to the outsine line of the plot
i_graph2 = figure(width=600, plot_height=600,title="X vs Y2")
i_graph2.circle_dot(Ideal_X, Ideal_data['y' + str(Ideal_Y2[0])], size=10, color='cyan', alpha=0.5)
i_graph2.outline_line_color='yellow'#adding colour to the outsine line of the plot
i_graph3 = figure(width=600, plot_height=600,title="X vs Y3")
i_graph3.square_dot(Ideal_X, Ideal_data['y' + str(Ideal_Y3[0])], size=10, color='cyan', alpha=0.5)
i_graph3.outline_line_color='yellow'#adding colour to the outsine line of the plot
i_graph4 = figure(width=600, plot_height=600,title="X vs Y4")
i_graph4.triangle_dot(Ideal_X, Ideal_data['y' + str(Ideal_Y4[0])], size=10, color='cyan', alpha=0.5)
i_graph4.outline_line_color='yellow'#adding colour to the outsine line of the plot
# make a grid
output_file('ideal_functions.html')
g = gridplot([[i_graph1, i_graph2], [i_graph3, i_graph4]])
show(g)
# Plotting test data in Bokeh
t_graph1= figure(title="Test X vs Test Y")
t_graph1.circle_dot(Test_data['x'], Test_data['y'], size=12, color='red',alpha=0.5 )
t_graph1.outline_line_color='yellow' #adding colour to the outsine line of the plot
output_file("test_data_plotting.html")
show(t_graph1)