import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression as lr  
import pickle as pk 

#  data featching
# The CSV file is tab-delimited, so pass sep='\t'.
data = pd.read_csv("student_test_data.csv", sep="\t")


#  data cleaning

print(data.head())
print(data.tail())
print( data.isnull().sum())
print(data.describe())
print(data.info())

 # student_id	hours_studied	sleep_hours	attendance_percent	previous_scores	exam_score

a = data['hours_studied']
plt.subplot(2,2,1)
plt.hist(a,bins = 100)
plt.title('hours_studied')

b = data['sleep_hours']
plt.subplot(2,2,2)
plt.hist(b,bins = 100)
plt.title('sleep_hours')

c = data['attendance_percent']
plt.subplot(2,2,3)
plt.hist(c,bins = 100)
plt.title('attendance_percent')

d = data['previous_scores']
plt.subplot(2,2,4)
plt.hist(d,bins = 100)
plt.title('previous_scores')

plt.show()


test_x = data.iloc[:,1:5][0:100]
test_y = data['exam_score'][0:100]

pre_x = data.iloc[:,1:5][100:200]
pre_y = data['exam_score'][100:200]


# Linear regression model = 0.84
# SVM = 0.12
# decisiom tree = 0.54

modal = lr()
modal.fit(test_x,test_y)
doc = modal.score(pre_x,pre_y)

print(doc)

#a = pk.dump(modal,open("prediction_modal_.0_studentscore.pkl","wb"))
