import pickle
#matplotlib inline
import matplotlib.pyplot as plt     # import matplotlib


#Next you will practice using pickle to load data from the file “data1.pickle”. The pickle is storing an ndarray object (numpy multi-dimensional array). Load the pickle file and save its contents into a variable called mydata.

#Inspecting the shape of the array, we see that there are 120 samples (rows). Each sample is a 2-dimensional vector (the number of columns).

#Finally, let’s visualise the data with a plot. Treat the 1st dimension of the samples as the x-variable, and the 2nd dimension as the y-variable. In other words, plot the 1st column of the data vs. the 2nd column of the data.

picklepath=open("./data1.pickle","rb")
mydata=pickle.load(picklepath)
print(mydata)

# INSERT YOUR CODE HERE
