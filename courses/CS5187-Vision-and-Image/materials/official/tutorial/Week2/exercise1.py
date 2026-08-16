# setup matplotlib display
#matplotlib inline

import matplotlib.pyplot as plt     # import matplotlib


#The goal of the program is to count the number of factors (not including 1 and the number itself) for each number between 2 and 100. For example, the number of factors of 2 is 0, and the number of factors for 4 is 1.

#Here are two variables to get you started, xs stores the numbers from 2 to 100, and fs will store the factors for each number in xs.



# INSERT YOUR CODE HERE and DEFINE A FUNCTION NAMELY "numfactor"


def numfactor(n,xs):

    # INSERT YOUR CODE HERE



    return countnumber


if __name__=='__main__':
    xs = range(2,101)   # the number
    fs = []             # store number of factors in this list    
    for num in xs:
        #print(num)
        x=numfactor(num,xs)
        #print(x)
        fs.append(x)
    print(fs)
    
    #Write code to plot the number of factors (y-axis) vs the number (x-axis). Don’t forget to label your axes!
    # INSERT YOUR CODE HERE
    
    # Next we will plot a histogram of the number of factors.
    # INSERT YOUR CODE HERE
