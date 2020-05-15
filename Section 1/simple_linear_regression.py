import numpy as py
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

def computeGradient(theta,X,Y):
    N = float(len(X))
    gradient_theta = [0,0]
    for example in zip(X,Y): # zip function pulls corresponding tuple data
        x = example[0]
        y = example[1]
        gradient_theta[0] += (2/N) * ((theta[0]+theta[1]*x) - y)
        gradient_theta[1] += (2/N) * ((theta[0]+theta[1]*x) - y) * x

    return gradient_theta

def updataParameters(theta,gradient_theta,learning_rate):
    new_theta = theta
    for(j,theta_j) in enumerate(theta):
        new_theta[j]=theta_j - learning_rate * gradient_theta[j]
    return new_theta

def computeMSE(theta,X,Y):
    MSE=0
    for example in zip(X,Y):
        y_pred= theta[0] + theta[1]*example[0]
        MSE += (y_pred-example[1])**2
    return MSE/float(len(X))

def trainingLoop(init_theta,learning_rate,max_epochs,X,Y):
    theta = init_theta
    loss=[]
    theta_hist=[]

    for i in range (max_epochs):
        gradient_theta = computeGradient(theta,X,Y)                     # compute currrent gradients given current parameters
        theta = updataParameters(theta,gradient_theta,learning_rate)    # update parameters
        loss.append(computeMSE(theta,X,Y))                              # keep track of loss function
        theta_hist.append([theta[0],theta[1]])                          # keep record of how parametersare changing over time
        if i %100==0:
            print("loss {}".format(loss[i]))

    return {'theta':theta, 'theta_hist':theta_hist, 'loss':loss}
 
def animateData(theta_hist, X, Y, loss, bias, coef, max_epochs):
    '''
    Produces animated plots to illustrated linear regression learning

    args:
        theta_hist (list of float): a historical record of the parameters 
        X (array of float): shape [n_samples, n_features], the input data
        Y (array of float): shape [n_samples], the labels
        loss (list): historical record of the training loss
        bias (float): the bias value used to generate the data
        coef (float): the coefficient used to generate the data
        max_epochs (int): the maximum epochs 

    returns:
        None
    '''
    # Manually determine limits for cleaner plots rather than constantly change them
    # Determine maximum loss to set the y limit for loss plot. 
    max_loss = max(loss)

    # Determine limits for X and Y data to set axes limits for the regression plot
    max_Y = py.max(Y)
    min_Y = py.min(Y)
    max_X = py.max(X)
    min_X = py.min(X)

    # Determine limits for the learned parameters. 
    # axis=0 because we need the extremes for each feature separately
    min_theta_hist = py.min(theta_hist, axis=0) 
    max_theta_hist = py.max(theta_hist, axis=0)
    
    # create figure with three subplots. set figure size.
    fig, axes = plt.subplots(3, figsize=[8,8])
 
    # set loss plot details and limits
    axes[0].set_title("MSE")
    axes[0].set_xlabel("Epoch #")
    axes[0].set_ylabel("Error value")
    axes[0].set_xlim(0, max_epochs*1.05)
    axes[0].set_ylim(0, max_loss*1.05)
    
    # set regression details and limits
    axes[1].set_title("Simple Linear Regression")
    axes[1].set_ylabel("Output reponse")
    axes[1].set_xlabel("Input Feature x")
    axes[1].set_xlim(min_X*0.9, max_X*1.1)
    axes[1].set_ylim(min_Y*0.9, max_Y*1.1)

    # set parameter history limits
    axes[2].set_title("Parameter History")
    axes[2].set_ylabel(r"$\theta_1$")
    axes[2].set_xlabel(r"$\theta_0$")
    axes[2].set_xlim(min_theta_hist[0]*0.9, max_theta_hist[0]*1.1)
    axes[2].set_ylim(min_theta_hist[1]*0.9, max_theta_hist[1]*1.1)

    # Generate 'gound truth' regression line using bias and coef parameters
    x = py.linspace(min_X - 1, max_X + 1, 100)
    y = bias + coef * x

    # initialise lines to be animated
    loss_curve, = axes[0].plot([], [])
    predicted_line, = axes[1].plot([],[])
    params, = axes[2].plot([],[])
    lines = [loss_curve, predicted_line, params]
    
    # plot data points and ground truth line here as they don't need to be animated
    axes[1].scatter(X, Y)
    axes[1].plot(x, y)

    plt.subplots_adjust(hspace=1.0)

    # init function for animator
    def init(): 
        # apparently this is good when using blitting
        lines[0].set_data(x, [py.nan] * len(loss))
        lines[1].set_data(x, [py.nan] * len(x))
        # lines['params'].set_data(x, [np.nan] * len(theta0_hist))
        return lines
    
    #Store running values and update them every frame. 
    #Could also index the original data perhaps but most examples online just append.
    y_loss = []
    theta0_hist = []
    theta1_hist = []
    xdata = []
    
    def animate(i):
        xdata.append(i)
        y_loss.append(loss[i])

        y_hat = theta_hist[i][0] + theta_hist[i][1] * x #generate predicted line with current parameters
        theta0_hist.append(theta_hist[i][0])
        theta1_hist.append(theta_hist[i][1])

        # plot loss
        lines[0].set_data(xdata, y_loss)

        # plot predicted line
        lines[1].set_data(x, y_hat)
        
        # plot parameters
        lines[2].set_data(theta0_hist, theta1_hist)
        lines[2].set_marker('o') # to see current point
        lines[2].set_markevery([i])

        return lines
    
    ani = FuncAnimation(fig, animate, init_func=init, frames=max_epochs, blit=True, interval=2, save_count=50, repeat=False)
    
    # To save as a video file
    # f = r".\videos\example_regression.mp4" 
    # writervideo = FFMpegWriter(fps=30) 
    # ani.save(f, writer=writervideo)

    # Plotting legends doesn't work as expected. Need to find solution
    # plt.legend()    
    plt.show()

if __name__ == "__main__":
    # Generate data
    bias = 50
    X,Y,coeff= make_regression(n_samples=50,n_features=1,noise=20.0, bias=bias,coef=True)

    # hyperparameters
    max_epochs = 2000
    learning_rate = 0.001
    init_theta = [0,0]

    summary = trainingLoop(init_theta=init_theta, learning_rate=learning_rate,max_epochs=max_epochs,X=X,Y=Y)
    animateData(summary['theta_hist'],X,Y,summary['loss'],bias,coeff,max_epochs)
    print("Training complete\nFinal parameters\ntheta0 = {}, theta1 = {}".format(summary['theta'][0],summary['theta'][1]))


print('finished')