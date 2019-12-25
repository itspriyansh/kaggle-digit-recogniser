import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from pandas import DataFrame

def Roll(theta1,theta2):
	return np.r_[theta1.flatten(), theta2.flatten()]

def Unroll(theta,n,l1,c):
	theta1 = theta[:(n+1)*l1].reshape(n+1,l1)
	theta2 = theta[(n+1)*l1:].reshape(l1+1,c)
	return theta1,theta2

def visualize(X,Y,class_names):
	plt.figure(figsize=(10,10))
	for i in range(25):
	    plt.subplot(5,5,i+1)
	    plt.xticks([])
	    plt.yticks([])
	    plt.grid(False)
	    plt.imshow(X[i].reshape(28,28), cmap=plt.cm.binary)
	    plt.xlabel(class_names[int(Y[i].dot(np.array(range(len(class_names))).transpose()).sum())])
	plt.show()

def sigmoid(z):
	x = 1/(1+np.exp(-z))
	return x

def sigmoidGradient(z):
	x = sigmoid(z)
	return x*(1-x)

def costFunc(theta,X,Y,lambd,n,l1,c):
	m=X.shape[0]
	theta1,theta2=Unroll(theta,n,l1,c)

	z2 = np.c_[np.ones(m), X].dot(theta1)
	a2 = sigmoid(z2)
	z3 = np.c_[np.ones(m), a2].dot(theta2)
	a3 = sigmoid(z3)

	J = -(np.log(a3[Y==1]).sum()+np.log(1-a3[Y==0]).sum())/m + lambd * ((theta1[1:]**2).sum()+(theta2[1:]**2).sum())/(2*m)
	return J

def gradient(theta,X,Y,lambd,n,l1,c):
	m=X.shape[0]
	theta1,theta2=Unroll(theta,n,l1,c)

	z2 = np.c_[np.ones(m), X].dot(theta1)
	a2 = sigmoid(z2)
	z3 = np.c_[np.ones(m), a2].dot(theta2)
	a3 = sigmoid(z3)

	delta2 = a3-Y
	delta1 = delta2.dot(theta2[1:].transpose())*sigmoidGradient(z2)

	theta1_grad = (np.c_[np.ones(m), X].transpose().dot(delta1) + lambd * np.r_[np.ones((1,theta1.shape[1])), theta1[1:]])/m
	theta2_grad = (np.c_[np.ones(m), a2].transpose().dot(delta2) + lambd * np.r_[np.ones((1,theta2.shape[1])), theta2[1:]])/m

	return Roll(theta1_grad,theta2_grad)

def trainNeuralNetwork(X,Y,lambd,n,l1,c,iterations):
	theta = np.random.rand((n+1)*l1+(l1+1)*c)/(n*l1*c)

	theta = minimize(costFunc,theta,jac=gradient,args=(X,Y,lambd,n,l1,c),method='CG',options={'disp':True,'maxiter':iterations})
	return theta.x

def predict(X,theta,n,l1,c):
	theta1,theta2 = Unroll(theta,n,l1,c)
	m = X.shape[0]

	z2 = np.c_[np.ones(m), X].dot(theta1)
	a2 = sigmoid(z2)
	z3 = np.c_[np.ones(m), a2].dot(theta2)
	a3 = sigmoid(z3)

	y = np.zeros(a3.shape)

	for i in range(m):
		y[i, np.where(a3[i]==np.amax(a3[i]))]=1

	return y

def selectNormalization(X,Y,X_,Y_,n,l1,c):
	lambd=0
	theta=trainNeuralNetwork(X,Y,lambd,n,l1,c,100)

	minCost = costFunc(theta, X_,Y_,0,n,l1,c,)
	lambdas = [0.01,0.03,0.1,0.3,1,3,10,30]

	for l in lambdas:
		theta=trainNeuralNetwork(X,Y,l,n,l1,c,100)
		cost = costFunc(theta, X_,Y_,0,n,l1,c,)
		if(minCost>cost):
			minCost=cost
			lambd=l

	return lambd

def learningCurve(X,Y,X_,Y_,lambd,n,l1,c):
	i=1000
	m=X.shape[0]
	trainingCosts = []
	validationCosts = []
	indices = []
	while i<=m:
		theta = trainNeuralNetwork(X[:i],Y[:i],lambd,n,l1,c,100)

		trainingCosts.append(costFunc(theta,X[:i],Y[:i],0,n,l1,c,))
		validationCosts.append(costFunc(theta,X_,Y_,0,n,l1,c,))
		indices.append(i)
		i=i+1000
	plt.plot(indices,trainingCosts)
	plt.plot(indices,validationCosts)
	plt.show()


def main():
	dataset = np.array(pd.read_csv('./data/train.csv'))
	# X_test = np.array(pd.read_csv('./data/test.csv'))

	X = np.array(dataset[:,1:])/255.0
	y = np.array(dataset[:,0])
	X_test = np.array(pd.read_csv('./data/test.csv'))/255.0

	n = X.shape[1]
	c = 10

	class_names=np.array(range(10)).transpose()

	Y = np.zeros((X.shape[0],c))

	for i in range(y.shape[0]):
		Y[i][y[i]] = 1

	X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=0.2,random_state=1)

	n=X_train.shape[1]
	m=X_train.shape[0]
	l1=128
	iterations=300

	# lambd=selectNormalization(X_train[:6000],Y_train[:6000],X_validation[:6000],Y_validation[:6000],n,l1,c)
	lambd=0.3
	print(lambd)

	# learningCurve(X_train[:10000],Y_train[:10000],X_validation[:10000],Y_validation[:10000],lambd,n,l1,c)

	indices = np.array(range(c)).transpose()

	theta = trainNeuralNetwork(X_train,Y_train,lambd,n,l1,c,iterations)
	y_predict = predict(X_validation,theta,n,l1,c)
	accuracy = (Y_validation.dot(indices)==y_predict.dot(indices)).sum()*100/Y_validation.shape[0]
	print("Test accuracy: "+str(accuracy))

	theta = trainNeuralNetwork(X,Y,lambd,n,l1,c,iterations)

	# DataFrame(theta).to_csv('output.csv')

	y_predict = predict(X_test,theta,n,l1,c)

	# visualize(X_validation[:25],y_predict[:25],class_names)

	di = {'Label': y_predict.dot(indices).astype('int'), 'ImageId': np.array(range(1,X_test.shape[0]+1)).transpose()}

	DataFrame(di).to_csv('output.csv')

	# accuracy = (Y_validation.dot(indices)==y_predict.dot(indices)).sum()*100/Y_validation.shape[0]
	# print("Test accuracy: "+str(accuracy))
	print("Done")

main()