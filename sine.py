import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sys import exit


class SineNN(nn.Module):
	def __init__(self, inputDim, outputDim, hiddenDim1, hiddenDim2):
		super(SineNN, self).__init__()
		self.linear1 = nn.Linear(inputDim, hiddenDim1)
		self.linear2 = nn.Linear(hiddenDim1, hiddenDim2)
		self.linear3 = nn.Linear(hiddenDim2, outputDim)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.linear1(x)
		x = self.tanh(x)
		x = self.linear2(x)
		x = self.tanh(x)
		y = self.linear3(x)

		return y 
		

if __name__ == '__main__':
	x_values = np.linspace(-1, 1, 1000)
	x_values = x_values.astype('float32')

	y_values = np.sin(4*x_values)
	# y_values = x_values**2 - 2*x_values + 9

	x_values = x_values.reshape(-1, 1)
	y_values = y_values.reshape(-1, 1)
		
	plt.plot(x_values, y_values, label='Ground Truth')
	plt.legend(loc='best')
	plt.show()

	inputDim = 1
	outputDim = 1
	hiddenDim1 = 8
	hiddenDim2 = 4

	model = SineNN(inputDim, outputDim, hiddenDim1, hiddenDim2)

	criterion = nn.MSELoss()
	learningRate = 0.01
	epochs = 40000
	
	optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)


	for epoch in range(epochs):
		epoch += 1

		inputs = torch.from_numpy(x_values)
		labels = torch.from_numpy(y_values)

		optimizer.zero_grad()

		outputs = model(inputs)

		loss = criterion(outputs, labels)

		loss.backward()

		optimizer.step()

		if(epoch%1000 == 0 or epoch == 1):
			print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

	predicted = model(torch.from_numpy(x_values)).data.numpy()
	plt.plot(x_values, predicted, label='Predicted')
	plt.legend(loc='best')
	plt.show()