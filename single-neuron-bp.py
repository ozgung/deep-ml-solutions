import numpy as np



def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	weights = np.array(initial_weights)
	bias = initial_bias 
	nsamples = features.shape[0]
	mse_values = []

	for epoch in range(epochs):
		# fw
		z = np.dot(features, weights) + bias
		sigmoid = lambda z: 1. / (1 + np.exp(-z))
		probs = sigmoid(z)
		mse_loss = np.mean(np.pow(labels - probs, 2.))
		mse_values.append(round(float(mse_loss), 4))
		
		# back
		# derivatives
		# σ'(z)=σ(z)(1−σ(z))
		weights_gradient = 2. / nsamples  *  np.dot(features.T, (probs - labels) * probs * (1. - probs))
		bias_gradient = 2. / nsamples  *  np.sum((probs - labels) * probs * (1. - probs))

		weights  =  weights - learning_rate * weights_gradient
		bias = bias - learning_rate * bias_gradient
	
	# return
	weights = np.round(weights, 4)
	bias = float(round(bias, 4))

	return weights, bias, mse_values


res = train_neuron(features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), labels = np.array([1, 0, 0]), initial_weights = np.array([0.1, -0.2]), initial_bias = 0.0, learning_rate = 0.1, epochs = 2)


ans = (np.array([0.1036, -0.1425]), -0.0167, [0.3033, 0.2942])

print("Res:", res)
print("Ans:", ans)
