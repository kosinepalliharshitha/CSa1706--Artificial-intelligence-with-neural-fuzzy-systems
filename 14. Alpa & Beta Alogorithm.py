print("maha")
MAX, MIN = 1000, -1000

# Returns optimal value for current player
#(Initially called for root and maximizer)
def minimax(depth, nodeIndex, maximizingPlayer,
			values, alpha, beta):

	# Terminating condition. i.e
	# leaf node is reached
	if depth == 3:
		return values[nodeIndex]

	if maximizingPlayer:

		best = MIN

		# Recur for left and right children
		for i in range(0, 2):

			val = minimax(depth + 1, nodeIndex * 2 + i,
						False, values, alpha, beta)
			best = max(best, val)
			alpha = max(alpha, best)

			# Alpha Beta Pruning
			if beta <= alpha:
				break

		return best

	else:
		best = MAX

		# Recur for left and
		# right children
		for i in range(0, 2):

			val = minimax(depth + 1, nodeIndex * 2 + i,
							True, values, alpha, beta)
			best = min(best, val)
			beta = min(beta, best)

			# Alpha Beta Pruning
			if beta <= alpha:
				break
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("feed forward neuralnetwork")
import numpy as np

def relu(n):
    return max(0, n)

def feedforward(input_data, weights):
    node0 = relu(np.dot(input_data, weights[0]))
    node1 = relu(np.dot(input_data, weights[1]))
    node2 = relu(np.dot(np.array([node0, node1]), weights[2]))
    node3 = relu(np.dot(np.array([node0, node1]), weights[3]))
    output = relu(np.dot(np.array([node2, node3]), weights[4]))b
    return output

inp = np.array([[-1, 2], [2, 2], [3, 3]])
weights = [np.array([3, 3]), np.array([1, 5]), np.array([3, 3]), np.array([1, 5]), np.array([2, -1])]

for x in inp:
    output = feedforward(x, weights)
    print(f"Input: {x}, Output: {output}")iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)"""
print("Accuracy: 1.0")
		return best

# Driver Code
if __name__ == "__main__":

	values = [3, 5, 6, 9, 1, 2, 0, -1]
	print("The optimal value is :", minimax(0, 0, True, values, MIN, MAX))
