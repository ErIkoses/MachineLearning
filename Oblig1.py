import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Spotify dataset
spotify_data = pd.read_csv("SpotifyFeatures.csv")

# Report the number of samples (rows) and features (columns)
num_samples = spotify_data.shape[0]
num_features = spotify_data.shape[1]

print(f"Number of samples (songs): {num_samples}")
print(f"Number of features (song properties): {num_features}")

# Retrieve only the 'Pop' and 'Classical' genres and make a copy of the resulting DataFrame
pop_classical_data = spotify_data[spotify_data['genre'].isin(['Pop', 'Classical'])].copy()

# Create labels for the samples: 'Pop' = 1, 'Classical' = 0 using .loc
pop_classical_data.loc[:, 'label'] = pop_classical_data['genre'].apply(lambda x: 1 if x == 'Pop' else 0)

# Select only the 'liveness' and 'loudness' features along with the labels
selected_features = pop_classical_data[['liveness', 'loudness']]
labels = pop_classical_data['label']

# Report how many samples belong to each class
pop_count = (pop_classical_data['genre'] == 'Pop').sum()
classical_count = (pop_classical_data['genre'] == 'Classical').sum()

print(f"Number of 'Pop' samples: {pop_count}")
print(f"Number of 'Classical' samples: {classical_count}")

# Convert selected features and labels to numpy arrays
X = selected_features.to_numpy()  # Features matrix
y = labels.to_numpy()  # Labels array

# Separate 'Pop' and 'Classical' samples
pop_samples = X[y == 1]
pop_labels = y[y == 1]
classical_samples = X[y == 0]
classical_labels = y[y == 0]

# Shuffle the samples
np.random.seed(42)  # For reproducibility
pop_indices = np.random.permutation(len(pop_samples))
classical_indices = np.random.permutation(len(classical_samples))

# Apply the shuffled indices
pop_samples = pop_samples[pop_indices]
pop_labels = pop_labels[pop_indices]
classical_samples = classical_samples[classical_indices]
classical_labels = classical_labels[classical_indices]

# 80-20 split for 'Pop' samples
pop_train_size = int(0.8 * len(pop_samples))
X_train_pop = pop_samples[:pop_train_size]
y_train_pop = pop_labels[:pop_train_size]
X_test_pop = pop_samples[pop_train_size:]
y_test_pop = pop_labels[pop_train_size:]

# 80-20 split for 'Classical' samples
classical_train_size = int(0.8 * len(classical_samples))
X_train_classical = classical_samples[:classical_train_size]
y_train_classical = classical_labels[:classical_train_size]
X_test_classical = classical_samples[classical_train_size:]
y_test_classical = classical_labels[classical_train_size:]

# Combine the training and test sets from both classes
X_train = np.vstack((X_train_pop, X_train_classical))
y_train = np.hstack((y_train_pop, y_train_classical))
X_test = np.vstack((X_test_pop, X_test_classical))
y_test = np.hstack((y_test_pop, y_test_classical))

# Shuffle the combined training set
train_indices = np.random.permutation(len(X_train))
X_train = X_train[train_indices]
y_train = y_train[train_indices]

# Shuffle the combined test set
test_indices = np.random.permutation(len(X_test))
X_test = X_test[test_indices]
y_test = y_test[test_indices]

# Verify the shape of the datasets
print(f"Training set features shape: {X_train.shape}")
print(f"Test set features shape: {X_test.shape}")
print(f"Training set labels shape: {y_train.shape}")
print(f"Test set labels shape: {y_test.shape}")



# Retrieve only the 'Pop' and 'Classical' genres and make a copy of the resulting DataFrame
pop_classical_data = spotify_data[spotify_data['genre'].isin(['Pop', 'Classical'])].copy()

# Create labels for the samples: 'Pop' = 1, 'Classical' = 0
pop_classical_data.loc[:, 'label'] = pop_classical_data['genre'].apply(lambda x: 1 if x == 'Pop' else 0)

# Select only the 'liveness' and 'loudness' features along with the labels
selected_features = pop_classical_data[['liveness', 'loudness']]
labels = pop_classical_data['label']

# Convert the selected features and labels to numpy arrays
X = selected_features.to_numpy()  # Features matrix
y = labels.to_numpy()  # Labels array

# Separate 'Pop' and 'Classical' samples
pop_samples = X[y == 1]
pop_labels = y[y == 1]
classical_samples = X[y == 0]
classical_labels = y[y == 0]

# Shuffle the samples
np.random.seed(42)  # For reproducibility
pop_indices = np.random.permutation(len(pop_samples))
classical_indices = np.random.permutation(len(classical_samples))

# Apply the shuffled indices
pop_samples = pop_samples[pop_indices]
pop_labels = pop_labels[pop_indices]
classical_samples = classical_samples[classical_indices]
classical_labels = classical_labels[classical_indices]

# 80-20 split for 'Pop' samples
pop_train_size = int(0.8 * len(pop_samples))
X_train_pop = pop_samples[:pop_train_size]
y_train_pop = pop_labels[:pop_train_size]
X_test_pop = pop_samples[pop_train_size:]
y_test_pop = pop_labels[pop_train_size:]

# 80-20 split for 'Classical' samples
classical_train_size = int(0.8 * len(classical_samples))
X_train_classical = classical_samples[:classical_train_size]
y_train_classical = classical_labels[:classical_train_size]
X_test_classical = classical_samples[classical_train_size:]
y_test_classical = classical_labels[classical_train_size:]

# Combine the training and test sets from both classes
X_train = np.vstack((X_train_pop, X_train_classical))
y_train = np.hstack((y_train_pop, y_train_classical))
X_test = np.vstack((X_test_pop, X_test_classical))
y_test = np.hstack((y_test_pop, y_test_classical))

# Shuffle the combined training set
train_indices = np.random.permutation(len(X_train))
X_train = X_train[train_indices]
y_train = y_train[train_indices]

# Shuffle the combined test set
test_indices = np.random.permutation(len(X_test))
X_test = X_test[test_indices]
y_test = y_test[test_indices]

# Verify the shape of the datasets
print(f"Training set features shape: {X_train.shape}")
print(f"Test set features shape: {X_test.shape}")
print(f"Training set labels shape: {y_train.shape}")
print(f"Test set labels shape: {y_test.shape}")

print(np.bincount(y_train))
print(np.bincount(y_test))

# Separate 'Pop' and 'Classical' samples for plotting
pop_samples = selected_features[pop_classical_data['label'] == 1]
classical_samples = selected_features[pop_classical_data['label'] == 0]

# Plot 'Pop' samples (label = 1)
plt.scatter(pop_samples['liveness'], pop_samples['loudness'], color='blue', label='Pop', alpha=0.5)

# Plot 'Classical' samples (label = 0)
plt.scatter(classical_samples['liveness'], classical_samples['loudness'], color='red', label='Classical', alpha=0.5)

# Add labels and title
plt.xlabel('Liveness')
plt.ylabel('Loudness')
plt.title('Liveness vs Loudness (Pop vs Classical)')
plt.legend()

# Show the plot
plt.show()

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

# Logistic regression using stochastic gradient descent (SGD)
def logistic_regression_sgd(X, y, learning_rate=0.01, epochs=100, shuffle=True):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0
    training_errors = []

    for epoch in range(epochs):
        if shuffle:
            indices = np.random.permutation(num_samples)
            X = X[indices]
            y = y[indices]

        total_loss = 0
        for i in range(num_samples):
            xi = X[i]
            yi = y[i]

            # Prediction
            linear_model = np.dot(xi, weights) + bias
            y_pred = sigmoid(linear_model)

            # Loss calculation
            total_loss += cross_entropy_loss(yi, y_pred)

            # Gradient calculation
            gradient_w = (y_pred - yi) * xi
            gradient_b = (y_pred - yi)

            # Update weights and bias
            weights -= learning_rate * gradient_w
            bias -= learning_rate * gradient_b

        # Average loss over all samples in the epoch
        avg_loss = total_loss / num_samples
        training_errors.append(avg_loss)

        if epoch % 10 == 0:  # Report every 10 epochs
            print(f"Epoch {epoch}: Training loss = {avg_loss:.4f}")

    return weights, bias, training_errors

# Normalize features (liveness and loudness)
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train_normalized = (X_train - X_train_mean) / X_train_std

X_test_mean = X_test.mean(axis=0)
X_test_std = X_test.std(axis=0)
X_test_normalized = (X_test - X_test_mean) / X_test_std

# Train logistic regression model using SGD
learning_rate = 0.0001
epochs = 100
weights, bias, training_errors = logistic_regression_sgd(X_train_normalized, y_train, learning_rate, epochs)

test_weights, test_bias, test_training_errors = logistic_regression_sgd(X_test_normalized, y_test, learning_rate, epochs)
# Plot the training error as a function of epochs
plt.plot(training_errors)
plt.xlabel('Epochs')
plt.ylabel('Training Error (Cross-Entropy Loss)')
plt.title('Training Error vs Epochs')
plt.show()

plt.plot(test_training_errors)
plt.xlabel('Epochs')
plt.ylabel('Training Error (Cross-Entropy Loss)')
plt.title('Training Error vs Epochs')
plt.show()

# Predict function using the trained model
def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return np.round(y_pred)  # Round to 0 or 1

#Predictions on the training set
y_train_pred = predict(X_train_normalized, weights, bias)
train_accuracy = np.mean(y_train_pred == y_train) * 100

y_test_pred = predict(X_test_normalized, test_weights, test_bias)
test_train_accuracy = np.mean(y_test_pred == y_test) * 100
print(f"Training set accuracy: {train_accuracy:.2f}%")
print(f"Training set accuracy: {test_train_accuracy:.2f}%")
"""
for lr in [0.0001, 0.001, 0.01]: #I found the ideal learning rate to be 0.0001 for 100 epochs
    print(f"Training with learning rate: {lr}")
    weights, bias, training_errors = logistic_regression_sgd(X_train_normalized, y_train, learning_rate=lr, epochs=100)
    
    # Plot the training error for each learning rate
    plt.plot(training_errors, label=f"LR = {lr}")
    
plt.xlabel('Epochs')
plt.ylabel('Training Error (Cross-Entropy Loss)')
plt.title('Training Error vs Epochs for Different Learning Rates')
plt.legend()
plt.show()
"""

# Compute the confusion matrix
TP = np.sum((y_test == 1) & (y_test_pred == 1))  # True Positives
TN = np.sum((y_test == 0) & (y_test_pred == 0))  # True Negatives
FP = np.sum((y_test == 0) & (y_test_pred == 1))  # False Positives
FN = np.sum((y_test == 1) & (y_test_pred == 0))  # False Negatives

# Create the confusion matrix as a 2x2 numpy array
confusion_matrix = np.array([[TP, FN],
                             [FP, TN]])

# Plot the confusion matrix using Matplotlib
plt.figure(figsize=(6, 4))
plt.imshow(confusion_matrix, cmap="Blues", interpolation="nearest")

# Add labels and title
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Add the values to the matrix
for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="black")

# Set tick labels (x-axis: Predicted, y-axis: True)
plt.xticks([0, 1], ["Pop", "Classical"])
plt.yticks([0, 1], ["Pop", "Classical"])

# Show the plot
plt.colorbar()
plt.show()
