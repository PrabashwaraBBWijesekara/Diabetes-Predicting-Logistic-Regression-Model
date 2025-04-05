import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#TODO

diabetesData = pd.read_csv("DiabetesTrain.csv", usecols=range(9))
diabetesData.sample(5)

#TODO

total_patients = len(diabetesData)

#P(A)
P_A = (diabetesData['BMI'] < 25).sum() / total_patients
print("P_A: ", P_A)

#P(B)
P_B = (diabetesData['Glucose'] >100).sum() / total_patients
print("P_B: ", P_B)

#P(C)
P_C = (diabetesData['Pregnancies'] >2).sum() / total_patients
print("P_C: ", P_C)

#P(D)
P_D = (diabetesData['Outcome'] == 1).sum() / total_patients
print("P_D: ", P_D)

#P(A, D)
P_A_D =((diabetesData['BMI'] < 25) & (diabetesData['Outcome'] == 1)).sum()/total_patients
print("P_A_D: ", P_A_D)

#P(B, D)
P_B_D =((diabetesData['Glucose'] >100) & (diabetesData['Outcome'] == 1)).sum()/total_patients
print("P_B_D: ", P_B_D)

#P(C, D)
P_C_D =((diabetesData['Pregnancies'] >2) & (diabetesData['Outcome'] == 1)).sum()/total_patients
print("P_C_D: ", P_C_D)

#Indicate which one out of A, B, C contributes the most towards high risk of diabetes.
#Assign one of 'A', 'B', 'C' to the following variable Q2, indicating your answer.
#Hint: Compute the necessary conditional probabilities and then compare.

#Conditional  probabilities (ex:- Probabilty of D Given A )
P_D_G_A = P_A_D / P_A if P_A > 0 else 0

P_D_G_B = P_B_D / P_B if P_B > 0 else 0

P_D_G_C = P_C_D / P_C if P_C > 0 else 0

Q2 = max(('A', P_D_G_A), ('B', P_D_G_B), ('C', P_D_G_C), key=lambda x: x[1])[0]
print("Q2: ",Q2)


diabetesX = diabetesData.to_numpy()

diabetesX = diabetesX-np.mean(diabetesX,axis=0)

cov = (diabetesX.T @ diabetesX)/(diabetesX.shape[0])


#TODO
# Extract variances
var = np.diag(cov)

# Compute square root of variances
sqrtVar = np.sqrt(var)
# Compute varmat using outer product
varmat = np.outer(sqrtVar, sqrtVar)


#TODO
varmat[varmat == 0] = 1 #prevent division by zero
corr = cov / varmat

#Plot the heatmap.
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


#Corr(BMI, Outcome)
Corr1 = round(corr[5,8],2)
print(Corr1)

#Corr(Glucose, Outcome)
Corr2 = round(corr[1,8],2)
print(Corr2)

#Corr(Pregnancies, Outcome)
Corr3 = round(corr[0,8],2)
print(Corr3)

#print(Corr1, Corr2, Corr3)

#Out of the 8 features, which two features are the most correlated. Fill in the list variable bestcorr

# Get upper triangle of correlation matrix, ignoring diagonal
upper_triangle = np.triu(corr, k=1)

# Find the indices of the maximum absolute correlation value
max_corr_index = np.unravel_index(np.argmax(np.abs(upper_triangle)), corr.shape)
bestcorr_index = list(max_corr_index)
bestcorr = [diabetesData.columns[bestcorr_index[0]], diabetesData.columns[bestcorr_index[1]]]
print(bestcorr)


def normalizeData(X, mean = np.array([]), std = np.array([])):

    #TODO
    X=np.array(X)

    # Implement such that if the mean and std are empty arrays then mean and std is calculated here, else use the mean and std passed in as parameters
    if mean.size == 0 and std.size == 0:
        mean = np.mean(X, axis=0)  # Column-wise mean
        std = np.std(X, axis=0, ddof=1)  # Column-wise sample std (ddof=1)
    std[std == 0] = 1

    normalized_X = (X-mean)/std

    return normalized_X, mean, std



#Sigmoid function
def sigmoid(t):

    #TODO
    t = np.clip(t, -500, 500) 
    sig = 1/(1+np.exp(-t))
    
    return sig


#Derivative of sigmoid function
def derivSigmoid(p):

    #TODO
    
    deriv_p = p*(1-p)

    return deriv_p



def sigProg(X, w, b):

    #TODO
    p = sigmoid(np.dot(X,w)+b)
    
    return p

def gradient(X, y, w, b, reg = "none", Lambda = 0.1):

    #TODO
    # Number of data points
    N = X.shape[0]
    p = sigProg(X, w, b)
    p_prime = derivSigmoid(p)
    error = (p - y) * p_prime

    grad_w = (1 / N) * np.dot(X.T, error) 
    deriv_b = (1 / N) * np.sum(error)

    if reg == "ridge":
        grad_w += Lambda * w
    elif reg == "lasso":
        grad_w += Lambda * np.sign(w)

    return grad_w, deriv_b


def grad_descent(grad_w, deriv_b, w, b, eta = 0.01):

    #TODO
    w = w - eta*grad_w
    
    b = b - eta*deriv_b

    return w, b


# Compute the MSE loss
def compute_loss(X, y, w, b, reg="none", Lambda=0.1):
    p = sigmoid(np.dot(X, w) + b)
    loss = np.mean((p - y) ** 2) 
    if reg == "ridge":
        loss += Lambda * np.sum(w**2)
    elif reg == "lasso":
        loss += Lambda * np.sum(np.abs(w))
    return loss



def train(X, y, reg = 'none', Lambda = 0.1, eta = 0.01, max_iter = 1000):

    #TODO

    # Number of features
    d = X.shape[1]

    w = np.random.randn(d)  # Small random values for weights
    b = np.random.randn()  # Initialize bias to small random value

    losses = []
    w_norms = []

    for _ in range(max_iter):
        grad_w, deriv_b = gradient(X, y, w, b, reg, Lambda)
        w, b = grad_descent(grad_w, deriv_b, w, b, eta)
        losses.append(compute_loss(X, y, w, b, reg, Lambda))
        w_norms.append(np.linalg.norm(w))

    return w, b, losses, w_norms


def predict(X, w, b):

    #TODO
    p = sigProg(X, w, b)
    yhat = (p >= 0.5).astype(int)

    return yhat


def k_fold_cross_validation(X, y, eta_values=[0.001, 0.01, 0.1], lambda_values=[0.01, 0.1, 1.0], regularization_types=["none", "ridge", "lasso"], K=5):
    # Shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(len(X))

    # Split indices into K folds
    fold_size = len(X) // K
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(K)]

    best_hyperparams = None
    best_error = float("inf")  # Track lowest classification error

    # Iterate over all hyperparameter combinations
    for eta in eta_values:
        for Lambda in lambda_values:
            for reg in regularization_types:
                errors = []  # Store classification errors for each fold
                
                for k in range(K):
                    # Get validation fold
                    val_idx = folds[k]
                    X_val, y_val = X[val_idx], y[val_idx]

                    # Get training data (all other folds)
                    train_idx = np.hstack([folds[i] for i in range(K) if i != k])
                    X_train_cv, y_train_cv = X[train_idx], y[train_idx]

                    # Train model
                    w, b, _, _= train(X_train_cv, y_train_cv, reg=reg, Lambda=Lambda, eta=eta, max_iter=1000)

                    # Predict on validation set
                    y_val_pred = predict(X_val, w, b)

                    # Compute classification error
                    error = np.mean(y_val_pred != y_val)
                    errors.append(error)

                # Compute average error across all folds
                avg_error = np.mean(errors)

                # Track best hyperparameters
                if avg_error < best_error:
                    best_error = avg_error
                    best_hyperparams = (eta, Lambda, reg)
    return best_hyperparams



#TODO
#Train and evaluate your model.
X_train = diabetesData.iloc[:, :-1].values
y_train = diabetesData.iloc[:, -1].values 

X_train, mean, std = normalizeData(X_train)


best_hype_params = k_fold_cross_validation(X_train,y_train)
print("(eta, lambda, reg): ",best_hype_params)
# Train the logistic regression model
w, b,losses, w_norms = train(X_train, y_train, best_hype_params[2], best_hype_params[1], best_hype_params[0], max_iter=1000)

#w, b,losses, w_norms = train(X_train, y_train, "ridge" ,0.1 , 0.1, max_iter=1000)

# Predict on training data
y_pred = predict(X_train, w, b)
classification_error = np.mean(y_pred != y_train)
print(classification_error)

accuracy = np.mean(y_pred == y_train)
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs. Iterations")
plt.show()


plt.plot(w_norms, color='r')
plt.xlabel("Iterations")
plt.ylabel("L2 Norm of Weights")
plt.title("L2 Norm vs. Iterations")
plt.show()


#TODO

diabetesTest = pd.read_csv("DiabetesTest.csv")

X_test_norm, _, _ = normalizeData(diabetesTest, mean, std)

y_test_pred = predict(X_test_norm, w, b)

print(y_test_pred)