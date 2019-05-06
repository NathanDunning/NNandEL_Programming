import gplearn as gp
from gplearn.genetic import SymbolicClassifier
import numpy as np
import sys

# Global variables
feature_names = [
                "Clump Thickness", "Uniformity Cell Size", "Uniformity Cell Shape",
                "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
                "Bland Chromatin", "Normal Nucleoli", "Mitosis"
                ]

# Input Files
inputs = []

# Data sets
X_train = []
y_train = []
X_test = []
y_test = []

# How to split up the 699 data points
train_num_dat = 500
test_num_dat = 199


# Function to load the file
def load():
    global X_train, y_train, X_test, y_test, inputs


    if not (train_num_dat + test_num_dat == 699):
        raise AssertionError("Training and test split does not equal data points")

    # Load from the file
    file = open(r"./data/breast-cancer-wisconsin.data", "r")

    # Read the lines from the file
    dataset = file.readlines()

    for line in dataset:
        inputs.append(line)


    # Use the first set entries as training set
    for x in range(train_num_dat):
        line = inputs[x].split(",")

        # Line checks
        if not len(line) == 11:
            raise AssertionError("Expected 11 columns of data")

        # Append the values to temporary array
        inputSet = []
        # If value unknown, append -1
        for i in range(1, len(line)-1):
            if line[i] == "?":
                inputSet.append([-1])
            else:
                inputSet.append([int(line[i])])

        # Append values to training dataset
        X_train.append(inputSet)
        y_train.append(int(line[len(line)-1]))

    print("Loaded " + str(train_num_dat) + " elements into training set")

    # Using the rest of the data for test set
    for x in range(train_num_dat, len(inputs)):
        line = inputs[x].split(",")

        # Line checks
        if not len(line) == 11:
            raise AssertionError("Expected 11 columns of data")

        # Append the values to temporary array
        inputSet = []
        # If value unknown, append -1
        for i in range(1, len(line) - 1):
            if line[i] == "?":
                inputSet.append([-1])
            else:
                inputSet.append([int(line[i])])

        # Append values to test dataset
        X_test.append(inputSet)
        y_test.append(int(line[len(line) - 1]))

    print("Loaded " + str(test_num_dat) + " elements into test set")

    # Reshaping array
    X_train = np.reshape(X_train, (train_num_dat, 9))
    X_test = np.reshape(X_test, (test_num_dat, 9))

    # print(np.matrix(X_train))
    # print(np.matrix(y_train))
    # print(np.matrix(X_test))
    # print(np.matrix(y_test))


# Function to load the file
def loadFromFile(train = None, test = None):
    global X_train, y_train, X_test, y_test, inputs

    # If training and test set is given

    train_file = open(r"./data/" + str(train), "r")
    train_dataset = train_file.readlines()

    # Iterate through data and append
    for currentLine in train_dataset:
        splitLine = currentLine.split(",")

        # Line checks
        if not len(splitLine) == 11:
            raise AssertionError("Expected 11 columns of data")

        # Append the values to temporary array
        inputSet = []
        # If value unknown, append -1
        for i in range(1, len(splitLine) - 1):
            if splitLine[i] == "?":
                inputSet.append([-1])
            else:
                inputSet.append([int(splitLine[i])])

        # Append values to training dataset
        X_train.append(inputSet)
        y_train.append(int(splitLine[len(splitLine) - 1]))

    print("Loaded " + str(len(train_dataset)) + " elements into training set")


    test_file = open(r"./data/" + str(test), "r")
    test_dataset = test_file.readlines()

    # Iterate through data and append
    for currentLine in test_dataset:
        splitLine = currentLine.split(",")

        # Line checks
        if not len(splitLine) == 11:
            raise AssertionError("Expected 11 columns of data")

        # Append the values to temporary array
        inputSet = []
        # If value unknown, append -1
        for i in range(1, len(splitLine) - 1):
            if splitLine[i] == "?":
                inputSet.append([-1])
            else:
                inputSet.append([int(splitLine[i])])

        # Append values to training dataset
        X_test.append(inputSet)
        y_test.append(int(splitLine[len(splitLine) - 1]))

    X_train = np.reshape(X_train, (len(train_dataset), 9))
    X_test = np.reshape(X_test, (len(test_dataset), 9))

    print("Loaded " + str(len(test_dataset)) + " element into test set")


def train():
    est_gp = SymbolicClassifier(population_size=250, generations=20, tournament_size=20,
                                stopping_criteria=0.01, parsimony_coefficient=0.001,
                                p_crossover=0.9, p_subtree_mutation=0.05, p_hoist_mutation=0.0025,
                                p_point_mutation=0.01, p_point_replace=0.0025, verbose=1,
                                max_samples=0.9, feature_names=feature_names)

    est_gp.fit(X_train, y_train)
    print(est_gp._program)
    print(est_gp.score(X_train, y_train))
    print(est_gp.score(X_test, y_test))


if len(sys.argv) == 1:
    load()
elif len(sys.argv) == 2:
    print("If providing files, please provide both training and test")
    print("Else leave no arguments to use default set")
    exit("Invalid number of arguments")
elif len(sys.argv) == 3:
    loadFromFile(sys.argv[1], sys.argv[2])

train()
