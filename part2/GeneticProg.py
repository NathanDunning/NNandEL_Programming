from gplearn.genetic import SymbolicRegressor

# Global variables
X_train = []
y_train = []


# Function to load the file
def load():

    # Load from the file
    file = open(r"./data/regression.txt", "r")

    # Read the lines from the file
    lines = file.readlines()

    global X_train, y_train
    for line in lines:
        # Skip the line we don't need
        if line[0] == "-":
            continue

        line = line.split()

        # Skip the line we don't need
        if line[0] == "x":
            continue

        X_train.append([float(line[0])])
        y_train.append(float(line[1]))

    # print(X_train)
    # print(y_train)


def train():
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=20, stopping_criteria=0.001,
                               p_crossover=0.8, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.05,
                               max_samples=0.9, verbose=1, metric='mean absolute error',
                               parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_train, y_train)
    print(est_gp._program)
    print(est_gp.get_params())
    print(est_gp.predict(X_train))


load()
train()
