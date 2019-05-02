
# Global variables
inp = []
out = []


# Function to load the file
def loadFile():
    # Load from the file
    file = open(r"./data/regression.txt", "r")

    # Read the lines from the file
    lines = file.readlines()

    for line in lines:
        # Skip the line we don't need
        if line[0] == "-":
            continue

        line = line.split()

        # Skip the line we don't need
        if line[0] == "x":
            continue

        inp.append(line[0])
        out.append(line[1])

    for x in range(0, len(inp)):
        print(inp[x])
        print(out[x])


loadFile()
