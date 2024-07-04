# -------- CONFIGURATION -------------

# Global Swarm Configurations
MIN_SUPPORT = 0.5
MAX_ITERATIONS = 1
N_VAR = 1  # DO NOT CHANGE

# ACO-GRAANK Configurations:
EVAPORATION_FACTOR = 0.5

# GA-GRAANK Configurations:
N_POPULATION = 5
PC = 0.5
GAMMA = 1  # Cross-over
MU = 0.9  # Mutation
SIGMA = 0.9  # Mutation

# PSO-GRAANK Configurations:
VELOCITY = 0.9  # higher values helps to move to next number in search space
PERSONAL_COEFF = 0.01
GLOBAL_COEFF = 0.9
TARGET = 1
TARGET_ERROR = 1e-6
N_PARTICLES = 5

# PLS-GRAANK Configurations
STEP_SIZE = 0.5

# CluGRAD Configurations
ERASURE_PROBABILITY = 0.5  # determines the number of pairs to be ignored
SCORE_VECTOR_ITERATIONS = 10  # maximum iteration for score vector estimation
