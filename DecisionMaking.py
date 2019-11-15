import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MachineMaintenance:

    def __init__(self):
        self.name = 'Machine Maintenance Problem'
        self.n_var = 500
        self.plot_limit_f1 = [0, 2000]
        self.plot_limit_f2 = [1350, 1500]
        #
        # Import parameters data from all csv files as global variables
        self.groups = pd.read_csv('./data/ClusterDB.csv', names=['Cluster', 'eta', 'beta'])
        self.equipments = pd.read_csv('./data/EquipDB.csv', names=['ID', 't0', 'Cluster', 'Failure cost'])
        self.maintenance_planning = pd.read_csv('./data/MPDB.csv', names=['Maintenance type', 'k', 'Maintenance cost'])
        #
        # Adjust Equipment Data Base
        self.equipments['eta'] = 0
        self.equipments['beta'] = 0
        for eq, r in self.equipments.iterrows():   # todo - improve this gambs - remove warning
            self.equipments['eta'].loc[eq] = self.groups['eta'].loc[r['Cluster'] - 1]
            self.equipments['beta'].loc[eq] = self.groups['beta'].loc[r['Cluster'] - 1]
        #
        # Define failure probabilities for all equipments
        self.p = np.zeros([500, 3])  # Pre-allocate variable
        self.define_failure_probability_for_all_equipments()  # ok!

    def define_failure_probability_for_all_equipments(self):
        # Calculate failure probability
        for i in range(500):
            for j in range(3):
                machine = self.equipments.loc[i]
                plan = self.maintenance_planning.loc[j]
                f_i_t0_k_delta_t = self.evaluate_failure_function(t=machine['t0'] + plan['k'] * 5, eta=machine['eta'],
                                                                  beta=machine['beta'])
                f_i_t0 = self.evaluate_failure_function(t=machine['t0'], eta=machine['eta'], beta=machine['beta'])
                numerator = f_i_t0_k_delta_t - f_i_t0
                denominator = 1 - f_i_t0
                self.p[i][j] = numerator / denominator
        return self.p

    def evaluate_failure_function(self, t, eta, beta):
        f = 1 - np.exp(-(t / eta) ** beta)
        return f

    def evaluate_maintenance_cost(self, x):  # function 1 - min
        machine_maintenance_cost = []
        for i in range(500):
            plan = self.maintenance_planning.loc[x[i]]
            machine_maintenance_cost.append(plan['Maintenance cost'])
        total_maintenance_cost = np.sum(machine_maintenance_cost)
        return total_maintenance_cost

    def evaluate_failure_cost(self, x):  # function 2 - min - ok!
        machine_failure_cost = []
        for i in range(500):
            machine = self.equipments.loc[i]
            plan = self.maintenance_planning.loc[x[i]]
            machine_failure_cost.append(machine['Failure cost'] * self.p[i][int(plan['Maintenance type'] - 1)])
        total_expected_failure_cost = np.sum(machine_failure_cost)
        return total_expected_failure_cost


#
# Read file with solutions and pre allocate variables
solutions = pd.read_csv('Pareto_Frontier', header=None)
solutions = np.array(solutions)                 # transform it in a np array
solutions = np.unique(solutions, axis=0)        # Remove duplicates
data = pd.DataFrame(index=np.arange(len(solutions)), columns=['Maintenance Cost', 'Failure Cost', 'A_{Maintenance}',
                                                              'A_{Failure}', 'D', 'Variables'])
#
# Create data frame with the evaluation of each solution
for k in range(len(solutions)):
    aux_1 = MachineMaintenance().evaluate_maintenance_cost(solutions[k])
    aux_2 = MachineMaintenance().evaluate_failure_cost(solutions[k])
    data.loc[k] = pd.Series({'Maintenance Cost': aux_1, 'Failure Cost': aux_2, 'Variables': solutions[k]})
    print(str(k))
data = data.fillna(0)  # with 0s rather than NaNs
# ______________________________________________________________________________________________________________________
# AHP to select the importance weights of the objective functions
Sofia_opinion = [[1, 1/3],
                 [3, 1]]
Servilio_opinion = [[1, 1/3],
                    [3, 1]]
Rafael_opinion = [[1,   2],
                  [1/2, 1]]
#
# build aggregation of everybody's opinion
group_opinion = [Sofia_opinion, Servilio_opinion, Rafael_opinion]
group_opinion = np.mean(group_opinion, axis=0)      # everybody has the same importance
col_sum = np.sum(group_opinion, axis=0)
normalized_group_opinion = group_opinion / col_sum
ahp_criteria_weight = np.mean(normalized_group_opinion, axis=1)
#
# Calculate consistency
consistency_matrix = group_opinion * ahp_criteria_weight
weighted_sum_value = np.sum(consistency_matrix, axis=1)
lambda_vector = weighted_sum_value/ahp_criteria_weight
lambda_max = np.mean(lambda_vector)
n = consistency_matrix.shape[0]
consistency_index = (lambda_max - n)/(n - 1)
# ======================================================================================================================
# Fuzzy multi-criteria decision making by Bellman-Zadeh approach
#
# Definition of lambda (importance coefficient os objective functions)
lambda_maintenance = ahp_criteria_weight[0]
lambda_failure = ahp_criteria_weight[1]
#
# convert Data Frame to np.array
data_matrix = np.array(data)
for k in range(len(solutions)):
    #
    # Define normalized fuzzy membership function for Maintenance Cost
    numerator = data['Maintenance Cost'].max() - data['Maintenance Cost'][k]
    denominator = data['Maintenance Cost'].max() - data['Maintenance Cost'].min()
    data_matrix[k][2] = (numerator/denominator)**lambda_maintenance
    #
    # and for Failure Cost
    numerator = data['Failure Cost'].max() - data['Failure Cost'][k]
    denominator = data['Failure Cost'].max() - data['Failure Cost'].min()
    data_matrix[k][3] = (numerator/denominator)**lambda_failure
    #
    # Define the fuzzy solution af the problem
    data_matrix[k][4] = min(data_matrix[k][2], data_matrix[k][3])
#
# then return the array to data frame
data = pd.DataFrame(data_matrix,  columns=['Maintenance Cost', 'Failure Cost', 'A_{Maintenance}', 'A_{Failure}', 'D',
                                           'Variables'])
#
# Print the harmonious solution
harmonious_solution = data.iloc[np.argmax(data_matrix[:, 4])]
print('======= Harmonious Solution =======')
print(harmonious_solution)
#
# Plot solutions
plt.scatter(data['Maintenance Cost'], data['Failure Cost'], c='b')
plt.scatter(harmonious_solution['Maintenance Cost'], harmonious_solution['Failure Cost'], c='r', marker='X', s=130)
plt.grid()
plt.xlabel('Maintenance cost')
plt.ylabel('Failure cost')
plt.title(str(MachineMaintenance().name)+' for 500 variables')
plt.show()


print('======= END =======')
