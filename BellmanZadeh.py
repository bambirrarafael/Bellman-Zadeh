import math
import random
import numpy as np
import pandas as pd
import scipy.optimize as opt
from copy import deepcopy as deep_copy
import sympy.combinatorics.graycode as gc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

# define global variables
global_data_NEWAVE = 'C:/Users/bambi/PycharmProjects/NSGA2/data/NEWAVE_DATA.xlsx'
global_data_portfolio = 'C:/Users/bambi/PycharmProjects/NSGA2/data/PORTFOLIO_DATA.xlsx'
global_PLD = np.array(pd.read_excel(global_data_NEWAVE, 'PLD', header=None))
global_GSF = np.array(pd.read_excel(global_data_NEWAVE, 'GSF', header=None))
global_bought_energy = np.array(pd.read_excel(global_data_portfolio, 'BOUGHT'))
global_bought_price = np.array(pd.read_excel(global_data_portfolio, 'BOUGHT_PRICE'))
global_sold_energy = np.array(pd.read_excel(global_data_portfolio, 'SOLD'))
global_sold_price = np.array(pd.read_excel(global_data_portfolio, 'SOLD_PRICE'))
global_gFIS_HIDR = np.array(pd.read_excel(global_data_portfolio, 'HIDR'))


class EnergyRiskAndReturn:

    def __init__(self, nvar):
        # Parameters of simulation
        self.name = 'Energy Risk and Return'
        self.n_var = nvar
        self.upper_bound = np.ones(self.n_var) * 200
        self.lower_bound = np.ones(self.n_var) * -100
        self.plot_limit_f1 = [-250000000.0, 100000.0]
        self.plot_limit_f2 = [-1000000.0, 300000000.0]
        self.f = [0, 0]
        #
        # Parameters of the functions
        self.horizon = 12               # months
        self.alpha = 0.05               # CVaR's limit
        self.inflation = 0.0048         # VPL's rate
        self.price_delta_contract = 200     # R$/MWh
        #
        # Inputs from portfolio and NEWAVE
        self.newave_data = global_data_NEWAVE
        self.portfolio_data = global_data_portfolio
        self.PLD = global_PLD
        self.GSF = global_GSF
        self.bought_energy = global_bought_energy
        self.bought_price = global_bought_price
        self.sold_energy = global_sold_energy
        self.sold_price = global_sold_price
        self.gFIS_HIDR = global_gFIS_HIDR
        self.n_scen = len(self.PLD)
        self.n_worst_scenarios = int(self.alpha * self.n_scen)
        self.n_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]     # number of hours of each month
        self.gen_cost_HIDR = 3.3105     # US$/MWm [h]

    def function_1_max(self, x):    # Return    - OK
        #
        # Pre-allocate variables
        aux_bought = np.zeros([1, self.horizon])
        aux_sold = np.zeros([1, self.horizon])
        aux_gFIS_HIDR = np.zeros([1, self.horizon])
        aux_income = np.zeros(np.shape(self.PLD))
        net_present_value = np.zeros(len(self.PLD))
        #
        # Calculate auxiliary variables of portfolio - constants
        for i in range(len(self.bought_price)):
            aux_bought = aux_bought + self.bought_energy[i, :]*self.bought_price[i]
        for i in range(len(self.sold_price)):
            aux_sold = aux_sold + self.sold_energy[i, :]*self.sold_price[i]
        for i in range(len(self.gFIS_HIDR)):
            aux_gFIS_HIDR = aux_gFIS_HIDR + self.gFIS_HIDR[i, :]*self.gen_cost_HIDR
        #
        # Calculate the monthly income for each scenario
        for i in range(self.n_scen):
            # but first, calculate portfolio's exposition
            resources = np.sum(self.bought_energy, axis=0) + self.GSF[i, :]*np.sum(self.gFIS_HIDR, axis=0)
            exposition = resources - np.sum(self.sold_energy, axis=0) - x
            #
            # and then the income
            aux_income[i, :] = self.n_hours*(self.price_delta_contract*x + aux_sold + exposition*self.PLD[i, :]
                                             - aux_bought - aux_gFIS_HIDR)
            net_present_value[i] = np.npv(self.inflation, aux_income[i, :])
        expected_income = np.mean(net_present_value)
        self.f = - expected_income  # maximize it
        return self.f

    def function_1_min(self, x):  # Return    - OK - # Huge Gambs
        #
        # Pre-allocate variables
        aux_bought = np.zeros([1, self.horizon])
        aux_sold = np.zeros([1, self.horizon])
        aux_gFIS_HIDR = np.zeros([1, self.horizon])
        aux_income = np.zeros(np.shape(self.PLD))
        net_present_value = np.zeros(len(self.PLD))
        #
        # Calculate auxiliary variables of portfolio - constants
        for i in range(len(self.bought_price)):
            aux_bought = aux_bought + self.bought_energy[i, :] * self.bought_price[i]
        for i in range(len(self.sold_price)):
            aux_sold = aux_sold + self.sold_energy[i, :] * self.sold_price[i]
        for i in range(len(self.gFIS_HIDR)):
            aux_gFIS_HIDR = aux_gFIS_HIDR + self.gFIS_HIDR[i, :] * self.gen_cost_HIDR
        #
        # Calculate the monthly income for each scenario
        for i in range(self.n_scen):
            # but first, calculate portfolio's exposition
            resources = np.sum(self.bought_energy, axis=0) + self.GSF[i, :] * np.sum(self.gFIS_HIDR, axis=0)
            exposition = resources - np.sum(self.sold_energy, axis=0) - x
            #
            # and then the income
            aux_income[i, :] = self.n_hours * (self.price_delta_contract * x + aux_sold + exposition * self.PLD[i, :]
                                               - aux_bought - aux_gFIS_HIDR)
            net_present_value[i] = np.npv(self.inflation, aux_income[i, :])
        expected_income = np.mean(net_present_value)
        self.f = expected_income   # Gambs
        return self.f

    def function_2_min(self, x):    # risk or CVaR  - Ok
        #
        # Pre-allocate variables
        aux_bought = np.zeros([1, self.horizon])
        aux_sold = np.zeros([1, self.horizon])
        aux_gFIS_HIDR = np.zeros([1, self.horizon])
        aux_income = np.zeros(np.shape(self.PLD))
        #
        # Calculate auxiliary variables of portfolio - constants
        for i in range(len(self.bought_price)):
            aux_bought = aux_bought + self.bought_energy[i, :]*self.bought_price[i]
        for i in range(len(self.sold_price)):
            aux_sold = aux_sold + self.sold_energy[i, :]*self.sold_price[i]
        for i in range(len(self.gFIS_HIDR)):
            aux_gFIS_HIDR = aux_gFIS_HIDR + self.gFIS_HIDR[i, :]*self.gen_cost_HIDR
        #
        # Calculate the monthly income for each scenario
        for i in range(self.n_scen):
            # but first, calculate portfolio's exposition
            resources = np.sum(self.bought_energy, axis=0) + self.GSF[i, :]*np.sum(self.gFIS_HIDR, axis=0)
            exposition = resources - np.sum(self.sold_energy, axis=0) - x
            #
            # and then the income
            aux_income[i, :] = self.n_hours*(self.price_delta_contract * x + aux_sold + exposition *self.PLD[i, :]
                                             - aux_bought - aux_gFIS_HIDR)
        horizon_income = np.sum(aux_income, axis=1)
        expected_income = np.mean(horizon_income)
        #
        # Calculate the worst incomes
        sorted_horizon_income = np.sort(horizon_income)
        worst_incomes = sorted_horizon_income[:self.n_worst_scenarios]
        expected_tail_loss = np.mean(worst_incomes)
        conditional_value_at_risk = expected_income - expected_tail_loss
        self.f = conditional_value_at_risk
        return self.f

    def function_2_max(self, x):    # risk or CVaR  - Ok    # Gambs
        #
        # Pre-allocate variables
        aux_bought = np.zeros([1, self.horizon])
        aux_sold = np.zeros([1, self.horizon])
        aux_gFIS_HIDR = np.zeros([1, self.horizon])
        aux_income = np.zeros(np.shape(self.PLD))
        #
        # Calculate auxiliary variables of portfolio - constants
        for i in range(len(self.bought_price)):
            aux_bought = aux_bought + self.bought_energy[i, :]*self.bought_price[i]
        for i in range(len(self.sold_price)):
            aux_sold = aux_sold + self.sold_energy[i, :]*self.sold_price[i]
        for i in range(len(self.gFIS_HIDR)):
            aux_gFIS_HIDR = aux_gFIS_HIDR + self.gFIS_HIDR[i, :]*self.gen_cost_HIDR
        #
        # Calculate the monthly income for each scenario
        for i in range(self.n_scen):
            # but first, calculate portfolio's exposition
            resources = np.sum(self.bought_energy, axis=0) + self.GSF[i, :]*np.sum(self.gFIS_HIDR, axis=0)
            exposition = resources - np.sum(self.sold_energy, axis=0) - x
            #
            # and then the income
            aux_income[i, :] = self.n_hours*(self.price_delta_contract * x + aux_sold + exposition *self.PLD[i, :]
                                             - aux_bought - aux_gFIS_HIDR)
        horizon_income = np.sum(aux_income, axis=1)
        expected_income = np.mean(horizon_income)
        #
        # Calculate the worst incomes
        sorted_horizon_income = np.sort(horizon_income)
        worst_incomes = sorted_horizon_income[:self.n_worst_scenarios]
        expected_tail_loss = np.mean(worst_incomes)
        conditional_value_at_risk = expected_income - expected_tail_loss
        self.f = - conditional_value_at_risk    # Gambs max it
        return self.f

    def constrains(self, x):
        penalties = 0.0
        return penalties

# define functions to optmize
problem = EnergyRiskAndReturn(nvar=1)
income_min = problem.function_1_min
income_max = problem.function_1_max
risk_min = problem.function_2_min
risk_max = problem.function_2_max
#
# Set initial guess
x0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
x0_bounds = [(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0),(-100.0, 200.0)]
#
# Find minimum and maximum value for each function
res_min_income = opt.minimize(income_min, x0, bounds=x0_bounds)
res_max_income = opt.minimize(income_max, x0, bounds=x0_bounds)
res_min_risk = opt.minimize(risk_min, x0, bounds=x0_bounds)
res_max_risk = opt.minimize(risk_max, x0, bounds=x0_bounds)


def MaxMin(x, f1_min=res_min_income, f2_min=res_min_risk, f1_max=res_max_income, f2_max=res_max_risk):
    #
    # Calculate the normalized value for each function
    mu_1 = (-income_max(x) - f1_min.fun) / (-f1_max.fun - f1_min.fun)     # Maximize
    mu_2 = (-f2_max.fun - risk_min(x)) / (-f2_max.fun - f2_min.fun)       # Minimize
    d = np.min(np.array([mu_1, mu_2]))
    return - d


res_max_min = opt.minimize(MaxMin, x0, bounds=x0_bounds)

print('----- Max Min Result -----')
print(res_max_min)
print('Value of income : ' + str(-income_max(res_max_min.x)))
print('Value of risk : ' + str(risk_min(res_max_min.x)))
print('--- Fim ---')

