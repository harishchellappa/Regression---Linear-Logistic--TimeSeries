# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:39:49 2019

@author: haris
"""

import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\haris\Documents\stat 608\week5\HoustonChronicle.csv")
print(df[0:10], '\n')

df['interaction']=df['Year_Binary']*df['%Low income students']

model = sm.ols('%Repeating 1st Grade ~ %Low income students + interaction + Year_Binary', data=df)
results = model.fit()
print(results.summary(), "\n")

# Save Predictions
pred = results.fittedvalues
# Save Residuals
resid = results.resid
# Calculate Hii
results.HC2_se
het = results.het_scale # het = r^2 / (1-Hii)
h = 1.0 - (1.0/het)*resid**2 # h = Hii
# Calculate Standardized Residuals
std_resid = np.sqrt(het)/math.sqrt(results.mse_resid)
std_resid = pd.Series(std_resid) # Move into Pandas Series
# Correct for negative signs
for i in range(df.shape[0]):
    if resid[i] < 0:
        std_resid[i] = -std_resid[i]
# Calculate Cook's D
D = ((std_resid**2)*h)/(2*(1.0-h))
df2 = pd.concat([pred, resid, std_resid, D, h], axis=1, \
                keys=["Predicted", "Residual", "Std. Residual", \
                      "Cooks D", "H-Hat"])
df = df.join(df2)
print(df, "\n")
print("Rule of Thumb for Cook's D: ", 4/df.shape[0])

# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
# Peform analysis of variance on fitted linear model
anova_results = anova_lm(results)
print('\nANOVA results')
print(anova_results)

import seaborn as sns
sns.set(style="ticks", color_codes=True)
sns.catplot(x=df.BILL, y=df.LATE, data=df)

# Plot Predicted vs Std. Residuals
fig, ax = plt.subplots(figsize=(8,6))
ax.set_title("Predicted vs Standardized Residuals", \
fontweight="bold", fontsize="14")
ax.set_xlabel("LATE", fontweight="bold", fontsize="12")
ax.set_ylabel("Std. Residual", fontweight="bold", fontsize="12")
ax.plot(df['Predicted'], df['Std. Residual'], 'o', label="Data")
ax.axhline(y=0, linewidth=2, color='b', linestyle='--')
ax.axhline(y=2, linewidth=2, color='r', linestyle='-')
ax.axhline(y=-2, linewidth=2, color='r', linestyle='-')
plt.show()

# Plot Predicted vs Cook's D
fig, ax = plt.subplots(figsize=(8,6))
ax.set_title("Predicted vs Cook's D", \
fontweight="bold", fontsize="14")
ax.set_xlabel("LATE", fontweight="bold", fontsize="12")
ax.set_ylabel("Cook's D", fontweight="bold", fontsize="12")
ax.plot(df['Predicted'], df['Cooks D'], 'o', label="Data")
ax.axhline(y=4/df.shape[0], linewidth=2, color='r', linestyle='-')
plt.show()

# Plot Cook's D vs Std. Residuals
fig, ax = plt.subplots(figsize=(8,6))
ax.set_title("Cook's D vs Std. Residuals", \
fontweight="bold", fontsize="14")
ax.set_xlabel("Cook's D", fontweight="bold", fontsize="12")
ax.set_ylabel("Std. Residual", fontweight="bold", fontsize="12")
ax.plot(df['Cooks D'], df['Std. Residual'], 'o', label="Data")
ax.axhline(y=0, linewidth=2, color='b', linestyle='--')
ax.axhline(y=2, linewidth=2, color='r', linestyle='-')
ax.axhline(y=-2, linewidth=2, color='r', linestyle='-')
ax.axvline(x=0.0816, linewidth=2, color='g', linestyle='-')
plt.show()
