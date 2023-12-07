import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('/home/css142/FinalProject/RainerData.csv')
df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].apply(lambda t:t.month)
df['day'] = df['date'].apply(lambda t:t.day)
df['year'] = df['date'].apply(lambda t:t.year)

y_avg = df.groupby('year').mean(numeric_only=False)
years = np.arange(df['year'].iloc[0], df['year'].iloc[-1]+1, 1)
temps = np.array(y_avg['temp'])

def f(x, m, b):
  return m*x+b

(p, C) = opt.curve_fit(f, years, temps)

x_grid = np.linspace(1940, 2022, 1000)

plt.plot(y_avg['temp'])
plt.plot(x_grid, f(x_grid, *p), 'r:', label=f"{np.round(p[0], 2)} °C per year")
plt.title('Yearly Average Temperature \n Mount Rainer')
plt.xlabel('Year')
plt.ylabel('°C')
plt.legend()

plt.savefig('Temperature Graph')
