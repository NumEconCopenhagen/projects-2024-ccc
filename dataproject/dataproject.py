# %% [markdown]
# # Import and Export of good and services

# %% [markdown]
# Imports and set magics:

# %%
#%pip install pandas-datareader
#%pip install git+https://github.com/alemartinello/dstapi
import numpy as np
import pandas as pd
import datetime

import pandas_datareader # install with `pip install pandas-datareader`
from dstapi import DstApi # install with `pip install git+https://github.com/alemartinello/dstapi`

import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})


# %% [markdown]
# # Import

# %%
#Import UHM dataset for DST 
trade = DstApi('UHM') 

# Retrieve a summary of the table which contains variable names in English
tabsum = trade.tablesummary(language='en')
# Display the summary of the table.



#Define base parameters for the data retrieval
params = trade._define_base_params(language='en')  # Returns a view, that we can edit
params

# Print out the base parameters, set the values for the variables
variables = params['variables']
variables[0]['values'] = ['1.A']
variables[1]['values'] =['1']
variables[2]['values'] = ['W1']
variables[3]['values'] = ['93']
variables[4]['values'] = ['2']
variables[5]['values'] = ['2022M01', '2022M02', '2022M03', '2022M04', '2022M05', '2022M06', '2022M07', '2022M08', '2022M09', '2022M10', '2022M11', '2022M12',
'2023M01', '2023M02', '2023M03', '2023M04', '2023M05', '2023M06', '2023M07', '2023M08', '2023M09', '2023M10', '2023M11', '2023M12',
'2024M01', '2024M02']
params

# Use the modified parameters to get data from the trade API and store it
data = trade.get_data(params=params)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Ensure 'INDHOLD' column is converted to numeric if necessary
df['INDHOLD'] = pd.to_numeric(df['INDHOLD'], errors='coerce')

# Display standard descriptive statistics for 'INDHOLD'
descriptive_stats_indhold = df['INDHOLD'].describe()

# Optional: Display the descriptive statistics in a more readable format, such as a table in a Jupyter Notebook
from IPython.display import display, HTML
display(HTML(descriptive_stats_indhold.to_frame().to_html()))

# %% [markdown]
# The descriptive statistics for import show a mean of approximately 141,623 mio., with a standard deviation of about 6,658 mio., indicating moderate variability. The minimum value is 128,759 mio. and the maximum is 160,425 mio., with the 25th, 50th, and 75th percentiles at 137,824 mio., 140,974 mio., and 145,841 mio. respectively.

# %%
#Convert the values in the INDHOLD column to a numeric data type
data['INDHOLD'] = pd.to_numeric(data['INDHOLD'], errors='coerce').fillna(0).astype(int)

plt.figure(figsize=(15, 8))  # Set the size of the graph
plt.plot(data['TID'], data['INDHOLD'], marker='o')  # Plot the line graph with markers

# Format the y-axis labels with commas for thousands
plt.gca().get_yaxis().set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.title('Monthly Import', fontsize=20)  # Set the title of the graph
plt.xlabel('Month', fontsize=14)  # Set the label for the x-axis
plt.ylabel('Import', fontsize=14)  # Set the label for the y-axis

plt.grid(True)  # Enable the grid
plt.tight_layout()  # Adjust the layout
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

plt.show()  # Display the graph


# %% [markdown]
# The graph displays Denmark's seasonally adjusted monthly import values from November 2021 to February 2024, smoothing out periodic fluctuations to reveal underlying trends. A noteworthy feature is the pronounced volatility in imports month-to-month, suggesting responsive adjustments to economic conditions, policy changes, or shifts in global market dynamics. The most striking peak occurs in July 2022, where imports surge past 150,000 million DKK, hinting at either a temporary spike in demand for foreign goods and services or a reflection of broader economic expansion during that period.
# 
# Equally important are the distinct dips observed, particularly in December 2022 and August 2023, which could be attributed to factors such as domestic policy impacts, shifts in consumer behavior, or external economic shocks. The graph does not present a clear upward or downward long-term trajectory but rather a zigzag pattern, emphasizing the dynamic nature of import activity.
# 
# From an economic standpoint, these fluctuations are significant as they may influence Denmark's trade balance and, consequently, its currency valuation and fiscal policies. The lack of a persistent trend in either direction suggests that Denmark's economy has not consistently moved towards either a stronger reliance on imports or a drive towards import substitution.

# %% [markdown]
# # Export

# %%
#Import UHM dataset for DST 
Export = DstApi('UHM') 
# Retrieve a summary of the table which contains variable names in English
tabsum = Export.tablesummary(language='en')



#Define base parameters for the data retrieval
params = Export._define_base_params(language='en')  # Returns a view, that we can edit
params

# Print out the base parameters, set the values for the variables
variables = params['variables']
variables[0]['values'] = ['1.A']
variables[1]['values'] =['2']
variables[2]['values'] = ['W1']
variables[3]['values'] = ['93']
variables[4]['values'] = ['2']
variables[5]['values'] = ['2022M01', '2022M02', '2022M03', '2022M04', '2022M05', '2022M06', '2022M07', '2022M08', '2022M09', '2022M10', '2022M11', '2022M12',
'2023M01', '2023M02', '2023M03', '2023M04', '2023M05', '2023M06', '2023M07', '2023M08', '2023M09', '2023M10', '2023M11', '2023M12',
'2024M01', '2024M02']
params

# Use the modified parameters to get data from the trade API and store it
Export_data = Export.get_data(params=params)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(Export_data)

# Ensure 'INDHOLD' column is converted to numeric if necessary
df['INDHOLD'] = pd.to_numeric(df['INDHOLD'], errors='coerce')

# Display standard descriptive statistics for 'INDHOLD'
descriptive_stats_indhold = df['INDHOLD'].describe()

# Optional: Display the descriptive statistics in a more readable format, such as a table in a Jupyter Notebook
from IPython.display import display, HTML
display(HTML(descriptive_stats_indhold.to_frame().to_html()))


# %% [markdown]
# The descriptive statistics for export show a mean value of approximately 162,559 mio., with a standard deviation of around 7,658 mio., indicating moderate variability. The minimum value is 152,222 mio. and the maximum is 183,460 mio., with the 25th, 50th, and 75th percentiles at 157,031 mio., 161,485 mio., and 166,645 mio. respectively, suggesting a fairly symmetric distribution around the mean.

# %%
#Convert the values in the INDHOLD column to a numeric data type
Export_data['INDHOLD'] = pd.to_numeric(Export_data['INDHOLD'], errors='coerce').fillna(0).astype(int)

plt.figure(figsize=(15, 8))  # Set the size of the graph
plt.plot(Export_data['TID'], Export_data['INDHOLD'], marker='o')  # Plot the line graph with markers

# Format the y-axis labels with commas for thousands
plt.gca().get_yaxis().set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.title('Monthly Export', fontsize=20)  # Set the title of the graph
plt.xlabel('Month', fontsize=14)  # Set the label for the x-axis
plt.ylabel('Export', fontsize=14)  # Set the label for the y-axis

plt.grid(True)  # Enable the grid
plt.tight_layout()  # Adjust the layout
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

plt.show()  # Display the graph

# %% [markdown]
# 
# This graph depicts Denmark's monthly export values of goods and services from November 2021 to February 2024, with the data being seasonally adjusted to account for periodic variations. Initial observations highlight a marked variability in export levels, which could be reflective of Denmark's economic resilience, adaptability to market conditions, and the competitiveness of Danish goods and services in the global market.
# 
# There is a sharp peak in export value in June 2022, where it reaches nearly 180,000 million DKK, potentially signifying a period of high economic activity or the fruition of favorable trade conditions or agreements. Following this peak, there is a significant decline, reaching its nadir around September 2022, which could be indicative of economic contraction, a temporary decrease in external demand, or issues such as production bottlenecks.
# 
# The overall pattern does not suggest a consistent long-term increase or decrease in export values but rather a series of rises and falls. This could point to the Danish economy's cyclical nature and its possible sensitivity to external economic shocks or seasonal industry patterns.
# 
# The rise in export values towards the end of the period, particularly noticeable in November 2023, may be a sign of economic recovery or an effective response to market opportunities. Considering Denmark's export strategy, this rebound could be the result of targeted economic policies or improved competitiveness in key sectors. The graph's implications for Denmark's trade balance are vital, as the relationship between import and export values can affect the country's current account balance.

# %% [markdown]
# 
# ### Net Export Analysis
# 
# In this section, we calculate and visualize the net export values over time using the actual data extracted from the API and merging the two data sets. Net export is calculated as the difference between export and import values. This metric helps to understand the trade balance and economic health of a country.
# 

# %%

# Calculate net export (export - import)
merged = Export_data['INDHOLD'] - data['INDHOLD']

# Plotting Net Export Values
plt.figure(figsize=(15, 8))
plt.plot(Export_data['TID'], merged,  marker='o')  # Plot the line graph with markers

# Format the y-axis labels with commas for thousands
plt.gca().get_yaxis().set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.title('Monthly Net Export', fontsize=20)  # Set the title of the graph
plt.xlabel('Month', fontsize=14)  # Set the label for the x-axis
plt.ylabel('Net Export', fontsize=14)  # Set the label for the y-axis

plt.grid(True)  # Enable the grid
plt.tight_layout()  # Adjust the layout
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

plt.show()  # Display the graph

# %% [markdown]
# ### Net Export Analysis
# 
# The figure shows the monthly net export values from January 2022 to February 2024. Net export, calculated as exports minus imports, provides insight into the trade balance. 
# 
# The values exhibit significant fluctuations, with the highest net export in July 2022 at approximately 30,000 million DKK, indicating increased exports or reduced imports. A notable low occurs in October 2022, dropping to around 10,000 million DKK, suggesting increased imports or decreased exports. These patterns may reflect seasonal trends and underlying economic factors affecting trade dynamics. Understanding these variations helps in assessing the country's economic health and trade balance over time.


