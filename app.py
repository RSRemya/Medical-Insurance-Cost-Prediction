import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import squarify
import os
import warnings
warnings.filterwarnings('ignore')
######################################################################################################
st.set_page_config(page_title="Insurance!!!", page_icon=":bar_chart:",layout="wide")
st.title(" :bar_chart: Data Analysis - Insurance")

# Data Ingestion
df = pd.read_csv("new_insurance.csv")
############################################################################################

# Sidebar filters
sex_filter = st.sidebar.multiselect('Select Sex', df['sex'].unique(), default=df['sex'].unique())
region_filter = st.sidebar.multiselect('Select Region', df['region'].unique(), default=df['region'].unique())
children_filter = st.sidebar.multiselect('Select Number of Children', df['children'].unique(), default=df['children'].unique())
smoker_filter = st.sidebar.multiselect('Select Smoker Status', df['smoker'].unique(), default=df['smoker'].unique())

# Apply filters to the dataframe
filtered_df = df[df['sex'].isin(sex_filter) & df['region'].isin(region_filter) &
                 df['children'].isin(children_filter) & df['smoker'].isin(smoker_filter)]

# Display filtered data
st.write(f"Displaying data for: Sex={sex_filter}, Region={region_filter}, "
         f"Children={children_filter}, Smoker={smoker_filter}")

st.dataframe(filtered_df, use_container_width=True)

######################################################################################################

# Charts and Graphs
st.subheader('Charts and Graphs')

# Create a 2x2 subplot grid
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(24, 28))

#################################################################################################################

# Age Distribution
sns.histplot(filtered_df['age'], bins=20, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution', fontsize=16)

# Line plot of Age vs Charges with Smoker Influence
sns.lineplot(x='age', y='charges', hue='smoker', data=filtered_df, ax=axes[0, 1], palette='Set1')
axes[0, 1].set_title('Age vs Charges with Smoker Influence', fontsize=16)

# Calculate the correlation matrix
numeric_df = filtered_df.select_dtypes(include='number')

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=axes[0, 2])
axes[0, 2].set_title('Correlation Heatmap', fontsize=16)

########################################################################################################################

# Proportion chart of Sex
sex_distribution = filtered_df['sex'].value_counts()
ax_sex_distribution = axes[1, 0]
ax_sex_distribution.pie(sex_distribution, labels=sex_distribution.index, autopct='%1.1f%%', startangle=90)
ax_sex_distribution.set_title('Proportion of Gender', fontsize=16)

# Influence of Sex on Charges
sns.barplot(x='sex', y='charges', hue='smoker', data=filtered_df, ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title('Influence of Sex on Charges', fontsize=16)

# Proportion chart of Smoker as a donut chart
smoker_distribution = filtered_df['smoker'].value_counts()
ax_smoker_distribution = axes[1, 2]

# Create a pie chart
ax_smoker_distribution.pie(smoker_distribution, labels=smoker_distribution.index,
                            autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))

# Draw a white circle in the center to create the donut appearance
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax_smoker_distribution.add_artist(centre_circle)
ax_smoker_distribution.axis('equal')
ax_smoker_distribution.set_title('Proportion of Smoker-Non Smoker', fontsize=16)

##########################################################################################################################

# Bar plot for comparing number of sample from different Region
sns.countplot(x='region', data=filtered_df, ax=axes[2, 0], palette='viridis')
axes[2, 0].set_title('Count of Regions', fontsize=16)

# Influence of Region on Charges wrt smoker/non_smoker
sns.barplot(x='region', y='charges', hue='smoker', data=df, ax=axes[2, 1], palette='Reds_r')
axes[2, 1].set_title(' Influence of Region on Charges wrt smoker/non_smoker', fontsize=16)

# Bar plot for the count of Children
sns.countplot(x='children', data=filtered_df, ax=axes[2, 2], palette='viridis')
axes[2, 2].set_title('Count of Children', fontsize=16)


#########################################################################################################################


# BMI Distribution
sns.histplot(filtered_df['bmi'], bins=20, kde=True, ax=axes[3, 0])
axes[3, 0].set_title('BMI Distribution', fontsize=16)

# Scatter plot of BMI vs Charges with Smoker Influence
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=filtered_df, ax=axes[3, 1], palette='Set1')
axes[3, 1].set_title('BMI vs Charges with Smoker Influence', fontsize=16)

# Violin plot of Children_count vs Charges with Smoker Influence
sns.violinplot(x='children', y='charges', hue='smoker', data=filtered_df, orient='v',ax=axes[3, 2])
axes[3, 2].set_title('Children_count vs Charges with Smoker Influence', fontsize=16)

############################################################################################################################


# Adjust layout
plt.tight_layout()

# Display the entire subplot
st.pyplot(fig)


###########################################################################################################################

# Load the best trained model
model = pickle.load(open('insurance_charges_model.p', 'rb'))

# Machine Learning Deployement
st.subheader("Predicting charges:")


# Collect user inputs
age = st.number_input('Age', min_value=1, max_value=100, value=25)
st.write("Choose 0 for Male and 1 for Female")
sex = st.number_input('Sex', min_value=0, max_value=1)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=10.0, step=0.001)  # Allow decimal values
children = st.number_input('Children', min_value=0, max_value=5)
st.write("Choose 0 for Non-smoker and 1 for Smoker")
smoker = st.number_input('Smoker', min_value=0, max_value=1)
region_northeast = st.number_input('Region Northeast', min_value=0, max_value=1)
region_northwest = st.number_input('Region Northwest', min_value=0, max_value=1)
region_southeast = st.number_input('Region Southeast', min_value=0, max_value=1)
region_southwest = st.number_input('Region Southwest', min_value=0, max_value=1)

output = ""

if st.button("Predict"):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region_northeast, region_northwest, region_southeast, region_southwest]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'])

    # Make prediction
    result = model.predict(input_data)
    st.success('The output of the above is {}'.format(result))