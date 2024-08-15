# BCG Customer Churn Visualization Project

## Overview

The BCG Customer Churn Visualization Project aims to analyze customer churn data and create interactive visualizations to gain insights into the factors contributing to customer attrition. By leveraging data visualization techniques, this project helps businesses identify patterns, trends, and relationships within the customer churn data, enabling them to make data-driven decisions to reduce churn and improve customer retention.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Insights and Recommendations](#insights-and-recommendations)
- [Conclusion](#conclusion)
- [License](#license)

## Project Description

The objective of this project is to create a comprehensive set of interactive visualizations that provide valuable insights into customer churn for a telecommunications company. By analyzing various attributes related to customer demographics, usage patterns, and service interactions, the project aims to identify the key factors that influence customer churn.

### Key Steps in the Project:

1. **Data Preprocessing**: Cleaning and preparing the dataset for analysis.
2. **Exploratory Data Analysis (EDA)**: Understanding the dataset through visualizations and statistical analysis.
3. **Feature Engineering**: Creating new features or transforming existing ones to enhance the visualization's effectiveness.
4. **Visualization Development**: Building interactive visualizations using Plotly and Dash.
5. **Insights Generation**: Analyzing the visualizations to derive insights and identify patterns related to customer churn.
6. **Recommendations**: Providing actionable recommendations based on the insights gained from the visualizations.

## Dataset

The dataset used in this project is based on the BCG Telco Customer Churn dataset, which contains information about customer demographics, usage patterns, and service interactions. The key features include:

### client_data.csv
● id = client company identifier

● activity_new = category of the company’s activity

● channel_sales = code of the sales channel

● cons_12m = electricity consumption of the past 12 months

● cons_gas_12m = gas consumption of the past 12 months

● cons_last_month = electricity consumption of the last month

● date_activ = date of activation of the contract

● date_end = registered date of the end of the contract

● date_modif_prod = date of the last modification of the product

● date_renewal = date of the next contract renewal

● forecast_cons_12m = forecasted electricity consumption for the next 12 months

● forecast_cons_year = forecasted electricity consumption for the next calendar year

● forecast_discount_energy = forecasted value of current discount

● forecast_meter_rent_12m = forecasted bill of meter rental for the next 2 months

● forecast_price_energy_off_peak = forecasted energy price for 1st period (off-peak)

● forecast_price_energy_peak = forecasted energy price for 2nd period (peak)

● forecast_price_pow_off_peak = forecasted power price for 1st period (off-peak)

● has_gas = indicated if the client is also a gas client

● imp_cons = current paid consumption

● margin_gross_pow_ele = gross margin on power subscription

● margin_net_pow_ele = net margin on power subscription

● nb_prod_act = number of active products and services

● net_margin = total net margin

● num_years_antig = antiquity of the client (in number of years)

● origin_up = code of the electricity campaign the customer first subscribed to

● pow_max = subscribed power

● churn = has the client churned over the next 3 months


### price_data.csv

● id = client company identifier

● price_date = reference date

● price_off_peak_var = price of energy for the 1st period (off-peak)

● price_peak_var = price of energy for the 2nd period (peak)

● price_mid_peak_var = price of energy for the 3rd period (mid peak)

● price_off_peak_fix = price of power for the 1st period (off-peak)

● price_peak_fix = price of power for the 2nd period (peak)

● price_mid_peak_fix = price of power for the 3rd period (mid peak)

Note: some fields are hashed text strings. This preserves the privacy of the original data but the 
commercial meaning is retained and so they may have predictive power

## Technologies Used

- **Python**: Programming language used for data analysis and visualization development.
- **Pandas**: Library for data manipulation and analysis.
- **NumPy**: Library for numerical computations.
- **Dash**: Framework for building interactive web applications based on Plotly.
- **SweetViz** : Library for data visualizations.

## Installation

To run this project locally, you will need to have Python installed along with the required libraries. You can install the necessary packages using pip:

```bash
pip install pandas numpy dash sweetviz
```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/CollinsNyatundo/BCG-customer-churn-visualization.git
   cd BCG-customer-churn-visualization
   ```

2. **Run the Jupyter Notebook**:
   Launch Jupyter Notebook and open the `BCG-customer-churn-visualization.ipynb` file.

3. **Follow the Notebook Cells**:
   Execute each cell sequentially to preprocess the data, perform EDA, engineer features, and develop the visualizations using Plotly and Dash.

## Visualizations

The project includes a set of interactive visualizations that provide insights into customer churn:

- **Churn Rate by Customer Attributes**: Visualizes churn rates across different customer demographics and service attributes.
- **Churn Rate by Tenure**: Analyzes the relationship between customer tenure and churn.
- **Churn Rate by Monthly Charges**: Investigate the impact of monthly charges on customer churn.
- **Churn Rate by Contract Type**: Compares churn rates across different contract types.
- **Churn Rate by Payment Method**: Examines the influence of payment method on customer churn.

## Insights and Recommendations

The interactive visualizations generated by this project provide valuable insights into customer churn, such as:

- **Customers with longer tenures have lower churn rates**, suggesting the importance of retaining customers over time.
- **Customers with higher monthly charges are more likely to churn**, indicating the need for competitive pricing strategies.
- **Customers on month-to-month contracts have higher churn rates**, emphasizing the value of encouraging long-term commitments.

Based on these insights, the project provides actionable recommendations to reduce customer churn, such as:

- **Offering incentives for long-term contracts**
- **Implementing targeted retention strategies for customers with higher monthly charges**
- **Enhancing customer service and support to improve satisfaction and loyalty**

## Conclusion

The BCG Customer Churn Visualization Project demonstrates how interactive data visualizations can provide valuable insights into customer churn. By leveraging the power of Plotly and Dash, this project enables businesses to explore customer churn data, identify key drivers, and make data-driven decisions to improve customer retention and reduce attrition.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
