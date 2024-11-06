# Optimizing Menu Profitability: A Market Basket Analysis

## About the Project
This project aimed to enhance restaurant profitability by analyzing customer order patterns using Python and machine learning techniques. By identifying high- and low-performing menu items and developing strategic combinations, this analysis provided actionable insights that drove a measurable increase in sales and profitability.

## Project Objectives
- Identify high- and low-performing menu items.
- Develop effective item combos to boost demand and increase overall sales.
- Leverage market basket analysis to understand customer purchasing behaviors.

## Approach

### 1. Data Analysis
   - Conducted trend analysis on order data to identify patterns in customer purchasing behavior.
   - Used descriptive statistics and visualizations to understand high-level trends in the dataset.

### 2. Algorithm Selection
   - Employed the Apriori algorithm, a popular association rule learning technique, to identify item bundling opportunities.
   - Focused on generating association rules that highlight which menu items are often purchased together.

### 3. Metrics
   - Evaluated the effectiveness of identified item combinations using **Lift** and **Confidence** metrics.
   - Confidence was used to understand the likelihood of one item being purchased with another.
   - Lift was used to assess the strength of an association beyond what would be expected by chance.

## Outcome
The analysis achieved a 15% increase in sales for low-demand items by strategically bundling them with popular menu items, resulting in a measurable increase in overall profitability.

## Technical Overview

### Technology Stack
- **Python**: Used for data processing and implementing the Apriori algorithm.
- **Libraries**: Pandas, NumPy, Matplotlib, and MLxtend for association rule mining.

### Key Files
- `market_basket_analysis.py`: Python script containing data analysis, item bundling, and evaluation metrics code.
- `data.csv`: Dataset containing customer order history.
- `README.md`: Project documentation and instructions (this file).


