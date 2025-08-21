
# Retail Forecasting
<!--
<img width="847" height="314" alt="sales202CA4" src="https://github.com/user-attachments/assets/99372e2e-a293-4817-9caa-119c3ea44344" />
-->
## Overview

This project develops state-of-the-art machine learning models (LightGBM) to predict sales for the next 8 days at a shop-product level, using 3 years of historical data from a large retailer. The system supports large-scale modeling, training multiple models in parallel.


## Challenges faced

<img align="right" width="300" height="200" src="https://github.com/user-attachments/assets/733099ad-1eba-4a19-8223-7d5be7952ff1">

- **Hierarchical data structure**:
we modeled sales at the **store–product level**, allowing the forecasts to reflect individual product demand in each store.  

- **Intermittent demand**: we created a variable to capture this effect and help making predictions about it.


- **Massive modeling**: to deal with having to model for multiple elements, we used the fast and light algorithm LightGBM.

- Forecasting for **several days in the future**: we use a recursive prediction. 
The model makes a prediction for the next day and then uses that prediction to estimate the following day. 




## Results


<i>For context, the average daily sales per product is around 20 units (25th-75th percentile: 6-37), but it varies greatly and depends heavily on the product.
</i>

- 1-day forecast average absolute error of less than 6 units sold.
- 8-day forecast average absolute error of 8.5 units sold.

<i> These metrics are just very rough estimates, but it is safe to say that it provides an adequate prediction of overall demand trends.
</i>

<br>

<img width="847" height="314" alt="sales252CA3" src="https://github.com/user-attachments/assets/c6f43efe-09dd-4257-899d-6dfb2905f930" />

<img width="847" height="314" alt="sales120CA4" src="https://github.com/user-attachments/assets/a1c77e31-5a81-473f-9ebd-f1585a36b64c" />

<img width="847" height="314" alt="sales714CA4" src="https://github.com/user-attachments/assets/ead5686e-655a-44bd-b6c3-15cfeaf73e74" />


<br>

## Notebooks & Scripts

- [Development notebooks](https://github.com/adrianriverof/forecasting-retail/tree/master/03_Notebooks/02_Development)  
- [Retraining script](https://github.com/adrianriverof/forecasting-retail/blob/master/03_Notebooks/03_System/08_Retraining_code.py)
- [Execution script](https://github.com/adrianriverof/forecasting-retail/blob/master/03_Notebooks/03_System/09_Execution_code.py)




