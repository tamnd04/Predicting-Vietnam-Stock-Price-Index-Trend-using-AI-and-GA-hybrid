# Overview
Predicting stock market prices is a challenging task due to the complex and dynamic nature of financial markets. Various machine learning and statistical approaches have been used to tackle this problem, with Genetic Algorithms (GAs) emerging as a powerful technique due to their ability to optimize complex, multi-dimensional spaces effectively.

In this project, we will go through data input modifications for the model to be trained on. Setting up the ANN model to predict and calculate the accuracy of the prediction. Next, we focus on applying Genetic Algorithms to enhance the accuracy in the prediction of the trend of the stock market by minimizing the input features. Inspired by natural selection, GAs use mechanisms such as selection, crossover, and mutation to iteratively evolve a population of candidate solutions. Then we compare which input features will yield the highest ANN accuracy and in turn show which features should the investors be concerned about when it comes to making profits in the stock market.

## Requirements
The input data for the VN30 index must consist of daily records spanning from January 1, 2019, to November 10, 2024 (which is 2140 days). Eleven technical indicators are calculated, each represented by four input variables corresponding to different historical time spans—3-day, 5-day, 10-day, and 15-day periods leading up to the prediction date. These inputs are designed to create diverse subsets of data, which are subsequently refined by a Genetic Algorithm (GA) to select the most effective features. The optimized input set is then fed into an Artificial Neural Network (ANN) to forecast the VN30 index trend.

## Methodology
<p align="center">
![image](https://github.com/user-attachments/assets/86f765d7-9d82-4940-bedb-d57c7c4a1428)
</p>


### Data preparation and Preprocessing
This study utilized a dataset comprising daily closing values of a single company named ACB in the VN30 index recorded between January 1, 2019, and November 10, 2024, spanning 2,140 days. After calculating the 11 technical indicators (Figure 1) with 4 different periods and cleaning the dataset there are only 1409 days left. Within this timeframe, the stock index increased on 666 occasions (47.3\%) and decreased on 743 occasions (52.7\%), as detailed in Table 1.

<p align="center">
![image](https://github.com/user-attachments/assets/e0544010-2ed9-4a06-ad0c-a3bf4fa901f9)
</p>
