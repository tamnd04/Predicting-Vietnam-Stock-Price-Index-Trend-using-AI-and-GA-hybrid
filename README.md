# Overview
Predicting stock market prices is a challenging task due to the complex and dynamic nature of financial markets. Various machine learning and statistical approaches have been used to tackle this problem, with Genetic Algorithms (GAs) emerging as a powerful technique due to their ability to optimize complex, multi-dimensional spaces effectively.

In this project, we will go through data input modifications for the model to be trained on. Setting up the ANN model to predict and calculate the accuracy of the prediction. Next, we focus on applying Genetic Algorithms to enhance the accuracy in the prediction of the trend of the stock market by minimizing the input features. Inspired by natural selection, GAs use mechanisms such as selection, crossover, and mutation to iteratively evolve a population of candidate solutions. Then we compare which input features will yield the highest ANN accuracy and in turn show which features should the investors be concerned about when it comes to making profits in the stock market.

## Requirements
The input data for the VN30 index must consist of daily records spanning from January 1, 2019, to November 10, 2024 (which is 2140 days). Eleven technical indicators are calculated, each represented by four input variables corresponding to different historical time spans—3-day, 5-day, 10-day, and 15-day periods leading up to the prediction date. These inputs are designed to create diverse subsets of data, which are subsequently refined by a Genetic Algorithm (GA) to select the most effective features. The optimized input set is then fed into an Artificial Neural Network (ANN) to forecast the VN30 index trend.

## Methodology

![image](https://github.com/user-attachments/assets/86f765d7-9d82-4940-bedb-d57c7c4a1428)

<p align="center">
Figure 1: Methodology workflow for the project.
</p>

### Data preparation and Preprocessing
This study utilized a dataset comprising daily closing values of a single company named ACB in the VN30 index recorded between January 1, 2019, and November 10, 2024, spanning 2,140 days. After calculating the 11 technical indicators (Figure 2) with 4 different periods and cleaning the dataset there are only 1409 days left. Within this timeframe, the stock index increased on 666 occasions (47.3\%) and decreased on 743 occasions (52.7\%), as detailed in Table 1.

![image](https://github.com/user-attachments/assets/e0544010-2ed9-4a06-ad0c-a3bf4fa901f9)

<p align="center">
Figure 2. Technical indicators used in this study and their equations.
</p>

<div align="center">

| Year | Up (times) | Up (%) | Down (times) | Down (%) | Total |
|------|------------|--------|--------------|----------|-------|
| 2019 | 90         | 41.3%  | 128          | 58.7%    | 218   |
| 2020 | 120        | 49.4%  | 123          | 50.6%    | 243   |
| 2021 | 126        | 50.4%  | 124          | 49.6%    | 250   |
| 2022 | 119        | 48.0%  | 129          | 52.0%    | 248   |
| 2023 | 114        | 47.1%  | 128          | 52.9%    | 242   |
| 2024 | 97         | 46.6%  | 111          | 53.4%    | 208   |
| **Total** | **666**    | **47.3%**  | **743**      | **52.7%**  | **1409** |

</div>

<p align="center">
Table 1. Yearly Up and Down distributions with percentages and totals.
</p>

### Building the hybrid model (ANN + GA)
![image](https://github.com/user-attachments/assets/e88779e3-ba0e-403e-bbff-5ee22494847e)

<p align="center">
Figure 3. Steps of running the ANN + GA hybrid algorithm.
</p>

#### Step 1: Population initialization
The initial population in this study was represented as a matrix with dimensions of Population Size × Chromosome Length, consisting solely of randomly generated binary digits. Here, Population Size refers to the number of chromosomes (or individuals) in the population, while Chromosome Length (or Genome Length) denotes the number of bits (or genes) in each chromosome. To ensure enough coverage of the search space, it is generally advisable to set the population size to be at least equal to the chromosome length. In this work, the Chromosome Length was set to 44, and the Population Size was chosen as 50.

#### Step 2: Decode (Feature selection)
Decode the chromosomes (bit strings) to determine which input variables are inserted into the ANN. For example, if the selected features are:
<p align="center">
[2  3  5  6  8 10 11 14 15 16 18 19 21 22 23 24 27 36 37 38 39 40] 
</p>
Then these 12 columns will be the input variables for the ANN.

#### Step 3: Artificial Neural Network (ANN)
Run a three-layered feedforward ANN model to predict the next-day VN30 index of the ACB company.

#### Step 4: Fitness evaluation
Accuracy was employed to guide chromosome selection (subsets of input variables) for
generating the next generation in GA, as well as to evaluate the prediction model’s performance.
The fitness values in GA were defined as the accuracy values, which are calculated using the
following formula:
$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$
Where TP represents true positives, FP is false positives, TN is true negatives, and FN is false
negatives.

#### Step 5
#### Step 6
#### Step 7
#### Step 8
#### Step 9
#### Step 10


