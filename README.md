# Overview
Predicting stock market prices is a challenging task due to the complex and dynamic nature of financial markets. Various machine learning and statistical approaches have been used to tackle this problem, with Genetic Algorithms (GAs) emerging as a powerful technique due to their ability to optimize complex, multi-dimensional spaces effectively.

In this project, we will go through data input modifications for the model to be trained on. Setting up the ANN model to predict and calculate the accuracy of the prediction. Next, we focus on applying Genetic Algorithms to enhance the accuracy in the prediction of the trend of the stock market by minimizing the input features. Inspired by natural selection, GAs use mechanisms such as selection, crossover, and mutation to iteratively evolve a population of candidate solutions. Then we compare which input features will yield the highest ANN accuracy and in turn show which features should the investors be concerned about when it comes to making profits in the stock market.

## How to run this project


## Input Data
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
Accuracy was employed to guide chromosome selection (subsets of input variables) for generating the next generation in GA, as well as to evaluate the prediction model’s performance. The fitness values in GA were defined as the accuracy values, which are calculated using the following formula:

<p align="center">
Accuracy = (TP + TN) / (TP + TN + FP + FN)
</p>

Where TP represents true positives, FP is false positives, TN is true negatives, and FN is false negatives.

#### Step 5: Stopping criterion
The stopping criterion was based on whether the accuracy of the best individual of the current generation was higher than the accuracy of the previous generation. If this continues for 20 consecutive generations (the stall value is 20), endsthe program.

#### Step 6: Selection mechanism
Tournament Selection Process

The selection mechanism in GA ensures that the population of solution candidates consistently improves in terms of overall fitness values. This mechanism allows GA to eliminate suboptimal solutions while retaining the best individuals. Among the various selection techniques available, Tournament Selection with a size of 3 was adopted in this study for its simplicity, speed, and efficiency. Tournament selection applies higher selection pressure to GA, accelerating convergence and ensuring that the worst solutions do not progress to the next generation. 


#### Step 7: Crossover function
Single-point crossover

The crossover operator in the GA combines two parent individuals (chromosomes) to create offspring for the next generation. To perform the crossover operation, two parent chromosomes are selected through the tournament selection process. We implemented the
single-point method for the crossover section. This method helps to explore new regions of the search space by exchanging parts of the chromosomes between the parents. The crossover function works by selecting a random crossover point along the chromosome and swapping the genetic material (bits) after this point between the two parents, resulting in two offspring.

#### Step 8: Mutation function
Mutation in a genetic algorithm (GA) refers to a genetic perturbation that alters individuals in the population. It plays a crucial role in maintaining genetic diversity and exploring a broader solution space. In this work, we applied uniform mutation as our mutation strategy. The uniform mutation operator randomly selects genes (bits) in a chromosome and flips their values, which introduces variability and prevents the algorithm from getting stuck in local optima by exploring different areas of the solution space.

#### Step 9: Replacement (New population)
The genetic algorithm continues evolving until the new population is fully populated. The new population is formed by adding individuals from Elite kids (There are 2 elite children), Crossover kids, and Mutation kids:

<p align="center">
New Population = Elite Kids + Crossover Kids + Mutation Kids
</p>

Once the new population is formed, it is evaluated, and the selection and reproduction processes are repeated until the stopping condition is satisfied.

#### Step 10: Repeat until occurring the stopping conditions
There are two stopping conditions applicable to this work that are: stopping by hitting the maximum number of generations or when reaching the stall value. The stall value (as mentioned in step 5) and the generation number are 20 and 50 respectively.

## Results
<div align="center">
  
| Pop_size | ANN     | GA + ANN | Best individual                                                                 |
|----------|---------|----------|---------------------------------------------------------------------------------|
| 40       | 52.07%  | 67.21%   | [ 2  3  5  6  8 10 11 14 15 16 18 19 21 22 23 24 27 36 37 38 39 40]           |
| 50       | 52.07%  | 68.16%   | [ 0  3  4  5  6  8 11 12 18 20 21 23 24 28 29 34 35 37 38 39 41 42 43]       |
| 60       | 52.07%  | 68.73%   | [ 1  4  8  9 10 11 16 18 19 20 21 22 25 34 37 40 43]                         |

</div>

<p align="center">
Table 2. Results for different Pop size values.
</p>

<div align="center">

| Epoch | ANN     | GA + ANN | Best individual                                                                 |
|-------|---------|----------|---------------------------------------------------------------------------------|
| 500   | 52.07%  | 67.21%   | [ 0  1  3  4  5  8  9 11 12 13 14 16 18 19 20 21 22 24 28 29 32 33 34 36 37 38 39 40 42] |
| 1000  | 53.88%  | 68.55%   | [ 0  1  2  3  4  5  8  9 11 12 14 18 19 21 23 25 26 27 30 35 37 41 42 43]       |
| 2000  | 51.34%  | 67.14%   | [ 0  1  2  3  4  5  8  9 11 12 14 19 21 23 25 26 27 30 35 37 40 42 43]       |

</div>

<p align="center">
Table 3. Results for different Epoch values.
</p>

<div align="center">

| Mutation Rate | ANN     | GA + ANN | Best individual                                                                 |
|---------------|---------|----------|---------------------------------------------------------------------------------|
| 0.1           | 52.07%  | 67.21%   | [ 0  1  2  3  4  5  6  8  9 10 11 12 14 18 20 23 25 26 28 33 34 35 36 37 38 42] |
| 0.2           | 52.07%  | 67.21%   | [ 0  1  2  3  4  5  6  8  9 10 11 12 14 18 20 23 25 26 28 33 34 35 36 37 38 42] |
| 0.3           | 52.07%  | 69.53%   | [ 0  1  3  5  9 11 16 17 19 20 21 22 23 25 27 29 30 34 35 37 42 43]             |
| 0.4           | 52.07%  | 68.46%   | [ 0  2  7  8  9 10 11 14 15 19 21 22 25 29 32 34 39 40 41]                     |

</div>

<p align="center">
Table 4. Results for different Mutation Rate values.
</p>

# Conclusion
In our project, we developed a hybrid model combining **Artificial Neural Networks (ANN) and Genetic Algorithms (GA)** to predict stock index movements, testing it on a large dataset of historical stock trading data. The goal was to achieve better prediction accuracy compared to a standalone ANN model. The test results demonstrated that the hybrid model successfully met this objective, with an average improvement of 15.14\%, resulting in a prediction accuracy of 67.94\%. However, the accuracy remains relatively modest, and we are exploring the integration of ANN with other machine learning models to further enhance prediction performance.
