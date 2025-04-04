import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class ANN:
    def __init__(self, input_size, hidden_size=100, output_size=1, learning_rate=0.1, momentum=0.1, l2_lambda=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_lambda = l2_lambda  # L2 regularization parameter

        self.weights1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))

        self.momentum1 = np.zeros_like(self.weights1)
        self.momentum2 = np.zeros_like(self.weights2)

        # Calculate class weights based on the dataset distribution
        # Class 0: 743 samples, Class 1: 666 samples, Total: 1409
        total_samples = 1409
        self.class_weights = {
            0: total_samples/(2 * 743),  # ≈ 0.948
            1: total_samples/(2 * 666),  # ≈ 1.058
        }

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights1)
        self.hidden_output = self.tanh(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights2)
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        # Apply class weights to the error
        weights = np.array([self.class_weights[int(label)] for label in y])
        self.output_error = (y.reshape(-1, 1) - output) * weights.reshape(-1, 1)
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        self.hidden_error = np.dot(self.output_delta, self.weights2.T)
        self.hidden_delta = self.hidden_error * self.tanh_derivative(self.hidden_output)

        # Calculate weight updates with L2 regularization
        weights2_update = np.dot(self.hidden_output.T, self.output_delta) - self.l2_lambda * self.weights2
        weights1_update = np.dot(X.T, self.hidden_delta) - self.l2_lambda * self.weights1

        # Apply momentum
        self.momentum2 = self.momentum * self.momentum2 + self.learning_rate * weights2_update
        self.momentum1 = self.momentum * self.momentum1 + self.learning_rate * weights1_update

        # Update weights
        self.weights2 += self.momentum2
        self.weights1 += self.momentum1

    def train(self, X, y, epochs=2000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


def create_year_folds(years, trends, n_splits=5):
    unique_years = np.unique(years)

    year_trend_indices = {}
    for year in unique_years:
        year_mask = years == year
        year_indices = np.where(year_mask)[0]

        up_indices = year_indices[trends[year_indices] == 1]
        down_indices = year_indices[trends[year_indices] == 0]

        year_trend_indices[year] = {
            'up': up_indices,
            'down': down_indices,
            'up_count': len(up_indices),
            'down_count': len(down_indices)
        }

        total = len(year_indices)
        up_percent = (len(up_indices) / total) * 100
        down_percent = (len(down_indices) / total) * 100
        print(f"\nYear {year}:")
        print(f"Total samples: {total}")
        print(f"Up movements: {len(up_indices)} ({up_percent:.1f}%)")
        print(f"Down movements: {len(down_indices)} ({down_percent:.1f}%)")

    folds = [[] for _ in range(n_splits)]

    for year in unique_years:
        for trend_type in ['up', 'down']:
            indices = year_trend_indices[year][trend_type]
            np.random.shuffle(indices)  # Randomize indices
            n_samples = len(indices)
            samples_per_fold = n_samples // n_splits

            for i in range(n_splits):
                start_idx = i * samples_per_fold
                end_idx = start_idx + samples_per_fold if i < n_splits - 1 else n_samples
                fold_indices = indices[start_idx:end_idx]
                folds[i].extend(fold_indices)

    folds = [np.array(fold) for fold in folds]

    splits = []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        splits.append((train_idx, test_idx))

        print(f"\nFold {i + 1} distribution:")
        for year in unique_years:
            year_mask = years[test_idx] == year
            year_test_idx = test_idx[year_mask]
            total = len(year_test_idx)
            if total > 0:
                up_count = np.sum(trends[year_test_idx] == 1)
                down_count = np.sum(trends[year_test_idx] == 0)
                up_percent = (up_count / total) * 100
                down_percent = (down_count / total) * 100
                print(f"Year {year}: Up={up_count}({up_percent:.1f}%), Down={down_count}({down_percent:.1f}%)")

    return splits


def evaluate_baseline_ann(X, y, splits):
    """
    Evaluate ANN performance using all features with year-based 5 folds cross-validation
    """
    accuracies = []

    print("\nBaseline ANN Evaluation (All Features):")
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create and train ANN
        ann = ANN(input_size=X.shape[1])
        ann.train(X_train, y_train)

        # Evaluate
        predictions = ann.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")

    mean_accuracy = np.mean(accuracies)
    print(f"Average Baseline Accuracy: {mean_accuracy:.4f}")
    return mean_accuracy


def evaluate_features(chromosome, X, y, splits):
    """
    Evaluate a feature selection chromosome using year-based 5 folds cross-validation
    """
    selected_features = X[:, np.array(chromosome, dtype=bool)]

    if np.sum(chromosome) == 0:  # If no features selected
        return 0

    accuracies = []

    for train_idx, test_idx in splits:
        X_train, X_test = selected_features[train_idx], selected_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ann = ANN(input_size=int(np.sum(chromosome)))
        ann.train(X_train, y_train)

        predictions = ann.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        accuracies.append(acc)

    return np.mean(accuracies)


def initialize_population(pop_size, n_features):
    """Initialize random population for GA"""
    return np.random.randint(2, size=(pop_size, n_features))


def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select parent using tournament selection"""
    tournament_idx = np.random.choice(len(population), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_idx]
    return population[tournament_idx[np.argmax(tournament_fitness)]]


def crossover(parent1, parent2):
    """Single point crossover"""
    point = np.random.randint(len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2


def mutation(chromosome, mutation_rate=0.1):
    """Bit flip mutation"""
    mask = np.random.random(len(chromosome)) < mutation_rate
    chromosome[mask] = 1 - chromosome[mask]
    return chromosome


def genetic_algorithm(X, y, baseline_accuracy, splits, pop_size=50, generations=50):
    n_features = X.shape[1]
    population = initialize_population(pop_size, n_features)
    best_solution = None
    best_fitness = baseline_accuracy
    stall = 20  # Generations without improvement before stopping
    counter = 0

    print("\nStarting GA Feature Selection:")
    print('\nInitial Setup:')
    print(f'Population Size: {pop_size}')
    print(f'Number of Generations: {generations}')
    print(f'Mutation Rate: 0.1')
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

    for gen in range(generations):
        fitness_scores = [evaluate_features(chrom, X, y, splits) for chrom in population]

        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_indices = sorted_indices[:2]
        elite_chromosomes = population[elite_indices].copy()
        if fitness_scores[sorted_indices[0]] > best_fitness:
            best_fitness = fitness_scores[sorted_indices[0]]
            best_solution = population[sorted_indices[0]]
            print(f"\nGeneration {gen + 1}: New Best Fitness = {best_fitness:.4f}")
            print("Selected Features:", np.where(best_solution == 1)[0])
            print(f"Improvement over baseline: {(best_fitness - baseline_accuracy) * 100:.2f}%")
            counter = 0
        else:
            print(f"\nGeneration {gen + 1}: Best Fitness = {best_fitness:.4f}")
            counter += 1
            if best_solution is not None:
                print("Selected Features:", np.where(best_solution == 1)[0])
            else:
                print("All features selected")
                counter += 1
        if counter == stall:
            print(f'\nStopping early: No improvement for {stall} generations')
            break

        new_population = []
        new_population.extend(elite_chromosomes)
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutation(child1)
            child2 = mutation(child2)

            new_population.extend([child1, child2])

        population = np.array(new_population[:pop_size])
    return best_solution, best_fitness


def main():
    print("Loading and preprocessing data...")
    data = pd.read_csv('dataset.csv')
    X = data.iloc[:, :44].values  # First 44 columns of the dataset
    print(data.iloc[:,:44])
    y = data['Trend'].values
    years = data['Year'].values
    splits = create_year_folds(years, y) # Split the dataset for 5 folds cross-validation

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    baseline_accuracy = evaluate_baseline_ann(X_scaled, y, splits)
    best_chromosome, best_fitness = genetic_algorithm(X_scaled, y, baseline_accuracy, splits)

    selected_features = np.where(best_chromosome == 1)[0]
    feature_names = data.columns[:44]

    print("\nFinal Results:")
    print(f"Baseline Accuracy (All Features): {baseline_accuracy:.4f}")
    print(f"Best Accuracy (Selected Features): {best_fitness:.4f}")
    print(f"Improvement: {(best_fitness - baseline_accuracy) * 100:.2f}%")
    print(f"Number of Selected Features: {len(selected_features)} out of 44")
    print("\nSelected Features:")
    for idx in selected_features:
        print(f"- {feature_names[idx]}")


if __name__ == "__main__":
    main()
