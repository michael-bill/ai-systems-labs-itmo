import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Загрузка и предварительная обработка данных
data = pd.read_csv('./data/train.csv')

# Выбираем необходимые признаки и обрабатываем данные
data['Age'] = data['Age'].fillna(data['Age'].mean())
data = data[['Survived', 'Pclass', 'Sex', 'Age']]
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

print(data.head())

# Визуализация и статистика данных
for feature in data.columns:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(data[feature], bins=10, edgecolor='black', alpha=0.7)
    ax.set_title(f'Распределение признака {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Частота')

    mean = data[feature].mean()
    std = data[feature].std()
    median = data[feature].median()
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)

    ax.axvline(mean, color='r', linestyle='--', label=f'Среднее: {mean:.2f}')
    ax.axvline(median, color='g', linestyle='-', label=f'Медиана: {median:.2f}')
    ax.axvline(q1, color='b', linestyle=':', label=f'25% квантиль: {q1:.2f}')
    ax.axvline(q3, color='b', linestyle=':', label=f'75% квантиль: {q3:.2f}')
    ax.axvspan(mean - std, mean + std, color='green', alpha=0.2, label=f'Стд. отклонение: {std:.2f}')

    ax.legend()
    plt.show()

# Подготовка обучающего и тестового наборов данных
train_data = pd.read_csv('./data/train.csv')
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
train_data = train_data[['Survived', 'Pclass', 'Sex', 'Age']]
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

test_data = pd.read_csv('./data/test.csv')
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data = test_data[['Pclass', 'Sex', 'Age']]
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

X_train = train_data.drop('Survived', axis=1).values
y_train = train_data['Survived'].values
X_test = test_data.values

print("Размер обучающего набора:", X_train.shape)
print("Размер тестового набора:", X_test.shape)

# Нормализация данных
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Определение функций для логистической регрессии
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, y_pred):
    m = len(y)
    epsilon = 1e-5  # для предотвращения деления на 0
    return -(1 / m) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

def predict_proba(X, weights):
    return sigmoid(np.dot(X, weights))

def gradient_descent(X, y, weights, learning_rate):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    error = predictions - y
    gradient = np.dot(X.T, error) / m
    weights -= learning_rate * gradient
    return weights

def newton_method(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    gradient = np.dot(X.T, predictions - y)
    R = np.diag(predictions * (1 - predictions))
    H = np.dot(np.dot(X.T, R), X)
    weights -= np.linalg.solve(H, gradient)
    return weights

def fit(X, y, method='newton', num_iterations=1000, learning_rate=0.01):
    X = np.insert(X, 0, 1, axis=1)  # Добавляем столбец единиц для свободного члена (bias)
    n_features = X.shape[1]
    weights = np.zeros(n_features)

    for i in range(num_iterations):
        if method == 'newton':
            weights = newton_method(X, y, weights)
        elif method == 'gradient':
            weights = gradient_descent(X, y, weights, learning_rate)
        else:
            raise ValueError("Invalid method. Use 'newton' or 'gradient'.")

        # Отслеживаем функцию потерь каждые 100 итераций
        if (i + 1) % 100 == 0:
            y_pred = predict_proba(X, weights)
            loss = log_loss(y, y_pred)
            print(f'Iteration {i + 1}, Loss: {loss}')
    return weights

def predict(X, weights, threshold=0.5):
    X = np.insert(X, 0, 1, axis=1)
    probabilities = predict_proba(X, weights)
    return (probabilities >= threshold).astype(int)

def print_metrics(y_true, y_pred):
    accuracy_param = accuracy_score(y_true, y_pred)
    precision_param = precision_score(y_true, y_pred)
    recall_param = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy_param:.4f}')
    print(f'Precision: {precision_param:.4f}')
    print(f'Recall: {recall_param:.4f}')
    print(f'F1-Score: {f1:.4f}')

# Загрузка истинных значений для тестового набора
y_test = pd.read_csv('./data/gender_submission.csv')['Survived'].values

# Исследование гиперпараметров
learning_rates = [0.1, 0.01, 0.001]  # Различные значения скорости обучения
num_iterations_list = [100, 500, 1000]  # Различные количества итераций
methods = ['gradient', 'newton']  # Методы оптимизации

results = {}

for method in methods:
    if method == 'gradient':
        for lr in learning_rates:
            for num_iter in num_iterations_list:
                print(f"\nМетод: {method}, Learning rate: {lr}, Iterations: {num_iter}")
                weights = fit(X_train, y_train, method=method, num_iterations=num_iter, learning_rate=lr)
                predictions = predict(X_test, weights)
                acc = accuracy_score(y_test, predictions)
                prec = precision_score(y_test, predictions)
                rec = recall_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)
                results[(method, lr, num_iter)] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
                print_metrics(y_test, predictions)
    elif method == 'newton':
        for num_iter in num_iterations_list:
            print(f"\nМетод: {method}, Iterations: {num_iter}")
            weights = fit(X_train, y_train, method=method, num_iterations=num_iter)
            predictions = predict(X_test, weights)
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions)
            rec = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            results[(method, num_iter)] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
            print_metrics(y_test, predictions)

# Вывод результатов
print("\nРезультаты исследования гиперпараметров:")
for key, value in results.items():
    print(f"{key}: {value}")
