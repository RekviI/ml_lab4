from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# 1.Відкриття файлу та зчитування даних
data = pd.read_csv('dataset2_l4.txt', sep=',')

# 2.Визначити та вивести кількість записів.
print("Кількість записів у файлі: ", data.shape[0])

# 3.Вивести атрибути набору даних.
print("Атрибути набору даних: ", ", ".join(data.columns))

# 4.З’ясувати збалансованість набору даних.
print(data['Class'].value_counts())

# 5.Отримати двадцять варіантів перемішування набору даних та розділення його на 
# навчальну (тренувальну) та тестову вибірки, використовуючи функцію ShuffleSplit. 
# Сформувати начальну та тестові вибірки на основі обраного користувачем варіанту.
def user_input_for_select_shuffle_split_number(n: int) -> int:
  try:
    selected_variant = int(input(f'Виберіть номер варіанту від 1 до {n}: ')) - 1
    if selected_variant < 0 or selected_variant >= n:
      print('\nВведене число виходить за визначені межі!\n')
      return user_input_for_select_shuffle_split_number(n)
  except:
    print('\nВведене число не є натуральним!\n')
    return user_input_for_select_shuffle_split_number(n)

  return selected_variant

test_size = 0.2
n_splits = 20

shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=5)

split_variants = []
for train_index, test_index in shuffle_split.split(data):
    train_df = data.iloc[train_index]
    test_df = data.iloc[test_index]
    split_variants.append((train_df, test_df))

selected_variant = user_input_for_select_shuffle_split_number(n_splits)

train_df_selected, test_df_selected = split_variants[selected_variant]

print(test_df_selected.head(10))


# 6.Використовуючи функцію KNeighborsClassifier бібліотеки scikit-learn, збудувати 
# класифікаційну модель на основі методу k найближчих сусідів (кількість сусідів обрати
# самостійно, вибір аргументувати) та навчити її на тренувальній вибірці, вважаючи, 
# що цільова характеристика визначається стовпчиком Class, а всі інші виступають вролі вихідних аргументів.
k = 3

k_neighbors_classifier = KNeighborsClassifier(n_neighbors=k)

x_train = train_df_selected.drop(columns=['Class'])
y_train = train_df_selected['Class']
k_neighbors_classifier.fit(x_train, y_train)

x_test = test_df_selected.drop(columns=['Class'])
y_test = test_df_selected['Class']


# 7.Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки.
# Представити результати роботи моделі на тестовій вибірці графічно.
def calculate_metrics(model, x_cord, y_cord):
    pred = model.predict(x_cord)
    accuracy = metrics.accuracy_score(y_cord, pred)
    precision = metrics.precision_score(y_cord, pred, average='weighted')
    recall = metrics.recall_score(y_cord, pred, average='weighted')
    f1_score = metrics.f1_score(y_cord, pred, average='weighted')
    mcc = metrics.matthews_corrcoef(y_cord, pred)
    ba = metrics.balanced_accuracy_score(y_cord, pred)
    return [accuracy, precision, recall, f1_score, mcc, ba]

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'MCC', 'Balanced Accuracy']
metrics_test_data = calculate_metrics(k_neighbors_classifier, x_test, y_test)
metrics_train_data = calculate_metrics(k_neighbors_classifier, x_train, y_train)

data_test_graph = pd.DataFrame({'Тестова вибірка': metrics_test_data}, index=metric_names)

data_test_graph.plot(kind='bar', figsize=(10, 6))
plt.title('Метрики для тестової вибірки')
plt.ylabel('Значення')
plt.xlabel('Метрика')
plt.xticks(rotation=45)
plt.legend()
plt.savefig("Metrics_Test_Data.png")
plt.show()

data_test_train_graph = pd.DataFrame({'Тестова вибірка': metrics_test_data, 'Тренувальна вибірка': metrics_train_data}, index=metric_names)
data_test_train_graph.plot(kind='bar', figsize=(10, 6))
plt.title('Порівняння метрик для тестової та тренувальної вибірок')
plt.ylabel('Значення')
plt.xlabel('Метрика')
plt.xticks(rotation=45)
plt.legend()
plt.savefig("Metrics_Comparison.png")
plt.show()


# 8.Обрати алгоритм KDTree та з’ясувати вплив розміру листа (від 20 до 200 з кроком 5) 
# на результати класифікації. Результати представити графічно.
leaf_sizes = range(20, 201, 5)
accuracy_results = []
precision_results = []
recall_results = []
f1_score_results = []
mcc_results = []
ba_results = []

for leaf_size in leaf_sizes:
    model = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', leaf_size=leaf_size)
    model.fit(x_train, y_train)
    test_metrics = calculate_metrics(model, x_train, y_train)
    
    accuracy_results.append(test_metrics[0])
    precision_results.append(test_metrics[1])
    recall_results.append(test_metrics[2])
    f1_score_results.append(test_metrics[3])
    mcc_results.append(test_metrics[4])
    ba_results.append(test_metrics[5])

plt.figure(figsize=(10, 6))
plt.plot(leaf_sizes, accuracy_results, label="Accuracy")
plt.plot(leaf_sizes, precision_results, label="Precision")
plt.plot(leaf_sizes, recall_results, label="Recall")
plt.plot(leaf_sizes, f1_score_results, label="F1-score")
plt.plot(leaf_sizes, mcc_results, label="MCC")
plt.plot(leaf_sizes, ba_results, label="Balanced Accuracy")
plt.xticks(np.arange(min(leaf_sizes), max(leaf_sizes)+1, 10), rotation=45)
plt.title('Метрики на розмір листа')
plt.xlabel('Розмір листа')
plt.ylabel('Значення метрики')
plt.legend()
plt.grid()
plt.savefig("Metrics_Leaf_Size.png")
plt.show()
