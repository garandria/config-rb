import os
import pandas
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import graphviz
from utils import *
import sys
import alternatives

pandas.set_option('future.no_silent_downcasting', True)

def build_df(kallpath, reppath, cspath):
    kall = Path(kallpath).read_text().splitlines()
    repro = list(map(
        lambda x: {'True': 1, 'False': 0}[x], Path(reppath).read_text().splitlines()))
    table = dict()
    for opt in kall:
        table[opt] = [0 for _ in range(len(repro))]
    table["REPRODUCIBLE"] = repro
    for i, conf in enumerate(sorted(os.listdir(cspath))):
        for k, v in readconfig(os.path.join(cspath, conf)).items():
            table[k][i] = v
    return pandas.DataFrame.from_dict(table)


def filterdf(df):
    non_tristate = df.loc[:, ~(df.isin(['y', 'm'])).any()]\
               .drop(columns=["REPRODUCIBLE"]).columns
    df = df.drop(columns=non_tristate)
    encoding_map = {
        'y': 1,
        'm': 2,
    }
    df = df.replace(encoding_map)
    constant_columns = df.columns[df.nunique() == 1]
    df = df.drop(constant_columns, axis=1)
    return df


def model_train(df):
    # Separate the target variable and features
    X = df.drop(columns=["REPRODUCIBLE"])
    y = df["REPRODUCIBLE"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    print("Train: ", X_train.shape, "Test", X_test.shape)

    # Initialize the DecisionTreeClassifier
    clf = DecisionTreeClassifier()

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    # print(X_train.columns.to_list() == list(clf.feature_names_in_))

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    return clf


def dt2img(clf, output):
    # Export the decision tree to DOT format
    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=clf.feature_names_in_,
                               class_names=['Non-reproducible', 'Reproducible'],
                               filled=True,
                               special_characters=True)

    # Render and save the visualization using graphviz
    graph = graphviz.Source(dot_data)
    graph.render(output, format="png")
    # display(graph)


def dt_nodes(clf):
    return set(
        filter(
            None,
            [ clf.feature_names_in_[i] if i != -2 else None
              for i in clf.tree_.feature
             ]))


def mk_evolution(df, version):
    # Separate the target variable and features
    X = df.drop(columns=["REPRODUCIBLE"])
    y = df["REPRODUCIBLE"]

    # Range of test sizes to explore and number of repetitions
    test_sizes = np.arange(0.05, 1.0, 0.05)
    n_repetitions = 10

    # Dictionary to store accuracies for each test size
    accuracies = {test_size: [] for test_size in test_sizes}


    # Initialize dictionaries to store precision and recall for each test size
    precisions = {test_size: [] for test_size in test_sizes}
    recalls = {test_size: [] for test_size in test_sizes}

    # Loop over different test sizes and repeat training n times
    for test_size in test_sizes:
        for _ in range(n_repetitions):
            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)

            # Initialize and train the classifier
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)

            # Predict on the test set
            y_pred = clf.predict(X_test)

            # Calculate accuracy, precision, and recall
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Append the metrics to their respective lists
            accuracies[test_size].append(accuracy)
            precisions[test_size].append(precision)
            recalls[test_size].append(recall)

    # Calculate mean and standard deviation of precision and recall
    mean_precisions = [np.mean(precisions[test_size]) for test_size in test_sizes]
    std_precisions = [np.std(precisions[test_size]) for test_size in test_sizes]
    mean_recalls = [np.mean(recalls[test_size]) for test_size in test_sizes]
    std_recalls = [np.std(recalls[test_size]) for test_size in test_sizes]

    # Calculate mean and standard deviation of accuracies
    mean_accuracies = [np.mean(accuracies[test_size]) for test_size in test_sizes]
    std_accuracies = [np.std(accuracies[test_size]) for test_size in test_sizes]

    # Total number of samples in the dataset
    total_samples = len(df)

    # Convert test sizes to training sizes
    training_sizes = 1 - np.array(list(test_sizes))

    # Convert training sizes (percentage) to absolute number of training samples
    training_samples = [int(size * total_samples) for size in training_sizes]

    # Plotting accuracy, precision, and recall with shaded standard deviation
    plt.figure(figsize=(12, 8))

    # Plot mean accuracy, precision, and recall
    plt.plot(training_samples, mean_accuracies, '-o', label='Mean Accuracy')

    # Shade the standard deviation for each metric
    plt.fill_between(training_samples,
                    np.array(mean_accuracies) - np.array(std_accuracies),
                    np.array(mean_accuracies) + np.array(std_accuracies),
                    alpha=0.2)

    plt.title('Accuracy vs Number of configurations in the training set')
    plt.xlabel('Number of configurations')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.savefig(f"accuracy_vs_number_of_configurations-v{version}.png")
    # plt.show()

    # Plotting accuracy, precision, and recall with shaded standard deviation
    plt.figure(figsize=(12, 8))


    plt.plot(training_samples, mean_precisions, '-o', label='Mean Precision')
    plt.plot(training_samples, mean_recalls, '-o', label='Mean Recall')

    # Shade the standard deviation for each metric
    plt.fill_between(training_samples,
                    np.array(mean_precisions) - np.array(std_precisions),
                    np.array(mean_precisions) + np.array(std_precisions),
                    alpha=0.2)
    plt.fill_between(training_samples,
                    np.array(mean_recalls) - np.array(std_recalls),
                    np.array(mean_recalls) + np.array(std_recalls),
                    alpha=0.2)

    plt.title('Precision/Recall vs Number of configurations in the training set')
    plt.xlabel('Number of configurations')
    plt.ylabel('Precision/recall')
    plt.legend(loc='lower right')

    plt.savefig(f"precision_recall_vs_number_of_configurations-v{version}.png")
    # plt.show()




def main():
    version = sys.argv[1]
    kall = f"configs/kall-v.{version}"
    rb = f"results/rb-v{version}"
    confs = f"configs/{version}"
    df = filterdf(build_df(kall, rb, confs))
    # mk_evolution(df, version)
    clf = model_train(df)
    nodes = dt_nodes(clf)
    src = f"linux-{version}"
    output = "results"
    with open(os.path.join(output, f"dt_nodes-{version}"), "w") as streamout:
        streamout.write("\n".join(nodes))
    dt2img(clf, os.path.join(output, f"dt_v{version}"))

    # for n in nodes:
    #     filename = alternatives.get_option_source(n, src)
    #     if alternatives.is_choice(filename, n):
    #         alts = alternatives.get_alternatives(filename, n)
    #         print(f"* {n} [is_choice]")
    #         for alt in alts:
    #             print(f"  - {alt}")
    #     sels = alternatives.get_select(n, src)
    #     if sels:
    #         print(f"* {n} [select]")
    #         for s in sels:
    #             print(f"  - {s}")

    #     else:
    #         print(f"* {n}")


if __name__ == "__main__":
    main()
