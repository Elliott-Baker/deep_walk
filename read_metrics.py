import sys
import re
import matplotlib.pyplot as plt


# Draws the plots for different metrics vs. embedding size
def draw_metrics_embed_size(lines):
    x = []
    accuracy = []
    precision = []
    recall = []
    macro_f1 = []
    micro_f1 = []

    for line in lines:
        if "Embedding Size: " in line:
            matches = re.findall(r"[0-9]+", line)
            x.append(matches[0])
        # if "Walk length: " in line:
        #     matches = re.findall(r"[0-9]+", line)
        #     x.append(matches[0])
        # if "Walks per vertex: " in line:
        #     matches = re.findall(r"[0-9]+", line)
        #     x.append(matches[0])
        elif "Accuracy: " in line:
            accuracy.append(find_match(line))
        elif "Precision: " in line:
            precision.append(find_match(line))
        elif "Recall: " in line:
            recall.append(find_match(line))
        elif "Macro" in line:
            macro_f1.append(find_match(line))
        elif "Micro" in line:
            micro_f1.append(find_match(line))

    plt.plot(x, accuracy, label="Accuracy")
    plt.plot(x, precision, label="Precision")
    plt.plot(x, recall, label="Recall")
    plt.plot(x, macro_f1, label="Macro F-1 Score")
    plt.plot(x, micro_f1, label="Micro F-1 Score")
    plt.xlabel("Walk Length")
    plt.xticks((x[0], x[int(len(x) / 2)], x[-1]))
    plt.title("Metrics vs Walk Length")
    plt.legend()
    plt.savefig("plots/karate_graph_embed_size")


def draw_metrics_walk_length(lines):
    x = []
    accuracy = []
    precision = []
    recall = []
    macro_f1 = []
    micro_f1 = []

    for line in lines:
        if "Walk length: " in line:
            matches = re.findall(r"[0-9]+", line)
            x.append(matches[0])
        elif "Accuracy: " in line:
            accuracy.append(find_match(line))
        elif "Precision: " in line:
            precision.append(find_match(line))
        elif "Recall: " in line:
            recall.append(find_match(line))
        elif "Macro" in line:
            macro_f1.append(find_match(line))
        elif "Micro" in line:
            micro_f1.append(find_match(line))

    plt.plot(x, accuracy, label="Accuracy")
    plt.plot(x, precision, label="Precision")
    plt.plot(x, recall, label="Recall")
    plt.plot(x, macro_f1, label="Macro F-1 Score")
    plt.plot(x, micro_f1, label="Micro F-1 Score")
    plt.xlabel("Walk Length")
    plt.xticks((x[0], x[int(len(x) / 2)], x[-1]))
    plt.title("Metrics vs Walk Length")
    plt.legend()
    plt.savefig("plots/karate_graph_walk_length")


def draw_metrics_walks_per_vertex(lines):
    x = []
    accuracy = []
    precision = []
    recall = []
    macro_f1 = []
    micro_f1 = []

    for line in lines:
        if "Walks per vertex: " in line:
            matches = re.findall(r"[0-9]+", line)
            x.append(matches[0])
        elif "Accuracy: " in line:
            accuracy.append(find_match(line))
        elif "Precision: " in line:
            precision.append(find_match(line))
        elif "Recall: " in line:
            recall.append(find_match(line))
        elif "Macro" in line:
            macro_f1.append(find_match(line))
        elif "Micro" in line:
            micro_f1.append(find_match(line))

    plt.plot(x, accuracy, label="Accuracy")
    plt.plot(x, precision, label="Precision")
    plt.plot(x, recall, label="Recall")
    plt.plot(x, macro_f1, label="Macro F-1 Score")
    plt.plot(x, micro_f1, label="Micro F-1 Score")
    plt.xlabel("Walks per vertex")
    plt.xticks((x[0], x[int(len(x) / 2)], x[-1]))
    plt.title("Metrics vs Walks per vertex")
    plt.legend()
    plt.savefig("plots/karate_graph_walks_per_vertex")


def find_match(line):
    matches = re.findall(r"0\.[0-9]+", line)
    if len(matches) == 0:
        return 0
    return float(matches[0])


# Reads the input file and returns the lines
def read_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()

        f.close()
    return lines


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 read_metrics.py [filename] [hyperparameter]")

    filename = sys.argv[1]
    hyperparameter = sys.argv[2]
    lines = read_file(file_name=filename)
    if hyperparameter == "embeddingsize":
        draw_metrics_embed_size(lines)
    elif hyperparameter == "walklength":
        draw_metrics_walk_length(lines)
    elif hyperparameter == "walkspervertex":
        draw_metrics_walks_per_vertex(lines)
    else:
        print("Invalid hyperparameter")


if __name__ == "__main__":
    main()
