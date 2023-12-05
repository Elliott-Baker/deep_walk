import sys
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import KMeans


# Generates random walks then trains the skigram model
def deepwalk(graph, window_size, embedding_size, walks_per_vertex, walk_length):
    walks = generate_random_walks(graph, walks_per_vertex, walk_length)
    model = Word2Vec(
        sentences=walks,
        vector_size=embedding_size,
        window=window_size,
        min_count=1,
        sg=1,
        workers=4,
    )
    model.train(walks, total_examples=len(walks), epochs=10)
    # draw_random_walk(graph, walks, 0)
    return model


# Generates random walks
def generate_random_walks(graph, walks_per_vertex, walk_length):
    walks = []
    # for v in range(len(graph.nodes())):
    for v in graph.nodes:
        for i in range(walks_per_vertex):
            walks.append(random_walk(graph, v, walk_length))
    return walks


# Generates a single random walks for a given vertex
def random_walk(graph, v, walk_length):
    walk = [v]
    current_vertex = v
    for i in range(walk_length):
        current_vertex = choose_random_vertex(graph, current_vertex)
        walk.append(current_vertex)
    return [str(vertex) for vertex in walk]


# Chooses the next vertex in the random walk
def choose_random_vertex(graph, vertex):
    neighbors = list(graph.neighbors(vertex))
    if len(neighbors) == 0:
        return vertex
    if len(neighbors) == 1:
        return neighbors[0]
    random_index = random.randint(0, len(neighbors) - 1)
    random_vertex = neighbors[random_index]
    return random_vertex


# Prints the order of a given random walk
def print_random_walk(walk):
    print("A Random Walk for Vertex {}".format(walk))
    print(walk)


# Draws the random walk in matplotlib
def draw_random_walk(graph, walks, starting_node):
    walk = walks[starting_node]
    node_colors = ["green" if str(node) in walk else "gray" for node in graph.nodes]
    node_colors[starting_node] = "orange"
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color=node_colors)
    plt.legend(
        {"Starting Node": "orange", "Random Walk": "green", "Unvisited Nodes": "gray"}
    )
    plt.savefig("plots/random_walk_vertex_{}".format(walk[0]))


# Draws the karate graph in matplotlib
def draw_karate_graph(graph):
    node_colors = [
        "red" if graph.nodes[node]["club"] == "Mr. Hi" else "blue"
        for node in graph.nodes
    ]
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color=node_colors)
    plt.savefig("plots/graph")


# Plots the graph with the actual labels as colors
def draw_actual_labels(graph, model):
    for i in range(len(graph.nodes)):
        if graph.nodes[i]["club"] == "Mr. Hi":
            plt.scatter(model.wv.vectors[i][0], model.wv.vectors[i][1], c="red")
        else:
            plt.scatter(model.wv.vectors[i][0], model.wv.vectors[i][1], c="blue")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.legend(["Mr. Hi", "Officer"])
    plt.title("Word Vector Embeddings with Actual Labels")
    plt.savefig("plots/actual_labels")


# Plots the graph with the predicted labels as colors
def draw_predicted_labels(graph, predictions, model):
    for i in range(len(graph.nodes)):
        if predictions[i] == 1:
            plt.scatter(model.wv.vectors[i][0], model.wv.vectors[i][1], c="red")
        else:
            plt.scatter(model.wv.vectors[i][0], model.wv.vectors[i][1], c="blue")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.legend(["Mr. Hi", "Officer"])
    plt.title("Word Vector Embeddings with Predicted Labels")
    plt.savefig("plots/predicted_labels")


# Reads in the twitch user graph data
def read_graph(file_name):
    with open(file_name, "r") as f:
        G = nx.Graph()
        isFirst = True
        every_ten = 0
        for line in f:
            every_ten += 1
            if isFirst:
                isFirst = False
                continue
            if every_ten % 100 == 0:
                line = line.strip().split(",")
                G.add_edge(int(line[0]), int(line[1]))
        return G


def reduce_graph(graph, target_nodes_ratio, random_seed=None):
    # Make a copy of the original graph to avoid modifying it
    reduced_graph = graph.copy()

    # Set a random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Calculate the target number of nodes to keep
    target_nodes = int(target_nodes_ratio * len(reduced_graph.nodes))

    # Remove all zero-degree nodes
    degrees = dict(reduced_graph.degree())
    nodes_to_remove = [node for node, degree in degrees.items() if degree == 0]
    while nodes_to_remove != []:
        reduced_graph.remove_node(nodes_to_remove[0])
        del nodes_to_remove[0]

    # Continue removing nodes until the target is reached
    while len(reduced_graph.nodes) > target_nodes:
        # Get nodes with degree 1 (or any other criteria)
        low_degree_nodes = [
            node for node, degree in reduced_graph.degree() if degree <= 10
        ]

        # If there are no nodes with degree 1, break the loop
        if not low_degree_nodes:
            break

        # Randomly choose a node with low degree to remove
        node_to_remove = random.choice(low_degree_nodes)

        # Remove the chosen node and its incident edges
        reduced_graph.remove_node(node_to_remove)

    return reduced_graph


def logistic_regression(X, y):
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf


# Creates graph and hyperparameters based on user input and runs deepwalk
def main():
    if len(sys.argv) > 6 or len(sys.argv) < 2 or sys.argv[1] == "--help":
        print(
            "Usage: python3 deepwalk.py [graph name] [window size] [embedding size] [walks/vertex] [walk length] or python3 deepwalk.py --help"
        )
        print("Graphs: karate, twitch")
    else:
        graph_name = sys.argv[1]
        window_size = int(sys.argv[2])
        embedding_size = int(sys.argv[3])
        walks_per_vertex = int(sys.argv[4])
        walk_length = int(sys.argv[5])

        if graph_name == "karate":
            G = nx.karate_club_graph()
            embeddings = deepwalk(
                graph=G,
                window_size=window_size,
                embedding_size=embedding_size,
                walks_per_vertex=walks_per_vertex,
                walk_length=walk_length,
            )

            X = embeddings.wv.vectors
            y = []  # actual labels

            # Plot the word vector embeddings with actual labels as colors
            vec_0 = []
            vec_1 = []
            for i in range(len(X)):
                vec_0.append(X[i][0])
                vec_1.append(X[i][1])
                if G.nodes[i]["club"] == "Mr. Hi":
                    y.append(1)
                else:
                    y.append(0)

            y = np.array(y)
            clf = logistic_regression(X, y)
            predictions = clf.predict(X)
            print("Embedding Size: {}".format(embedding_size))
            print("Walk length: {}".format(walk_length))
            print("Walks per vertex: {}".format(walks_per_vertex))
            print("Accuracy: {}".format(clf.score(X, y)))
            print("Precision: {}".format(precision_score(y, predictions)))
            print("Recall: {}".format(recall_score(y, predictions)))
            print(
                "Macro F-1 Score: {}".format(f1_score(y, predictions, average="macro"))
            )
            print(
                "Micro F-1 Score: {}".format(f1_score(y, predictions, average="micro"))
            )
            print("\n")

            # # Plot the word vector embeddings with predicted labels as colors
            # draw_predicted_labels(G, predictions, embeddings)

            # # Plot the word vector embeddings with actual labels as colors
            # draw_actual_labels(G, embeddings)

            # # Plot the graph
            # draw_karate_graph(G)

        if graph_name == "twitch":
            G = read_graph("large_twitch_edges.csv")
            G = reduce_graph(G, 0.2, random_seed=0)
            print("Training Deepwalk with Twitch user data...")
            print("Number of nodes in graph: {}".format(len(G.nodes)))
            print("Number of edges in graph: {}".format(len(G.edges)))
            embeddings = deepwalk(
                graph=G,
                window_size=window_size,
                embedding_size=embedding_size,
                walks_per_vertex=walks_per_vertex,
                walk_length=walk_length,
            )

            vecs = embeddings.wv.vectors
            kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(vecs)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            plt.scatter(vecs[:, 0], vecs[:, 1], c=labels)
            plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=100)
            plt.title("K-means Clustering of Latent Embeddings")
            plt.xlabel("Vector Feature 1")
            plt.ylabel("Vector Feature 2")
            plt.savefig("plots/twitch_clustering3")
            print("Done! See plots/twitch_clustering2.png for results.")


if __name__ == "__main__":
    main()
