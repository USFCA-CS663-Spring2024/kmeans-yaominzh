from cluster import cluster
import numpy as np
class KMeans(cluster):
    def __init__(self, k=5, max_iterations=100, balanced=False):
        super().__init__()
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
        self.balanced = balanced

    # according to slide 13:
    # it is often attractive to fix the maximum iterations to some finite number — eg. 100
    def generate_pairs(self, k,X):
        pairs = []
        xs = [x[0] for x in X]
        ys = [y[1] for y in X]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        for i in range(k):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            pair = [x, y]
            pairs.append(pair)
        return pairs
    def fit(self, X):
        """
        pseudocode from lesson notes:
        place k centroids μ1, μ2, ..., μk ∈ ℝ^n randomly

        # Repeat until convergence
        repeat to convergence:
            # For each point x in the dataset
            foreach x ∈ test_x:
                c^(i) = index of closest centroid to x

            # For each centroid
            foreach k ∈ centroids:
                μk = mean of all points assigned to centroid k
        """
        # converge when no x^i changes its cluster
        np.random.seed()

        # self.centroids = [X[i] for i in np.random.choice(range(len(X)), self.k, replace=False)]
        pairs = self.generate_pairs(self.k,X)
        # to illustrate the random pairs for performance comparison
        print(pairs)
        self.centroids = pairs

        for _ in range(self.max_iterations):
            # Assignment step
            clusters = [[] for _ in range(self.k)]
            for xi in X:
                # calculate the Euclidean distance between two points
                distances = [np.linalg.norm(np.array(xi) - np.array(centroid)) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(xi)

            # Update step
            new_centroids = [np.mean(cluster, axis=0).tolist() for cluster in clusters if cluster]

            # Check for convergence (if centroids do not change)
            if new_centroids == self.centroids:
                break

            self.centroids = new_centroids
        # Assign labels based on final centroids
        if not self.balanced:
            labels = []
            for xi in X:
                distances = [np.linalg.norm(np.array(xi) - np.array(centroid)) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                labels.append(cluster_index)

            return labels, self.centroids

        # Implement balanced clustering
        # my idea is to prepare k slots, and then assign each point to the closest centroid(slot), similar idea as Round Robin
        n = len(X)
        visited = set()
        slots = [[] for _ in range(self.k)] #  each list contains the indices of the points in the cluster
        labels = [0] * len(X)
        for i in range(n):
            cur_centroiids = self.centroids[i % self.k]
            distances = []
            for j in range(n):
                if j in visited:
                    distances.append(float('inf'))
                else:
                    distances.append(np.linalg.norm(np.array(X[j]) - np.array(cur_centroiids)))
            min_index = distances.index(min(distances))
            visited.add(min_index)
            slots[i % self.k].append(min_index)

        #print(slots)
        for i in range(self.k):
            for idx in slots[i]:
                labels[idx] = i
        return labels, self.centroids

# Test
if __name__ == '__main__':
    X_example = [[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]]
    kmeans_example = KMeans(k=2, max_iterations=100,balanced=True)
    labels_example, centroids_example = kmeans_example.fit(X_example)

    print(labels_example)
    print(centroids_example)