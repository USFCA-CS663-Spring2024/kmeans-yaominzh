[toc]

# goals

1. To develop your version K-Means using the algorithm specified below. (This has been an interview question for jobs using machine learning.)
2. To compare the performance of different implementations.
3. To demonstrate your understanding of clustering algorithms like K-Means, DBSCAN and Hierarchical. (This will also start introducing you to skills needed for data challenges.)
4. To extend the functionality of the developed K-Means implementation through additional parameters. (This shows your ability to develop novel or custom algorithms.)

# background

We covered the algorithms and hyperparameter tuning for K-Means, DBSCAN and Hierarchical clustering. The algorithm (in psuedocode) for the K-Means algorithm is as follows:

place k centroids \(\mu_1,\mu_2,...,\mu_k \in \mathbb{R}^n\) randomly
repeat to convergence:
$\quad$    foreach x \(\in\) test_x: # Assignment
$\quad \quad$         $c^{(i)}$ = index of closest centroid to x
$\quad$    foreach k \(\in\) centroids: # Update step
$\quad \quad$         \(\mu_k\) = mean of all points assigned to centroid k
> converge when the centroids do not change


```python
# Place k centroids randomly
place k centroids μ1, μ2, ..., μk ∈ ℝ^n randomly

# Repeat until convergence
repeat to convergence:
    # For each point x in the dataset
    foreach x ∈ test_x:
        c^(i) = index of closest centroid to x

    # For each centroid
    foreach k ∈ centroids:
        μk = mean of all points assigned to centroid k
```
# process

1. Implement K-Means as described above.
2. Do a performance analysis among the expected labels, your implementation of K-Means (excluding the extended version), the version offered by the Scikit-learn library
3. Choose and run clustering algorithms against some datasets and evaluate the results.

In addition to the required (above), there is one optional part:

4. Extend K-Means so that it balances the number of instances (rows) per cluster