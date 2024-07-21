#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <numeric>

#include <ctime>

using namespace std;

class KMeans {
public:
    int n_clusters;
    int max_iter;
    vector<vector<double>> centroids;

    KMeans(int n_clusters = 2, int max_iter = 100) : n_clusters(n_clusters), max_iter(max_iter) {
        srand(time(0));
    }

    vector<int> fit_predict(vector<vector<double>>& data) {
        // Randomly initialize centroids
        initialize_centroids(data);

        vector<int> cluster_group(data.size());

        for (int i = 0; i < max_iter; i++) {
            // Assign clusters
            assign_clusters(data, cluster_group);

            // Move centroids
            vector<vector<double>> new_centroids = move_centroids(data, cluster_group);

            // Check convergence
            if (centroids == new_centroids) {
                break;
            }

            centroids = new_centroids;
        }

        return cluster_group;
    }

private:
    void initialize_centroids(vector<vector<double>>& data) {
        centroids.clear();
        vector<int> random_indices(data.size());
        iota(random_indices.begin(), random_indices.end(), 0);
        random_shuffle(random_indices.begin(), random_indices.end());

        for (int i = 0; i < n_clusters; i++) {
            centroids.push_back(data[random_indices[i]]);
        }
    }

    void assign_clusters(vector<vector<double>>& data, vector<int>& cluster_group) {
        for (size_t i = 0; i < data.size(); i++) {
            double min_distance = numeric_limits<double>::max();
            int cluster_index = 0;

            for (int j = 0; j < n_clusters; j++) {
                double distance = euclidean_distance(data[i], centroids[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    cluster_index = j;
                }
            }

            cluster_group[i] = cluster_index;
        }
    }

    vector<vector<double>> move_centroids(vector<vector<double>>& data, vector<int>& cluster_group) {
        vector<vector<double>> new_centroids(n_clusters, vector<double>(data[0].size(), 0));
        vector<int> counts(n_clusters, 0);

        for (size_t i = 0; i < data.size(); i++) {
            int cluster_index = cluster_group[i];
            for (size_t j = 0; j < data[i].size(); j++) {
                new_centroids[cluster_index][j] += data[i][j];
            }
            counts[cluster_index]++;
        }

        for (int i = 0; i < n_clusters; i++) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < new_centroids[i].size(); j++) {
                    new_centroids[i][j] /= counts[i];
                }
            }
        }

        return new_centroids;
    }

    double euclidean_distance(const vector<double>& a, const vector<double>& b) {
        double sum = 0;
        for (size_t i = 0; i < a.size(); i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sqrt(sum);
    }
};

int main() {
    vector<vector<double>> data = {
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {8.0, 8.0},
        {1.0, 0.6},
        {9.0, 11.0}
    };

    KMeans kmeans(2, 100);
    vector<int> clusters = kmeans.fit_predict(data);

    for (size_t i = 0; i < clusters.size(); i++) {
        cout << "Point " << i << " is in cluster " << clusters[i] << endl;
    }

    return 0;
}
