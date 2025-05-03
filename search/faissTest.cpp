#include <faiss/IndexFlat.h>
#include <iostream>
#include <vector>
#include <cmath>

// g++ -O3 -o faissTest faissTest.cpp -I ./faissLib/include/ -L ./faissLib/lib64/ -lfaiss -fopenmp  -lopenblas
// chaeck /opt/faiss for faiss library
// https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
//
// Function to normalize a vector to unit length
void normalize_vector(float* vec, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < dim; i++) {
        vec[i] /= norm;
    }
}

int main() {
    int d = 4;  // Dimensionality
    int nb = 8; // Number of database vectors
    int nq = 2; // Number of query vectors

    std::vector<float> database_vectors = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 1.0, 1.0,
    };

    std::vector<float> query_vectors = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    };

    // Normalize database vectors
    for (int i = 0; i < nb; i++) {
        normalize_vector(database_vectors.data() + i * d, d);
    }

    // Normalize query vectors
    for (int i = 0; i < nq; i++) {
        normalize_vector(query_vectors.data() + i * d, d);
    }

    // Initialize the FAISS index
    faiss::IndexFlatIP index(d); // Inner product index
    index.add(nb, database_vectors.data());

    // Search for nearest neighbors
    int k = 10; // Number of nearest neighbors
    std::vector<faiss::idx_t> indices(k * nq);
    std::vector<float> distances(k * nq);

    index.search(nq, query_vectors.data(), k, distances.data(), indices.data());

    // Display the results
    for (int i = 0; i < nq; i++) {
        std::cout << "Query " << i << ":" << std::endl;
        for (int j = 0; j < k; j++) {
            std::cout << "  Neighbor " << j << ": Index=" << indices[i * k + j]
                      << ", Cosine Similarity=" << distances[i * k + j] << std::endl;
        }
    }

    return 0;
}
