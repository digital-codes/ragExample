#include <faiss/IndexFlat.h>
#include <iostream>
#include <vector>
#include <cmath>

// g++ -O3 -o faissTest faissTest.cpp -I ./faissLib/include/ -L ./faissLib/lib64/ -lfaiss -fopenmp  -lopenblas
// chaeck /opt/faiss for faiss library
// on lap3
// g++ -O3 -o faissTest faissTest.cpp -I /opt/faiss/include/ -L /opt/faiss/lib/ -lfaiss -fopenmp  -lopenblas
// https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
//

// from wiki 
// https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning
/*
The I/O functions are:

    write_index(index, "large.index"): writes the given index to file large.index

    Index * index = read_index("large.index"): reads a file


*/

#include <faiss/index_io.h>

// Function to store the FAISS index to a file
void store_index(const faiss::Index& index, const std::string& filename) {
    try {
        faiss::write_index(&index, filename.c_str());
        std::cout << "Index successfully stored to " << filename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error storing index: " << e.what() << std::endl;
    }
}

// Function to load the FAISS index from a file
faiss::Index* load_index(const std::string& filename) {
    try {
        faiss::Index* index = faiss::read_index(filename.c_str());
        std::cout << "Index successfully loaded from " << filename << std::endl;
        return index;
    } catch (const std::exception& e) {
        std::cerr << "Error loading index: " << e.what() << std::endl;
        return nullptr;
    }
}
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


    // Store the index to a file
    std::string filename = "faiss_index.index";
    store_index(index, filename);
    // Load the index from the file
    faiss::Index* loaded_index = load_index(filename);
    if (loaded_index) {
        // Perform a search with the loaded index
        std::vector<faiss::idx_t> loaded_indices(k * nq);
        std::vector<float> loaded_distances(k * nq);

        loaded_index->search(nq, query_vectors.data(), k, loaded_distances.data(), loaded_indices.data());

        // Display the results from the loaded index
        for (int i = 0; i < nq; i++) {
            std::cout << "Loaded Index Query " << i << ":" << std::endl;
            for (int j = 0; j < k; j++) {
                std::cout << "  Neighbor " << j << ": Index=" << loaded_indices[i * k + j]
                          << ", Cosine Similarity=" << loaded_distances[i * k + j] << std::endl;
            }
        }

        delete loaded_index; // Clean up
    }
    // Clean up


    return 0;
}
