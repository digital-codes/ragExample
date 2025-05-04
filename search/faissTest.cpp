#include <faiss/IndexFlat.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

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
    int d = 16;  // Dimensionality
    int nb = 100; // Number of database vectors
    int nq = 4;   // Number of query vectors
    int k = 8;   // Number of nearest neighbors

    // Generate random database vectors
    std::vector<float> database_vectors(nb * d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < nb * d; i++) {
        database_vectors[i] = dis(gen);
    }

    // Normalize database vectors
    for (int i = 0; i < nb; i++) {
        normalize_vector(database_vectors.data() + i * d, d);
    }

    // Select query vectors from the database
    std::vector<float> query_vectors(nq * d);
    for (int i = 0; i < nq; i++) {
        int idx = i; // Select the first nq vectors as query vectors
        std::copy(database_vectors.begin() + idx * d, 
                  database_vectors.begin() + (idx + 1) * d, 
                  query_vectors.begin() + i * d);
    }

    // Print the first stored vector
    std::cout << "First stored vector:" << std::endl;
    for (int i = 0; i < d; i++) {
        std::cout << database_vectors[i] << " ";
    }
    std::cout << std::endl;

    // Print the first query vector
    std::cout << "First query vector:" << std::endl;
    for (int i = 0; i < d; i++) {
        std::cout << query_vectors[i] << " ";
    }
    std::cout << std::endl;

    // Initialize the FAISS index
    faiss::IndexFlatIP index(d); // Inner product index
    //faiss::IndexFlatL2 index(d); // L2 index
    index.add(nb, database_vectors.data());

    // Print the number of vectors in the index
    std::cout << "Number of vectors in the index: " << index.ntotal << std::endl;
    // Print the number of dimensions
    std::cout << "Number of dimensions: " << index.d << std::endl;
    // Print the vector at index 2 from the FAISS index
    std::vector<float> vector_at_index(d);
    index.reconstruct(2, vector_at_index.data());

    std::cout << "Vector at index 2:" << std::endl;
    for (int i = 0; i < d; i++) {
        std::cout << vector_at_index[i] << " ";
    }
    std::cout << std::endl;
    // Verify the retrieved vector against the database vector with the same ID
    bool is_equal = true;
    for (int i = 0; i < d; i++) {
        if (std::abs(vector_at_index[i] - database_vectors[2 * d + i]) > 1e-6) {
            is_equal = false;
            break;
        }
    }
    if (is_equal) {
        std::cout << "The databse vector matches the stored vector." << std::endl;
    } else {
        std::cout << "The database vector does not match the stored vector." << std::endl;
    }
    // Search for nearest neighbors
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

    return 0;
}
