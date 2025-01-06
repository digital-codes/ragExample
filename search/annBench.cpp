#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <cmath>
#include <future>
#include <Eigen/Dense>
#include <faiss/IndexFlat.h>
#include <faiss/utils/utils.h>


// -O3 important!
// g++ -O3 -o annBench annBench.cpp -I ./faissLib/include/ -L ./faissLib/lib64/ -lfaiss -fopenmp  -lopenblas -I /usr/include/eigen3

// Type aliases for simplicity
using Vector = Eigen::VectorXf;
using Matrix = Eigen::MatrixXf;

// Generate random embeddings for testing
Matrix generate_random_embeddings(size_t num_vectors, int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    Matrix embeddings(num_vectors, dim);
    for (size_t i = 0; i < num_vectors; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            embeddings(i, j) = dist(gen);
        }
        // Normalize each embedding
        embeddings.row(i) = embeddings.row(i) / embeddings.row(i).norm();
    }
    return embeddings;
}

// Brute-force top N search
std::vector<std::pair<int, float>> brute_force_search(const Vector& query, const Matrix& embeddings, int N) {
    std::vector<std::pair<int, float>> results;

    for (int i = 0; i < embeddings.rows(); ++i) {
        float similarity = query.dot(embeddings.row(i));
        results.emplace_back(i, similarity);
    }

    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.second > b.second; // Sort by similarity descending
    });

    results.resize(N); // Keep only top N
    return results;
}

// Brute-force top N search (parallelized)
std::vector<std::pair<int, float>> parallel_brute_force_search(const Vector& query, const Matrix& embeddings, int N, int num_threads) {
    size_t num_vectors = embeddings.rows();
    size_t chunk_size = num_vectors / num_threads;

    // Results for each thread
    std::vector<std::future<std::vector<std::pair<int, float>>>> futures;

    // Launch threads
    for (int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = (t == num_threads - 1) ? num_vectors : start_idx + chunk_size;

        futures.push_back(std::async(std::launch::async, [&, start_idx, end_idx]() {
            std::vector<std::pair<int, float>> local_results;
            for (size_t i = start_idx; i < end_idx; ++i) {
                float similarity = query.dot(embeddings.row(i));
                local_results.emplace_back(i, similarity);
            }

            // Partial sort to get top N results for this thread
            std::partial_sort(local_results.begin(), local_results.begin() + std::min(N, static_cast<int>(local_results.size())), local_results.end(),
                              [](const auto& a, const auto& b) { return a.second > b.second; });
            local_results.resize(std::min(N, static_cast<int>(local_results.size())));
            return local_results;
        }));
    }

    // Gather results from all threads
    std::vector<std::pair<int, float>> all_results;
    for (auto& future : futures) {
        auto thread_results = future.get();
        all_results.insert(all_results.end(), thread_results.begin(), thread_results.end());
    }

    // Global top N sorting
    std::partial_sort(all_results.begin(), all_results.begin() + std::min(N, static_cast<int>(all_results.size())), all_results.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
    all_results.resize(std::min(N, static_cast<int>(all_results.size())));

    return all_results;
}


std::pair<std::chrono::milliseconds, std::chrono::milliseconds>
faiss_index_and_search(const Vector& query, const Matrix& embeddings, int N) {

    size_t nb = embeddings.rows();
    size_t d = embeddings.cols();

    // Measure indexing time
    auto start_indexing = std::chrono::high_resolution_clock::now();
    // Initialize the FAISS index
    faiss::IndexFlatIP index(d); // Inner product index
    // add database vectors, already normalized
    index.add(nb, embeddings.data());
    auto end_indexing = std::chrono::high_resolution_clock::now();


    // Measure searching time
    // Search for nearest neighbors
    std::vector<faiss::idx_t> indices(N);
    std::vector<float> distances(N);

    auto start_searching = std::chrono::high_resolution_clock::now();
    index.search(1, query.data(), N, distances.data(), indices.data());
    auto end_searching = std::chrono::high_resolution_clock::now();


    return {
        std::chrono::duration_cast<std::chrono::milliseconds>(end_indexing - start_indexing),
        std::chrono::duration_cast<std::chrono::milliseconds>(end_searching - start_searching)};
}



void benchmark(size_t num_vectors, int dim, int top_n) {
    std::cout << "Benchmarking with " << num_vectors << " vectors of dimension " << dim << "\n";

    // Generate data
    auto start = std::chrono::high_resolution_clock::now();
    auto embeddings = generate_random_embeddings(num_vectors, dim);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Creating embeddings time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    Vector query = Vector::Random(dim).normalized();
    /*
    // Brute-force search
    start = std::chrono::high_resolution_clock::now();
    auto brute_results = brute_force_search(query, embeddings, top_n);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Brute-force time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    */
    // Parallel Brute-force search
    start = std::chrono::high_resolution_clock::now();
    auto brute_results = parallel_brute_force_search(query, embeddings, top_n,8);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel Brute-force time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    // faiss search (with separate timings)
    auto [faiss_indexing_time, faiss_searching_time] = faiss_index_and_search(query, embeddings, top_n);

    std::cout << "Faiss indexing time: " << faiss_indexing_time.count() << " ms\n";
    std::cout << "faiss searching time: " << faiss_searching_time.count() << " ms\n";
    std::cout << "----\n";



}

int main() {
    const int dim = 768;
    const int top_n = 10; // Find top 10 matches

    std::cout << "FAISS is compiled with GPU support: " 
              << faiss::get_compile_options() << std::endl;
    auto ompThreads = omp_get_max_threads();
    omp_set_num_threads(ompThreads); // Set the number of threads
    std::cout << "Number of faiss threads used: " << ompThreads << std::endl;


    // Test cases: Increasing number of vectors
    benchmark(1000, dim, top_n);    // 10^4
    benchmark(10000, dim, top_n);    // 10^4
    benchmark(100000, dim, top_n);    // 10^5
    benchmark(1000000, dim, top_n);   // 10^6
    // out of memory with faiss on 10^7
    //benchmark(10000000, dim, top_n);  // 10^7

    return 0;
}

/*
kugel@tux3:~/temp/py/ragExample/search$ g++  -O3 -o annBench annBench.cpp -I ./faissLib/include/ -L ./faissLib/lib64/ -lfaiss -fopenmp  -lopenblas -I /usr/include/eigen3
kugel@tux3:~/temp/py/ragExample/search$ ./annBench 
FAISS is compiled with GPU support: GENERIC 
Benchmarking with 1000 vectors of dimension 768
Creating embeddings time: 8 ms
Parallel Brute-force time: 0 ms
Faiss indexing time: 1 ms
faiss searching time: 3 ms
----
Benchmarking with 10000 vectors of dimension 768
Creating embeddings time: 51 ms
Parallel Brute-force time: 2 ms
Faiss indexing time: 12 ms
faiss searching time: 16 ms
----
Benchmarking with 100000 vectors of dimension 768
Creating embeddings time: 445 ms
Parallel Brute-force time: 25 ms
Faiss indexing time: 124 ms
faiss searching time: 167 ms
----
Benchmarking with 1000000 vectors of dimension 768
Creating embeddings time: 4471 ms
Parallel Brute-force time: 219 ms
Faiss indexing time: 1220 ms
faiss searching time: 1620 ms
----
Benchmarking with 10000000 vectors of dimension 768
Creating embeddings time: 47157 ms
Parallel Brute-force time: 1894 ms

*/

/*

If your query batch size is 1, FAISS might perform poorly compared to parallel brute-force implementations because the overhead of managing the index and threads becomes significant relative to the small workload. Processing queries one by one is not the most efficient way to use FAISS, especially for small datasets or brute-force search.
Why Query Batch Size 1 is Inefficient

    Thread Overhead:
        FAISS uses OpenMP for parallelization. With a batch size of 1, it’s difficult to fully utilize the available CPU threads because there's limited work to distribute.

    Index Overhead:
        Even lightweight FAISS indexes (IndexFlat) involve some initialization and search overhead that become noticeable when querying a single vector at a time.

    Poor Cache Utilization:
        Querying multiple vectors in a batch allows FAISS to benefit from data locality and efficient memory access. Single queries prevent these optimizations.


Reasons FAISS Might Be Slower

    Index Structure Overhead
        Some FAISS indexes, like IndexIVF or IndexHNSW, involve preprocessing steps (e.g., cluster assignment or graph traversal) that introduce overhead for small datasets.
        For up to 10M vectors, the additional computation for building or using these structures might not be worth the tradeoff compared to brute force.

    Flat Index on CPU
        If you're using IndexFlat for brute force search on a CPU, FAISS computes distances sequentially by default.
        On modern CPUs with many cores, a custom parallel brute-force implementation can outperform FAISS if FAISS doesn't leverage all available cores effectively.

    Incorrect Parallelization Configuration
        FAISS uses OpenMP for parallelization, but it may not be leveraging all available CPU threads due to incorrect environment settings.

    Query-Vector Overhead
        If the query batch size is too small, the overhead of using FAISS can outweigh its benefits. Parallel brute-force implementations might handle small batches more efficiently.

    Dataset Size and Dimensionality
        FAISS is optimized for large datasets (hundreds of millions of vectors) and high-dimensional data. For small datasets (e.g., 10M vectors) or low-dimensional data, the overhead might outweigh its benefits.

How to Address These Issues
1. Use IndexFlat with Parallelization

    FAISS's IndexFlat is a brute-force index that supports parallel search via OpenMP. Ensure OpenMP is properly configured to utilize multiple threads.


2. Increase Query Batch Size

    Querying vectors in larger batches reduces per-query overhead. Adjust the batch size to match your workload.

        int nq = 100; // Number of query vectors in a batch
        index.search(nq, queries.data(), k, distances.data(), indices.data());

3. Consider Using GPU


4. Switch to More Efficient Indexes

    If brute force is still faster, consider switching to a more efficient FAISS index type that reduces the search space, like IndexIVFFlat:
        Train the index:

        index.train(nb, data); // Train on a subset of vectors
        index.add(nb, data);   // Add vectors

        This can dramatically reduce the number of comparisons while maintaining high recall.



Why FAISS Is Designed This Way

FAISS is optimized for large-scale, high-dimensional datasets where exact brute force becomes infeasible. For small datasets, brute force or simpler custom implementations may outperform FAISS due to its overhead.
Recommendations

    Use IndexFlat for exact search with proper parallelization for datasets of up to 10M vectors.
    For larger datasets, switch to IndexIVFFlat or similar approximate indexes.
    For faster brute-force search, consider using GPU-based FAISS indexes.


*/