/**
 * @file annService.cpp
 * @brief A service for performing approximate nearest neighbor (ANN) search using brute-force method.
 *
 * This program loads vectors from binary files, normalizes them, and provides an HTTP service to perform
 * similarity searches against these vectors. The search is performed using a parallel brute-force method
 * to find the top N most similar vectors to a given query vector.
 *
 * The program uses the Eigen library for vector and matrix operations, and Boost libraries for HTTP server
 * and JSON handling.
 *
 * Usage:
 * @code
 *  g++ -o annService annService.cpp -O3 -I/usr/include/eigen3 -I /usr/include/boost  -I /usr/include/jsoncpp/ -l jsoncpp
 * ./annService <dimension> <port> <file1> [file2 ... fileN]
 * @endcode
 *
 * Command line arguments:
 * @param dimension The dimensionality of the vectors.
 * @param port The port number on which the HTTP server will listen.
 * @param file1, file2, ..., fileN Paths to the binary files containing the vectors.
 *
 * The HTTP service accepts POST requests with a JSON body containing the query vector and optional parameters.
 * The response is a JSON object containing the indices and similarity scores of the top N most similar vectors.
 *
 * Example JSON request:
 * @code
 * {
 *   "vector_index": 0,
 *   "top_n": 5,
 *   "vector": [0.1, 0.2, 0.3, ..., 0.9]
 * }
 * @endcode
 *
 * Example JSON response:
 * @code
 * {
 *   "results": [
 *     {"index": 1, "similarity": 0.95},
 *     {"index": 3, "similarity": 0.93},
 *     ...
 *   ]
 * }
 * @endcode
 *
 * Dependencies:
 * - Eigen library for matrix and vector operations.
 * - Boost libraries for HTTP server and JSON handling.
 *
 * @note Ensure that the binary files are correctly formatted and the dimensions match the specified dimension.
 * @note The program uses parallel processing to speed up the search, so it may utilize multiple CPU cores.
 *
 * @see Eigen library: https://eigen.tuxfamily.org/
 * @see Boost library: https://www.boost.org/
 */
#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <future>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <json/json.h>
// #include <boost/json.hpp>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;

using tcp = boost::asio::ip::tcp;

#include <sys/stat.h>

// g++ -o annService annService.cpp -O3 -I/usr/include/eigen3 -I /usr/include/boost  -I /usr/include/jsoncpp/ -l jsoncpp


// Type aliases
using Vector = Eigen::VectorXf;
using Matrix = Eigen::MatrixXf;

// Load vectors from a binary file
/**
 * @brief Loads vectors from a binary file and normalizes them.
 *
 * This function reads a specified number of vectors from a binary file and
 * normalizes each vector. The vectors are stored in a matrix where each row
 * represents a vector.
 *
 * @param filename The path to the binary file containing the vectors.
 * @param num_vectors The number of vectors to read from the file.
 * @param dim The dimensionality of each vector.
 * @return A matrix containing the loaded and normalized vectors.
 * @throws std::runtime_error If the file cannot be opened or the specified
 *         number of vectors cannot be read.
 * 
 * Eigen, by default, stores matrices in Column-Major layout — but your file contains row-major data (i.e., each embedding vector is stored sequentially, as written in Python using .tofile() from a (n, dim) array).
    So, when you load that binary file directly into a default Eigen matrix, you're interpreting the data with the wrong stride, resulting in scrambled values.
    ✅ Fix: Force RowMajor layout in Eigen
    Change your matrix definition to:
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> embeddings(num_vectors, dim);
    That ensures that the memory layout in C++ matches what Python wrote:
    Python NumPy .tofile() → row-major binary
 * 
 */
Matrix load_vectors_from_file(const std::string &filename, int num_vectors, int dim)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Error: Cannot open file " + filename);
    }

    // Matrix embeddings(num_vectors, dim);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> embeddings(num_vectors, dim);

    file.read(reinterpret_cast<char *>(embeddings.data()), num_vectors * dim * sizeof(float));

    if (!file)
    {
        throw std::runtime_error("Error: Unable to read the specified number of vectors.");
    }

    // Print the first row as floats
    /*
    std::cout << "First row: ";
    for (int j = 0; j < dim; ++j)
    {
        std::cout << embeddings(0, j) << " ";
    }
    std::cout << std::endl;
    */
    // Normalize each vector
    for (int i = 0; i < num_vectors; ++i)
    {
        embeddings.row(i) = embeddings.row(i).normalized();  // important
    }

    return embeddings;
}

// Parallel brute-force top N search
/**
 * @brief Performs a parallel brute force search to find the top N most similar vectors to a query vector.
 *
 * This function divides the search task among multiple threads to speed up the computation. Each thread processes
 * a chunk of the embeddings matrix, computes the similarity of each vector to the query vector, and keeps track
 * of the top N most similar vectors in its chunk. The results from all threads are then combined and the global
 * top N most similar vectors are determined.
 *
 * @param query The query vector for which we are finding the most similar vectors.
 * @param embeddings The matrix of vectors to search through.
 * @param top_n The number of top similar vectors to return.
 * @param num_threads The number of threads to use for parallel processing.
 * @return A vector of pairs, where each pair contains the index of a vector in the embeddings matrix and its similarity score to the query vector.
 */
std::vector<std::pair<int, float>> parallel_brute_force_search(const Vector &query, const Matrix &embeddings, int top_n, int num_threads)
{
    size_t num_vectors = embeddings.rows();
    size_t chunk_size = num_vectors / num_threads;

    // Results for each thread
    std::vector<std::future<std::vector<std::pair<int, float>>>> futures;

    // Launch threads
    for (int t = 0; t < num_threads; ++t)
    {
        size_t start_idx = t * chunk_size;
        size_t end_idx = (t == num_threads - 1) ? num_vectors : start_idx + chunk_size;

        futures.push_back(std::async(std::launch::async, [&, start_idx, end_idx]()
                                     {
            std::vector<std::pair<int, float>> local_results;

            for (size_t i = start_idx; i < end_idx; ++i) {
                Eigen::VectorXf embedding_i = embeddings.row(i); //.transpose();
                float similarity = query.dot(embedding_i);
                // float similarity = query.dot(embeddings.row(i));
                local_results.emplace_back(i, similarity);
            }

            // Partial sort to get top N results for this thread
            std::partial_sort(local_results.begin(), local_results.begin() + std::min(top_n, static_cast<int>(local_results.size())), local_results.end(),
                              [](const auto& a, const auto& b) { return a.second > b.second; });
            local_results.resize(std::min(top_n, static_cast<int>(local_results.size())));
            return local_results; }));
    }

    // Gather results from all threads
    std::vector<std::pair<int, float>> all_results;
    for (auto &future : futures)
    {
        auto thread_results = future.get();
        all_results.insert(all_results.end(), thread_results.begin(), thread_results.end());
    }

    // Global top N sorting
    // Sort all results to get the top N highest values
    std::sort(all_results.begin(), all_results.end(),
              [](const auto &a, const auto &b)
              { return a.second > b.second; });
    all_results.resize(std::min(top_n, static_cast<int>(all_results.size())));

    return all_results;
}

// Function to get the file size in bytes
/**
 * @brief Get the size of a file.
 *
 * This function retrieves the size of the specified file in bytes.
 *
 * @param filename The name of the file whose size is to be determined.
 * @return size_t The size of the file in bytes.
 * @throws std::runtime_error If the file size cannot be determined.
 */
size_t get_file_size(const std::string &filename)
{
    struct stat stat_buf;
    if (stat(filename.c_str(), &stat_buf) != 0)
    {
        throw std::runtime_error("Error: Cannot get file size for " + filename);
    }
    return stat_buf.st_size;
}

// Function to calculate the number of vectors based on file size and dimensions
/**
 * @brief Calculates the number of vectors in a file based on its size and the dimension of each vector.
 *
 * This function reads the size of the file specified by the filename and calculates the number of vectors
 * it contains, given that each vector has a specified dimension. It assumes that each dimension is represented
 * by a 32-bit float (4 bytes). If the file size is not a multiple of the vector size, it throws a runtime error.
 *
 * @param filename The path to the file containing the vectors.
 * @param dim The dimension of each vector (number of float values per vector).
 * @return The number of vectors in the file.
 * @throws std::runtime_error If the file size is not a multiple of the vector size.
 */
size_t calculate_num_vectors(const std::string &filename, int dim)
{
    size_t file_size = get_file_size(filename);
    size_t bytes_per_vector = dim * sizeof(float); // float32 is 4 bytes
    if (file_size % bytes_per_vector != 0)
    {
        throw std::runtime_error("Error: File size is not a multiple of the vector size. Check the dimensions.");
    }
    return file_size / bytes_per_vector;
}

/**
 * @brief Handles an HTTP request for performing a vector similarity search.
 *
 * This function processes a JSON request containing a query vector and other optional parameters,
 * performs a brute-force search to find the most similar vectors in the provided embeddings, and
 * returns the results in a JSON response.
 *
 * @param req The HTTP request containing the JSON body with the query parameters.
 * @param res The HTTP response to be populated with the search results or error message.
 * @param all_embeddings A vector of matrices containing the embeddings to search against.
 * @param dim The dimensionality of the vectors in the embeddings.
 *
 * The JSON request body should contain the following keys:
 * - "vector_index" (optional, int): The index of the embeddings set to search in. Defaults to 0.
 * - "top_n" (optional, int): The number of top results to return. Defaults to 5.
 * - "vector" (required, array of floats): The query vector to search for. Must match the dimensionality of the embeddings.
 *
 * The JSON response body will contain:
 * - "results" (array of objects): Each object contains:
 *   - "index" (int): The index of the matching vector in the embeddings.
 *   - "similarity" (float): The similarity score of the matching vector.
 *
 * If an error occurs, the response will contain:
 * - "error" (string): A description of the error.
 *
 * @throws std::runtime_error if the query vector size does not match the dimensions of the dataset,
 *                            if the "vector" key is missing or invalid, or if the vector_index is out of range.
 */
void handle_request(http::request<http::string_body> &req, http::response<http::string_body> &res,
                    const std::vector<Matrix> &all_embeddings, int dim, std::vector<std::string> collections)
{
    try
    {
        if (req.method() == http::verb::get)
        {
            // Handle GET request: return list of files
            Json::Value response_body(Json::arrayValue);
            for (const auto &collection : collections)
            {
                response_body.append(collection);
            }

            Json::StreamWriterBuilder writer;
            std::ostringstream oss;
            std::unique_ptr<Json::StreamWriter> json_writer(writer.newStreamWriter());
            json_writer->write(response_body, &oss);

            res.result(http::status::ok);
            res.set(http::field::content_type, "application/json");
            res.body() = oss.str();
            res.prepare_payload();
        }
        else if (req.method() == http::verb::post)
        {
            // Handle POST request: Perform search
            // Parse JSON request
            Json::Value body;
            std::istringstream req_body_stream(req.body());
            req_body_stream >> body;

            // Check for "collection" index and provide a default value
            int vector_index = body.isMember("collection") ? body["collection"].asInt() : 0;

            // Check for "top_n" and provide a default value
            int top_n = body.isMember("limit") ? body["limit"].asInt() : 5;

            // Check for "vector" and validate
            if (!body.isMember("vectors") || !body["vectors"].isArray())
            {
                throw std::runtime_error("Missing or invalid 'vectors' key.");
            }

            Json::Value query_vector_json = body["vectors"];
            if (query_vector_json.size() == 1 && query_vector_json[0].size() == dim)
            {
                query_vector_json = query_vector_json[0];
            }
            else if (query_vector_json.size() != dim)
            {
                throw std::runtime_error("Too many vectors or query vector size does not match the dimensions of the dataset.");
            }

            std::vector<float> query_vector(dim, 0.0f);
            for (size_t i = 0; i < query_vector_json.size(); ++i)
            {
                query_vector[i] = query_vector_json[static_cast<Json::ArrayIndex>(i)].asFloat();
            }

            // Validate vector_index
            if (vector_index < 0 || vector_index >= all_embeddings.size())
            {
                throw std::runtime_error("Invalid vector_index: Out of range.");
            }

            // Retrieve the specified embeddings
            const Matrix &embeddings = all_embeddings[vector_index];

            // Normalize the query vector
            Vector query = Eigen::Map<Vector>(query_vector.data(), dim).normalized();

            // Perform brute-force search
            auto results = parallel_brute_force_search(query, embeddings, top_n, 8);

            // like{'code': 0, 'cost': 6, 'data': [{'distance': -0.01647208, 'id': 'A_4_1_chunk_0'}, {'distance': -0.019615145, 'id': 'A_2_3_chunk_0'}, {'distance': -0.026594205, 'id': 'A_2_4_chunk_0'}]}


            // Prepare JSON response
            Json::Value response;
            response["data"] = Json::arrayValue;
            for (const auto &result : results)
            {
                Json::Value obj;
                obj["id"] = result.first;
                obj["similarity"] = result.second;
                response["data"].append(obj);
            }

            // Set HTTP response
            Json::Value response_body;
            response_body = response;

            Json::StreamWriterBuilder writer;
            std::ostringstream oss;
            std::unique_ptr<Json::StreamWriter> json_writer(writer.newStreamWriter());
            json_writer->write(response_body, &oss);

            res.result(http::status::ok);
            res.set(http::field::content_type, "application/json");
            res.body() = oss.str();
            res.prepare_payload();
        }
        else
        {
            res.result(http::status::method_not_allowed);
            res.set(http::field::content_type, "text/plain");
            res.body() = "Method Not Allowed";
            res.prepare_payload();
        }
    }
    catch (const std::exception &ex)
    {
        Json::Value error_response;
        error_response["error"] = ex.what();
        std::cout << "Error: " << ex.what() << "\n";

        Json::StreamWriterBuilder writer;
        std::ostringstream oss;
        std::unique_ptr<Json::StreamWriter> json_writer(writer.newStreamWriter());
        json_writer->write(error_response, &oss);

        res.result(http::status::bad_request);
        res.set(http::field::content_type, "application/json");
        res.body() = oss.str();
        res.prepare_payload();
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <dimension> <port> <file1> [file2 ... fileN]\n";
        return 1;
    }

    int dim = std::stoi(argv[1]);
    int port = std::stoi(argv[2]);
    std::vector<std::string> collections;

    try
    {
        // Load multiple vector files
        std::vector<Matrix> all_embeddings;
        for (int i = 3; i < argc; ++i)
        {
            std::string filename = argv[i];
            // Collect file names
            std::string collection = filename.substr(filename.find_last_of("/") + 1);
            collection = collection.substr(0, collection.find(".vec"));
            collections.push_back(collection);
            int num_vectors = calculate_num_vectors(filename, dim);
            std::cout << "Loading file: " << filename << "\n";
            all_embeddings.push_back(load_vectors_from_file(filename, num_vectors, dim));
        }

        std::cout << "Loaded " << all_embeddings.size() << " vector files.\n";

        // Start HTTP server
        net::io_context ioc;
        tcp::acceptor acceptor(ioc, tcp::endpoint(tcp::v4(), port));
        std::cout << "Server running on port " << port << "\n";

        while (true)
        {
            tcp::socket socket(ioc);
            acceptor.accept(socket);

            http::request<http::string_body> req;
            beast::flat_buffer buffer;
            http::read(socket, buffer, req);

            http::response<http::string_body> res;
            handle_request(req, res, all_embeddings, dim, collections);

            http::write(socket, res);
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    return 0;
}
