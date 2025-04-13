#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <string>
#include <sstream>
#include <map>
#include "functions.h"

void distribute_matrix_2d(int m, int n,
    std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
    std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
    int root, MPI_Comm comm_2d)
{
    int rank, p;
    MPI_Comm_rank(comm_2d, &rank);
    MPI_Comm_size(comm_2d, &p);
    local_matrix.clear();

    if (rank == root) {
        int q = static_cast<int>(sqrt(p));
        int block_rows = m / q;
        int remainder_rows = m % q;
        int block_cols = n / q;
        int remainder_cols = n % q;

        std::map<int, std::vector<std::pair<std::pair<int, int>, int>>> elements_map;

        for (const auto& elem : full_matrix) {
            int i = elem.first.first;
            int j = elem.first.second;

            // Calculate pi
            int pi;
            if (i < remainder_rows * (block_rows + 1)) {
                pi = i / (block_rows + 1);
            } else {
                int adjusted_i = i - remainder_rows * (block_rows + 1);
                pi = remainder_rows + adjusted_i / block_rows;
            }

            // Calculate pj
            int pj;
            if (j < remainder_cols * (block_cols + 1)) {
                pj = j / (block_cols + 1);
            } else {
                int adjusted_j = j - remainder_cols * (block_cols + 1);
                pj = remainder_cols + adjusted_j / block_cols;
            }

            // Determine destination rank
            int coords[2] = {pi, pj};
            int dest_rank;
            MPI_Cart_rank(comm_2d, coords, &dest_rank);
            elements_map[dest_rank].push_back(elem);
        }

        // Distribute the elements
        for (int dest = 0; dest < p; ++dest) {
            auto it = elements_map.find(dest);
            int count = (it != elements_map.end()) ? it->second.size() : 0;

            if (dest == root) {
                if (count > 0) {
                    local_matrix.insert(local_matrix.end(), it->second.begin(), it->second.end());
                }
            } else {
                MPI_Send(&count, 1, MPI_INT, dest, 0, comm_2d);
                if (count > 0) {
                    std::vector<int> buffer;
                    buffer.reserve(3 * count);
                    for (const auto& elem : it->second) {
                        buffer.push_back(elem.first.first);
                        buffer.push_back(elem.first.second);
                        buffer.push_back(elem.second);
                    }
                    MPI_Send(buffer.data(), 3 * count, MPI_INT, dest, 1, comm_2d);
                }
            }
        }
    } else {
        int count;
        MPI_Recv(&count, 1, MPI_INT, root, 0, comm_2d, MPI_STATUS_IGNORE);
        if (count > 0) {
            std::vector<int> buffer(3 * count);
            MPI_Recv(buffer.data(), 3 * count, MPI_INT, root, 1, comm_2d, MPI_STATUS_IGNORE);
            for (int k = 0; k < count; ++k) {
                int i = buffer[3 * k];
                int j = buffer[3 * k + 1];
                int val = buffer[3 * k + 2];
                local_matrix.emplace_back(std::make_pair(i, j), val);
            }
        }
    }
}