#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

void spgemm_2d(int m, int p, int n,
    std::vector<std::pair<std::pair<int,int>, int>>& A,
    std::vector<std::pair<std::pair<int,int>, int>>& B,
    std::vector<std::pair<std::pair<int,int>, int>>& C,
    std::function<int(int,int)> plus,
    std::function<int(int,int)> times,
    MPI_Comm row_comm,
    MPI_Comm col_comm)
{
    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    int Q;
    MPI_Comm_size(row_comm, &Q);

    C.clear();

    std::map<std::pair<int, int>, int> c_map;

    for (int k = 0; k < Q; ++k) {
        // Broadcast A's k-th column block in row_comm
        std::vector<std::pair<std::pair<int, int>, int>> A_part;
        {
            int send_a_count = (row_rank == k) ? static_cast<int>(A.size()) : 0;
            std::vector<int> send_a_data;
            send_a_data.reserve(3 * send_a_count);
            if (row_rank == k) {
                for (const auto &elem : A) {
                    send_a_data.push_back(elem.first.first);
                    send_a_data.push_back(elem.first.second);
                    send_a_data.push_back(elem.second);
                }
            }

            int a_count;
            MPI_Bcast(&send_a_count, 1, MPI_INT, k, row_comm);
            a_count = send_a_count;

            std::vector<int> recv_a_data(3 * a_count);
            if (row_rank == k) {
                std::copy(send_a_data.begin(), send_a_data.end(), recv_a_data.begin());
            }
            MPI_Bcast(recv_a_data.data(), 3 * a_count, MPI_INT, k, row_comm);

            A_part.clear();
            for (int i = 0; i < a_count; ++i) {
                int a_row = recv_a_data[3 * i];
                int a_col = recv_a_data[3 * i + 1];
                int a_val = recv_a_data[3 * i + 2];
                A_part.emplace_back(std::make_pair(a_row, a_col), a_val);
            }
        }

        // Broadcast B's k-th row block in col_comm
        std::vector<std::pair<std::pair<int, int>, int>> B_part;
        {
            int send_b_count = (col_rank == k) ? static_cast<int>(B.size()) : 0;
            std::vector<int> send_b_data;
            send_b_data.reserve(3 * send_b_count);
            if (col_rank == k) {
                for (const auto &elem : B) {
                    send_b_data.push_back(elem.first.first);
                    send_b_data.push_back(elem.first.second);
                    send_b_data.push_back(elem.second);
                }
            }

            int b_count;
            MPI_Bcast(&send_b_count, 1, MPI_INT, k, col_comm);
            b_count = send_b_count;

            std::vector<int> recv_b_data(3 * b_count);
            if (col_rank == k) {
                std::copy(send_b_data.begin(), send_b_data.end(), recv_b_data.begin());
            }
            MPI_Bcast(recv_b_data.data(), 3 * b_count, MPI_INT, k, col_comm);

            B_part.clear();
            for (int i = 0; i < b_count; ++i) {
                int b_row = recv_b_data[3 * i];
                int b_col = recv_b_data[3 * i + 1];
                int b_val = recv_b_data[3 * i + 2];
                B_part.emplace_back(std::make_pair(b_row, b_col), b_val);
            }
        }

        // Multiply A_part and B_part, accumulate into c_map
        std::map<int, std::vector<std::pair<int, int>>> b_map;
        for (const auto &elem : B_part) {
            int brow = elem.first.first;
            int bcol = elem.first.second;
            int bval = elem.second;
            b_map[brow].emplace_back(bcol, bval);
        }

        for (const auto &a_elem : A_part) {
            int arow = a_elem.first.first;
            int acol = a_elem.first.second;
            int aval = a_elem.second;

            auto bit = b_map.find(acol);
            if (bit != b_map.end()) {
                for (const auto &bcol_val : bit->second) {
                    int bc = bcol_val.first;
                    int bv = bcol_val.second;
                    int product = times(aval, bv);
                    auto key = std::make_pair(arow, bc);
                    auto cit = c_map.find(key);
                    if (cit != c_map.end()) {
                        cit->second = plus(cit->second, product);
                    } else {
                        c_map[key] = product;
                    }
                }
            }
        }
    }

    // Convert c_map to C
    C.reserve(c_map.size());
    for (const auto &entry : c_map) {
        C.emplace_back(entry.first, entry.second);
    }

    // for (int p = 0; p < world_size; ++p) {
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (world_rank == p) {
    //         std::cout << "\n===== RowRank " << row_rank << " ColRank " << col_rank << " world " << world_rank << " =====\n";

    //         // Print local A
    //         std::cout << "[A local]\n";
    //         for (const auto &a : A) {
    //             std::cout << "  (" << a.first.first << ", " << a.first.second << ") = " << a.second << "\n";
    //         }

    //         // Print local B
    //         std::cout << "[B local]\n";
    //         for (const auto &b : B) {
    //             std::cout << "  (" << b.first.first << ", " << b.first.second << ") = " << b.second << "\n";
    //         }

    //         // Print final C_map
    //         std::cout << "[C_map result]\n";
    //         for (const auto &entry : C_map) {
    //             int row = entry.first.first;
    //             int col = entry.first.second;
    //             int val = entry.second;
    //             std::cout << "  (" << row << ", " << col << ") = " << val << "\n";
    //         }

    //         std::cout << std::flush;
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD); // Final sync
}
