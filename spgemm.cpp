#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "functions.h"

void sparse_multiply(
    std::vector<std::pair<std::pair<int,int>, int>> &A,
    std::vector<std::pair<std::pair<int,int>, int>> &B,
    std::map<std::pair<int, int>, int> &C_map,
    std::function<int(int, int)> plus,
    std::function<int(int, int)> times       
){

    for (const auto &a : A){
        int i = a.first.first;
        int k = a.first.second;
        int a_val = a.second;
        for (const auto &b : B){
            if (b.first.first == k){
                int j = b.first.second;
                int b_val = b.second;
                auto key = std::make_pair(i,j);
                int product = times(a_val, b_val);
                if (product == 0) continue; 
                if (C_map.count(key)){
                    C_map[key] = plus(C_map[key], product);
                }else{
                    C_map[key] = product;
                }
            }
        }
    }
}

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    if (A.empty() && B.empty()) {
        return;
    }

    // TODO: Write your code here
    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    int row_size, col_size;
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_size(col_comm, &col_size);

    int dim = row_size;

    std::vector<std::pair<std::pair<int,int>, int>> A_bcast, B_bcast;
    std::map<std::pair<int, int>, int> C_map;

    for(int k = 0; k <dim; ++k){
        // prepare A_bcast
        A_bcast.clear();
        if(row_rank == k){
            A_bcast = A;
        }

        // broadcast A across row
        int a_size = 0;
        if (row_rank == k) a_size = A.size() * 3;
        MPI_Bcast(&a_size, 1, MPI_INT, k, row_comm);
        

        std::vector<int> a_buff(a_size);
        if (row_rank == k) {
            
            for (size_t i = 0; i < A.size(); ++i) {
                a_buff[i * 3 + 0] = A[i].first.first;
                a_buff[i * 3 + 1] = A[i].first.second;
                a_buff[i * 3 + 2] = A[i].second;
            }
        }

        MPI_Bcast(a_buff.data(), a_size, MPI_INT, k, row_comm);

        if (row_rank != k && a_size > 0) {
            A_bcast.resize(a_size / 3);
            for (size_t i = 0; i < A_bcast.size(); ++i) {
                A_bcast[i].first.first = a_buff[i * 3 + 0];
                A_bcast[i].first.second = a_buff[i * 3 + 1];
                A_bcast[i].second = a_buff[i * 3 + 2];
            }
        }

        // Prepare B_bcast
        B_bcast.clear();
        if (col_rank == k) {
            B_bcast = B;
        }
        // Broadcast B across column
        int b_size = 0;
        if (col_rank == k) b_size = B.size() * 3;
        MPI_Bcast(&b_size, 1, MPI_INT, k, col_comm);

        std::vector<int> b_buff(b_size);
        if (col_rank == k) {
            for (size_t i = 0; i < B.size(); ++i) {
                b_buff[i * 3 + 0] = B[i].first.first;
                b_buff[i * 3 + 1] = B[i].first.second;
                b_buff[i * 3 + 2] = B[i].second;
            }
        }
        MPI_Bcast(b_buff.data(), b_size, MPI_INT, k, col_comm);

        if (col_rank != k && b_size > 0) {
            B_bcast.resize(b_size / 3);
            for (size_t i = 0; i < B_bcast.size(); ++i) {
                B_bcast[i].first.first = b_buff[i * 3 + 0];
                B_bcast[i].first.second = b_buff[i * 3 + 1];
                B_bcast[i].second = b_buff[i * 3 + 2];
            }
        }

        if (a_size == 0 || b_size == 0)
            continue;
        // Multiply and accumulate
        sparse_multiply(A_bcast, B_bcast, C_map, plus, times);
        
    }
    // Convert C_map into COO format for output
    for (const auto &entry : C_map) {
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
