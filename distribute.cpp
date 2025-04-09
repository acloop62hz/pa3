#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <string>
#include <sstream>
#include "functions.h"

void distribute_matrix_2d(int m, int n, std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
                          std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
                          int root, MPI_Comm comm_2d)
{
    // TODO: Write your code here
    int rank, size;
    MPI_Comm_rank(comm_2d, &rank);
    MPI_Comm_size(comm_2d, &size);

    int coords[2];
    
    //always square num of processors
    int dim = static_cast<int>(std::sqrt(size));
    //get coords of the rank
    MPI_Cart_coords(comm_2d, rank, 2, coords);
    //calc total row & col
    int numRow = m/dim;
    int numCol = n/dim;

    int val, pRow, pCol, destRank;
    std::vector<int> sendCounts(size, 0);
    std::vector<int> offset(size, 0);
    std::vector<int> sendBuff;
    if (rank == 0){
        //sort
        std::unordered_map<int, std::vector<int>> pMap;
        for (const auto &element : full_matrix){
            val = element.second;
            pRow = std::min((element.first.first)/numRow,dim-1);
            pCol = std::min((element.first.second)/numCol,dim-1);
            int destCoords[2] = { pRow, pCol };
            // std::cout << pRow << ", " << pCol << std::endl;
            MPI_Cart_rank(comm_2d, destCoords, &destRank); //get dest rank
            pMap[destRank].push_back(element.first.first);
            pMap[destRank].push_back(element.first.second);
            pMap[destRank].push_back(val);
        }
        //prep scatter
        int currOffSet = 0;
        for (int r = 0; r < size; ++r) {
            const auto &data = pMap[r];
            sendCounts[r] = data.size();
            offset[r] = currOffSet;
            sendBuff.insert(sendBuff.end(), data.begin(), data.end());
            currOffSet += data.size();
        }
    }
    //scatter
    int recvcount;
    //MPI ensure only the root scatters
    MPI_Scatter(sendCounts.data(), 1, MPI_INT, &recvcount, 1, MPI_INT, root, comm_2d);

    std::vector<int> recvbuf(recvcount);
    MPI_Scatterv(
        sendBuff.data(), sendCounts.data(), offset.data(), MPI_INT,
        recvbuf.data(), recvcount, MPI_INT, root, comm_2d
    );

        for (size_t i = 0; i < recvbuf.size(); i += 3) {
        int row = recvbuf[i];
        int col = recvbuf[i + 1];
        int val = recvbuf[i + 2];
        local_matrix.emplace_back(std::make_pair(std::make_pair(row, col), val));
    }
    // MPI_Barrier(comm_2d);

    // for (int p = 0; p < size; ++p) {
    // if (rank == p) {
    //     std::cout << "Rank " << rank << " received:\n";
    //     for (const auto &entry : local_matrix) {
    //         int i = entry.first.first;
    //         int j = entry.first.second;
    //         int val = entry.second;
    //         std::cout << "  (" << i << ", " << j << ") = " << val << "\n";
    //     }
    //     std::cout << std::flush;
    // }
    // MPI_Barrier(comm_2d); // ensure ordered printing
    // }
}
