/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/convert.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace cusp
{

namespace detail
{

inline
void tokenize(std::vector<std::string>& tokens,
               const std::string& str,
               const std::string& delimiters = " ")
{
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

} // end namespace detail


    class io_exception : public std::exception
    {
        public:
            io_exception(const std::string _msg) : msg(_msg) {}
            ~io_exception() throw() {}
            const char* what() const throw() { return msg.c_str(); }

        private:
            std::string msg;
    };



struct matrix_market_banner
{
    std::string storage;    // "array" or "coordinate"
    std::string symmetry;   // "general", "symmetric", "hermitian", or "skew-symmetric" 
    std::string type;       // "complex", "real", "integer", or "pattern"
};
 

inline void read_matrix_market_banner(matrix_market_banner& banner, const std::string& filename)
{
    std::ifstream file(filename.c_str());
    std::string line;
    std::vector<std::string> tokens;

    if (!file)
        throw cusp::io_exception(std::string("invalid file: [") + filename + std::string("]"));

    // read first line
    std::getline(file, line);
    detail::tokenize(tokens, line); 

    if (tokens.size() != 5 || tokens[0] != "%%MatrixMarket" || tokens[1] != "matrix")
        throw cusp::io_exception("invalid MatrixMarket banner");

    banner.storage  = tokens[2];
    banner.type     = tokens[3];
    banner.symmetry = tokens[4];

    if (banner.storage != "array" && banner.storage != "coordinate")
        throw cusp::io_exception("invalid MatrixMarket storage format [" + banner.storage + "]");
    
    if (banner.type != "complex" && banner.type != "real" 
            && banner.type != "integer" && banner.type != "pattern")
        throw cusp::io_exception("invalid MatrixMarket data type [" + banner.type + "]");

    if (banner.symmetry != "general" && banner.symmetry != "symmetric" 
            && banner.symmetry != "hermitian" && banner.symmetry != "skew-symmetric")
        throw cusp::io_exception("invalid MatrixMarket symmetry [" + banner.symmetry + "]");
}

template <typename IndexType, typename ValueType>
void load_matrix_market_file(cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo, const std::string& filename)
{
    // read banner
    matrix_market_banner banner;
    read_matrix_market_banner(banner, filename);
   
    // read file contents line by line
    std::ifstream file(filename.c_str());
    std::string line;

    if (!file)
        throw cusp::io_exception(std::string("invalid file name: [") + filename + std::string("]"));
    
    // skip over banner and comments
    do {
        std::getline(file, line);
    } while (line[0] == '%');

    if (banner.storage == "coordinate"){
        // line contains [num_rows num_columns num_entries]
        std::vector<std::string> tokens;
        detail::tokenize(tokens, line); 

        if (tokens.size() != 3)
            throw cusp::io_exception("invalid MatrixMarket coordinate format");

        IndexType num_rows, num_cols, num_entries;

        std::istringstream(tokens[0]) >> num_rows;
        std::istringstream(tokens[1]) >> num_cols;
        std::istringstream(tokens[2]) >> num_entries;
   
        cusp::allocate_matrix(coo, num_rows, num_cols, num_entries);

        IndexType num_entries_read = 0;

        // read file contents
        if (banner.type == "pattern"){
            while(num_entries_read < coo.num_entries && !file.eof()){
                file >> coo.row_indices[num_entries_read];
                file >> coo.column_indices[num_entries_read];
                num_entries_read++;
            }
            std::fill(coo.values, coo.values + coo.num_entries, 1.0);
        } else if (banner.type == "real" || banner.type == "integer") {
            while(num_entries_read < coo.num_entries && !file.eof()){
                file >> coo.row_indices[num_entries_read];
                file >> coo.column_indices[num_entries_read];
                file >> coo.values[num_entries_read];
                num_entries_read++;
            }
        } else if (banner.type == "complex") {
            throw cusp::io_exception("complex MatrixMarket data type is not supported");
        } else {
            throw cusp::io_exception("invalid MatrixMarket data type");
        }

        if(num_entries_read != coo.num_entries)
            throw cusp::io_exception("unexpected EOF while reading MatrixMarket entries");


        // convert base-1 indices to base-0
        for(IndexType n = 0; n < coo.num_entries; n++){
            coo.row_indices[n]    -= 1;
            coo.column_indices[n] -= 1;
        }
        

        // expand symmetric formats to "general" format
        if (banner.symmetry != "general"){

            IndexType off_diagonals = 0;
            for (IndexType n = 0; n < coo.num_entries; n++){
                if(coo.row_indices[n] != coo.column_indices[n])
                    off_diagonals++;
            }

            IndexType general_num_entries = coo.num_entries + off_diagonals;
            
            cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> general;
            cusp::allocate_matrix(general, num_rows, num_cols, general_num_entries);
           
            if (banner.symmetry == "symmetric"){
                IndexType nnz = 0;
                for (IndexType n = 0; n < coo.num_entries; n++){
                    // copy entry over
                    general.row_indices[nnz]    = coo.row_indices[n];
                    general.column_indices[nnz] = coo.column_indices[n];
                    general.values[nnz]         = coo.values[n];
                    nnz++;

                    // duplicate off-diagonals
                    if (coo.row_indices[n] != coo.column_indices[n]){
                        general.row_indices[nnz]    = coo.column_indices[n];
                        general.column_indices[nnz] = coo.row_indices[n];
                        general.values[nnz]         = coo.values[n];
                        nnz++;
                    } 
                }       
            } else if (banner.symmetry == "hermitian"){
                throw cusp::io_exception("MatrixMarket I/O does not currently support hermitian matrices");
                //TODO
            } else if (banner.symmetry == "skew-symmetric"){
                //TODO
                throw cusp::io_exception("MatrixMarket I/O does not currently support skew-symmetric matrices");
            }

            // XXX replace with cusp::swap() then cusp::deallocate_matrix()
            cusp::deallocate_matrix(coo);
            coo = general;
        }
    } else {
        // dense format
        throw cusp::io_exception("MatrixMarket I/O currently only supports coordinate format");
    }
}

    
template <typename IndexType, typename ValueType>
void load_matrix_market_file(cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& csr, const std::string& filename)
{
    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo;
    cusp::load_matrix_market_file(coo, filename);
    cusp::convert_matrix(csr, coo);
    cusp::deallocate_matrix(coo);
}

// TODO simplify when convert(device, host) works properly
//    Dispatch on MatrixType::memory_space

template <typename IndexType, typename ValueType>
void load_matrix_market_file(cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& coo, const std::string& filename)
{
    // load and convert on host
    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> host_coo;
    load_matrix_market_file(host_coo, filename);

    // transfer to device
    cusp::allocate_matrix(coo, host_coo);
    cusp::memcpy_matrix(coo, host_coo);

    cusp::deallocate_matrix(host_coo);
}

template <typename IndexType, typename ValueType>
void load_matrix_market_file(cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, const std::string& filename)
{
    // load and convert on host
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> host_csr;
    load_matrix_market_file(host_csr, filename);

    // transfer to device
    cusp::allocate_matrix_like(csr, host_csr);
    cusp::memcpy_matrix(csr, host_csr);

    cusp::deallocate_matrix(host_csr);
}


} //end namespace cusp
