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
#include <cusp/complex.h>
#include <cusp/exception.h>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#ifdef __APPLE__

#include <stdio.h>

#define OUT_FILE_TYPE FILE *

#define CREATE_OUT_FILE(file,filename) FILE* file = fopen(filename, "w+");

#define WRITE_OUT_FILE(file,data) do{ \
    					fprintf(file, "%s", data); \
    					fclose(outfile); \
		  		  }while(0);

#define READ_FILE(stream,filename) do{ \
					FILE * infile = fopen(filename,"r"); \
					fseek(infile,0,SEEK_END); \
					size_t file_size = ftell(infile); \
					fseek(infile,0,SEEK_SET); \
					char * buffer = new char [file_size]; \
					fread(buffer, file_size, sizeof(char), infile); \
					fclose(infile); \
					stream.write(buffer, file_size); \
					delete[] buffer; \
		  		  }while(0);

#else

#include <fstream>

#define OUT_FILE_TYPE std::ostream&

#define CREATE_OUT_FILE(file,filename) std::ofstream file(filename);

#define WRITE_OUT_FILE(file,data) file << data;

#define READ_FILE(stream,filename) do{ \
					std::ifstream infile(filename); \
					stream << infile.rdbuf(); \
					infile.close(); \
		  		}while(0);

#endif

namespace cusp
{
namespace io
{

namespace detail
{

inline
void tokenize(std::vector<std::string>& tokens,
              const std::string& str,
              const std::string& delimiters = "\n\r\t ")
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

template<typename I, typename M>
void check_matrix_market_type(const cusp::coo_matrix<I,float,M>& coo, const std::string& type)
{
    if( type=="complex" )
        throw cusp::io_exception("complex-valued matrices incompatible with 'float' containers");
    // else: integer, real, and pattern are all allowed
}

template<typename I, typename M>
void check_matrix_market_type(const cusp::coo_matrix<I,double,M>& coo, const std::string& type)
{
    if( type=="complex" )
        throw cusp::io_exception("complex-valued matrices incompatible with 'double' containers");
    // else: integer, real, and pattern are all allowed
}

template<typename I, typename M>
void check_matrix_market_type(const cusp::coo_matrix<I,cusp::complex<float>,M>& coo, const std::string& type)
{
    // complex containers can hold for real and complex matrices
}

template<typename I, typename M>
void check_matrix_market_type(const cusp::coo_matrix<I,cusp::complex<double>,M>& coo, const std::string& type)
{
    // complex containers can hold for real and complex matrices
}

// TODO: Add ValueType=complex case here when complex data is supported

template<typename I, typename V, typename M>
void check_matrix_market_type(const cusp::coo_matrix<I,V,M>& coo, const std::string& type)
{
    if( type=="real" )
        throw cusp::io_exception("real-valued matrices require a container with 'float' or 'double' values");
}

} // end namespace detail




struct matrix_market_banner
{
    std::string storage;    // "array" or "coordinate"
    std::string symmetry;   // "general", "symmetric", "hermitian", or "skew-symmetric" 
    std::string type;       // "complex", "real", "integer", or "pattern"
};

inline void read_matrix_market_banner(matrix_market_banner& banner, std::istream& file)
{
    std::string line;
    std::vector<std::string> tokens;

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

inline void read_matrix_market_banner(matrix_market_banner& banner, const std::string& filename)
{
    std::stringstream file (std::stringstream::in | std::stringstream::out);

    READ_FILE(file,filename.c_str())

    if (!file.good())
        throw cusp::io_exception(std::string("invalid file: [") + filename + std::string("]"));

    read_matrix_market_banner(banner, file);
}

template <typename IndexType, typename ValueType>
void read_matrix_market_file(cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo, const std::string& filename)
{
    std::stringstream file (std::stringstream::in | std::stringstream::out);

    READ_FILE(file,filename.c_str())

    if (!file.good())
        throw cusp::io_exception(std::string("invalid file name: [") + filename + std::string("]"));

    read_matrix_market_stream(coo, file);
}



template <typename IndexType, typename ValueType>
struct if_type_is_complex{
  static void read_array(size_t & num_entries_read, const size_t num_entries, std::istream & file, 
			 cusp::array2d<ValueType,cusp::host_memory,cusp::column_major> & dense){
    throw cusp::not_implemented_exception("Cannot read complex MatrixMarket data type"
					  " without using a complex container");
  }
  static void read_coordinate(size_t & num_entries_read, const int num_entries, std::istream & file, 
			      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> & coo){
    throw cusp::not_implemented_exception("Cannot read complex MatrixMarket data type"
					  " without using a complex container");
  }
};

template <typename IndexType, typename ValueType>
struct if_type_is_complex<IndexType,cusp::complex<ValueType> >{
  static void read_array(size_t & num_entries_read, const size_t num_entries, std::istream & file, 
			 cusp::array2d<cusp::complex<ValueType>,cusp::host_memory,cusp::column_major> & dense){
    while(num_entries_read < num_entries && !file.eof())
      {
	ValueType v;
	file >> v;
	dense.values[num_entries_read].real(v);
	file >> v;
	dense.values[num_entries_read].imag(v);
	num_entries_read++;
      }
  }
  static void read_coordinate(size_t & num_entries_read, const size_t num_entries, std::istream & file, 
			      cusp::coo_matrix<IndexType,cusp::complex<ValueType>,cusp::host_memory> & coo){
    while(num_entries_read < coo.num_entries && !file.eof())
      {
	file >> coo.row_indices[num_entries_read];
	file >> coo.column_indices[num_entries_read];
	ValueType v;
	file >> v;
	coo.values[num_entries_read].real(v);
	file >> v;
	coo.values[num_entries_read].imag(v);
	num_entries_read++;
      }
  }
};

template <typename IndexType, typename ValueType>
void read_matrix_market_stream(cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo, std::istream& file)
{
    // read banner
    matrix_market_banner banner;
    read_matrix_market_banner(banner, file);

    // check for type mismatch with container
    detail::check_matrix_market_type(coo, banner.type);
   
    // read file contents line by line
    std::string line;
    
    // skip over banner and comments
    do
    {
        std::getline(file, line);
    } while (line[0] == '%');

    if (banner.storage == "array")
    {
        // dense format
        std::vector<std::string> tokens;
        detail::tokenize(tokens, line); 

        if (tokens.size() != 2)
            throw cusp::io_exception("invalid MatrixMarket array format");

        size_t num_rows, num_cols;

        std::istringstream(tokens[0]) >> num_rows;
        std::istringstream(tokens[1]) >> num_cols;
  
        cusp::array2d<ValueType,cusp::host_memory,cusp::column_major> dense(num_rows, num_cols);

        size_t num_entries = num_rows * num_cols;

        size_t num_entries_read = 0;
            
        // read file contents
        if (banner.type == "pattern")
        {
            throw cusp::not_implemented_exception("pattern array MatrixMarket format is not supported");
        } 
        else if (banner.type == "real" || banner.type == "integer")
        {
            while(num_entries_read < num_entries && !file.eof())
            {
                file >> dense.values[num_entries_read];
                num_entries_read++;
            }
        } 
        else if (banner.type == "complex")
        {
          if_type_is_complex<IndexType,ValueType>::read_array(num_entries_read,num_entries,file,dense); 
        }
        else
        {
            throw cusp::io_exception("invalid MatrixMarket data type");
        }

        if(num_entries_read != num_entries)
            throw cusp::io_exception("unexpected EOF while reading MatrixMarket entries");
     
        if (banner.symmetry != "general")
            throw cusp::not_implemented_exception("only general array symmetric MatrixMarket format is supported");

        // convert to coo
        coo = dense;
    }
    else if (banner.storage == "coordinate")
    {
        // line contains [num_rows num_columns num_entries]
        std::vector<std::string> tokens;
        detail::tokenize(tokens, line); 

        if (tokens.size() != 3)
            throw cusp::io_exception("invalid MatrixMarket coordinate format");

        size_t num_rows, num_cols, num_entries;

        std::istringstream(tokens[0]) >> num_rows;
        std::istringstream(tokens[1]) >> num_cols;
        std::istringstream(tokens[2]) >> num_entries;
  
        coo.resize(num_rows, num_cols, num_entries);

        size_t num_entries_read = 0;

        // read file contents
        if (banner.type == "pattern")
        {
            while(num_entries_read < coo.num_entries && !file.eof())
            {
                file >> coo.row_indices[num_entries_read];
                file >> coo.column_indices[num_entries_read];
                num_entries_read++;
            }

            std::fill(coo.values.begin(), coo.values.end(), ValueType(1));
        } 
        else if (banner.type == "real" || banner.type == "integer")
        {
            while(num_entries_read < coo.num_entries && !file.eof())
            {
                file >> coo.row_indices[num_entries_read];
                file >> coo.column_indices[num_entries_read];
                file >> coo.values[num_entries_read];
                num_entries_read++;
            }
        } 
        else if (banner.type == "complex")
        {
          if_type_is_complex<IndexType,ValueType>::read_coordinate(num_entries_read,num_entries,file,coo); 
        }
        else
        {
            throw cusp::io_exception("invalid MatrixMarket data type");
        }

        if(num_entries_read != coo.num_entries)
            throw cusp::io_exception("unexpected EOF while reading MatrixMarket entries");


        // check validity of row and column index data
        size_t min_row_index = *std::min_element(coo.row_indices.begin(), coo.row_indices.end());
        size_t max_row_index = *std::max_element(coo.row_indices.begin(), coo.row_indices.end());
        size_t min_col_index = *std::min_element(coo.column_indices.begin(), coo.column_indices.end());
        size_t max_col_index = *std::max_element(coo.column_indices.begin(), coo.column_indices.end());

        if (min_row_index < 1)            throw cusp::io_exception("found invalid row index (index < 1)");
        if (min_col_index < 1)            throw cusp::io_exception("found invalid column index (index < 1)");
        if (max_row_index > coo.num_rows) throw cusp::io_exception("found invalid row index (index > num_rows)");
        if (max_col_index > coo.num_cols) throw cusp::io_exception("found invalid column index (index > num_columns)");


        // convert base-1 indices to base-0
        for(size_t n = 0; n < coo.num_entries; n++)
        {
            coo.row_indices[n]    -= 1;
            coo.column_indices[n] -= 1;
        }
        

        // expand symmetric formats to "general" format
        if (banner.symmetry != "general")
        {
            size_t off_diagonals = 0;

            for (size_t n = 0; n < coo.num_entries; n++)
                if(coo.row_indices[n] != coo.column_indices[n])
                    off_diagonals++;

            size_t general_num_entries = coo.num_entries + off_diagonals;
            
            cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> general(num_rows, num_cols, general_num_entries);
           
            if (banner.symmetry == "symmetric")
            {
                size_t nnz = 0;

                for (size_t n = 0; n < coo.num_entries; n++)
                {
                    // copy entry over
                    general.row_indices[nnz]    = coo.row_indices[n];
                    general.column_indices[nnz] = coo.column_indices[n];
                    general.values[nnz]         = coo.values[n];
                    nnz++;

                    // duplicate off-diagonals
                    if (coo.row_indices[n] != coo.column_indices[n])
                    {
                        general.row_indices[nnz]    = coo.column_indices[n];
                        general.column_indices[nnz] = coo.row_indices[n];
                        general.values[nnz]         = coo.values[n];
                        nnz++;
                    } 
                }       
            } 
            else if (banner.symmetry == "hermitian")
            {
                throw cusp::not_implemented_exception("MatrixMarket I/O does not currently support hermitian matrices");
                //TODO
            } 
            else if (banner.symmetry == "skew-symmetric")
            {
                //TODO
                throw cusp::not_implemented_exception("MatrixMarket I/O does not currently support skew-symmetric matrices");
            }

            // store full matrix in coo
            coo.swap(general);
        } // if (banner.symmetry != "general")
    
        // sort indices by (row,column)
        coo.sort_by_row_and_column();
    } 
    else 
    {
        // should never happen
        throw cusp::io_exception("invalid MatrixMarket storage format [" + banner.storage + "]");
    }
}

template <typename IndexType, typename ValueType>
void write_matrix_market_stream(const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo, OUT_FILE_TYPE outfile)
{
    std::stringstream file (std::stringstream::in | std::stringstream::out);
    if(thrust::detail::is_same<ValueType,
			       cusp::complex<typename norm_type<ValueType>::type> 
			       >::value){
      file << "%%MatrixMarket matrix coordinate complex general\n";
    }else{
      file << "%%MatrixMarket matrix coordinate real general\n";
    }
    file << "\t" << coo.num_rows << "\t" << coo.num_cols << "\t" << coo.num_entries << "\n";

    for(size_t i = 0; i < coo.num_entries; i++)
    {
        file << (coo.row_indices[i]    + 1) << " ";
        file << (coo.column_indices[i] + 1) << " ";
        file <<  coo.values[i]              << "\n";
    }

    std::string file_data(file.str());
    WRITE_OUT_FILE(outfile, file_data.c_str())
}

template <typename IndexType, typename ValueType>
void write_matrix_market_file(const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo, const std::string& filename)
{
    // read file contents line by line
    CREATE_OUT_FILE(file,filename.c_str())

    if (!file)
        throw cusp::io_exception(std::string("unable to open file name: [") + filename + std::string("] for writing"));

    write_matrix_market_stream(coo, file);
}
    
template <typename MatrixType>
void read_matrix_market_file(MatrixType& mtx, const std::string& filename)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo;
    cusp::io::read_matrix_market_file(coo, filename);
    mtx = coo;
}

template <typename ValueType, typename MemorySpace>
void read_matrix_market_file(cusp::array1d<ValueType,MemorySpace> & vector, const std::string& filename)
{
    cusp::coo_matrix<int,ValueType,cusp::host_memory> coo;
    cusp::io::read_matrix_market_file(coo, filename);
    if(coo.num_cols != 1){
      throw cusp::io_exception(std::string("cannot read matrix with more than 1 column into an array1d while reading: [") + filename);
    }
    cusp::array1d<ValueType,cusp::host_memory> h_vector(coo.num_rows,0);
    for(size_t i = 0;i<coo.num_entries;i++){
      h_vector[coo.row_indices[i]] = coo.values[i];
    }
    vector = h_vector;
}

template <typename MatrixType>
void read_matrix_market_stream(MatrixType& mtx, std::istream& in)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo;
    cusp::io::read_matrix_market_stream(coo, in);
    mtx = coo;
}

template <typename ValueType, typename MemorySpace>
void read_matrix_market_stream(cusp::array1d<ValueType,MemorySpace> & vector, std::istream& in)
{
    cusp::coo_matrix<int,ValueType,cusp::host_memory> coo;
    cusp::io::read_matrix_market_stream(coo, in);
    if(coo.num_cols != 1){
      throw cusp::io_exception(std::string("cannot read matrix with more than 1 column into an array1d while reading stream"));
    }
    cusp::array1d<ValueType,cusp::host_memory> h_vector(coo.num_rows,0);
    for(size_t i = 0;i<coo.num_entries;i++){
      h_vector[coo.row_indices[i]] = coo.values[i];
    }
    vector = h_vector;
}

template <typename MatrixType>
void write_matrix_market_file(const MatrixType& mtx, const std::string& filename)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo;
    coo = mtx;
    cusp::io::write_matrix_market_file(coo, filename);
}

template <typename ValueType, typename MemorySpace>
void write_matrix_market_file(const cusp::array1d<ValueType,MemorySpace>& vector, const std::string& filename)
{
  cusp::array1d<ValueType,cusp::host_memory> h_vector(vector);
  cusp::coo_matrix<int,ValueType,cusp::host_memory> coo(vector.size(),1,vector.size());
  for(size_t i = 0;i<coo.num_entries;i++){
    coo.row_indices[i] = i;
    coo.column_indices[i] = 0;
    coo.values[i] = h_vector[i];
  }
  cusp::io::write_matrix_market_file(coo, filename);
}

template <typename MatrixType>
void write_matrix_market_stream(const MatrixType& mtx, OUT_FILE_TYPE out)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo;
    coo = mtx;
    cusp::io::write_matrix_market_stream(coo, out);
}

template <typename ValueType, typename MemorySpace>
void write_matrix_market_stream(const cusp::array1d<ValueType,MemorySpace>& vector, OUT_FILE_TYPE out)
{
  cusp::array1d<ValueType,cusp::host_memory> h_vector(vector);
  cusp::coo_matrix<int,ValueType,cusp::host_memory> coo(vector.size(),1,vector.size());
  for(size_t i = 0;i<coo.num_entries;i++){
    coo.row_indices[i] = i;
    coo.column_indices[i] = 0;
    coo.values[i] = h_vector[i];
  }
  cusp::io::write_matrix_market_stream(coo, out);
}

} //end namespace io
} //end namespace cusp

