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


namespace cusp
{

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const array1d<T1,Alloc1>& lhs,
                const array1d<T2,Alloc2>& rhs)
{
    return lhs.size() == rhs.size() && thrust::detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}
    
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const array1d<T1,Alloc1>&     lhs,
                const std::vector<T2,Alloc2>& rhs)
{
    return lhs.size() == rhs.size() && thrust::detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const std::vector<T1,Alloc1>& lhs,
                const array1d<T2,Alloc2>&     rhs)
{
    return lhs.size() == rhs.size() && thrust::detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const array1d<T1,Alloc1>& lhs,
                const array1d<T2,Alloc2>& rhs)
{
    return !(lhs == rhs);
}
    
template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const array1d<T1,Alloc1>&     lhs,
                const std::vector<T2,Alloc2>& rhs)
{
    return !(lhs == rhs);
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const std::vector<T1,Alloc1>& lhs,
                const array1d<T2,Alloc2>&     rhs)
{
    return !(lhs == rhs);
}

} // end namespace cusp

