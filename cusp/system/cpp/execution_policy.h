/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file thrust/system/cpp/execution_policy.h
 *  \brief Execution policies for Thrust's standard C++ system.
 */

#include <thrust/detail/config.h>

// get the execution policies definitions first
#include <thrust/system/cpp/detail/execution_policy.h>

// get the definition of par
#include <thrust/system/cpp/detail/par.h>

// now get all the algorithm definitions

#include <cusp/system/cpp/detail/convert.h>
#include <cusp/system/cpp/detail/elementwise.h>
#include <cusp/system/cpp/detail/multiply.h>
#include <cusp/system/cpp/detail/transpose.h>
