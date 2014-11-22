/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a count of the License at
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

#include <thrust/detail/config.h>

// the purpose of this header is to #include the hilbert_curve.h header
// of the sequential, host, and device systems. It should be #included in any
// code which uses adl to dispatch hilbert_curve

#include <cusp/system/detail/sequential/graph/hilbert_curve.h>

#define __CUSP_HOST_SYSTEM_HILBERT_CURVE_HEADER <__CUSP_HOST_SYSTEM_ROOT/detail/graph/hilbert_curve.h>
#include __CUSP_HOST_SYSTEM_HILBERT_CURVE_HEADER
#undef __CUSP_HOST_SYSTEM_HILBERT_CURVE_HEADER

#define __CUSP_DEVICE_SYSTEM_HILBERT_CURVE_HEADER <__CUSP_DEVICE_SYSTEM_ROOT/detail/graph/hilbert_curve.h>
#include __CUSP_DEVICE_SYSTEM_HILBERT_CURVE_HEADER
#undef __CUSP_DEVICE_SYSTEM_HILBERT_CURVE_HEADER
