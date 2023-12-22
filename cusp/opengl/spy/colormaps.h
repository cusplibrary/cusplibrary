/*
 *  Copyright 2008-2013 Steven Dalton
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

namespace cusp
{
namespace opengl
{
namespace spy
{

#ifdef _WIN32
#if _MSC_VER >= 0
	// disable the truncation warning
    #pragma warning ( push )
	#pragma warning ( disable : 4305 )
#endif // _MSC_VER >= 0
#endif

float rainbow_color_map[64][3] = 
{
    {0,0,0.5625},
    {0,0,0.625},
    {0,0,0.6875},
    {0,0,0.75},
    {0,0,0.8125},
    {0,0,0.875},
    {0,0,0.9375},
    {0,0,1},
    {0,0.0625,1},
    {0,0.125,1},
    {0,0.1875,1},
    {0,0.25,1},
    {0,0.3125,1},
    {0,0.375,1},
    {0,0.4375,1},
    {0,0.5,1},
    {0,0.5625,1},
    {0,0.625,1},
    {0,0.6875,1},
    {0,0.75,1},
    {0,0.8125,1},
    {0,0.875,1},
    {0,0.9375,1},
    {0,1,1},
    {0.0625,1,0.9375},
    {0.125,1,0.875},
    {0.1875,1,0.8125},
    {0.25,1,0.75},
    {0.3125,1,0.6875},
    {0.375,1,0.625},
    {0.4375,1,0.5625},
    {0.5,1,0.5},
    {0.5625,1,0.4375},
    {0.625,1,0.375},
    {0.6875,1,0.3125},
    {0.75,1,0.25},
    {0.8125,1,0.1875},
    {0.875,1,0.125},
    {0.9375,1,0.0625},
    {1,1,0},
    {1,0.9375,0},
    {1,0.875,0},
    {1,0.8125,0},
    {1,0.75,0},
    {1,0.6875,0},
    {1,0.625,0},
    {1,0.5625,0},
    {1,0.5,0},
    {1,0.4375,0},
    {1,0.375,0},
    {1,0.3125,0},
    {1,0.25,0},
    {1,0.1875,0},
    {1,0.125,0},
    {1,0.0625,0},
    {1,0,0},
    {0.9375,0,0},
    {0.875,0,0},
    {0.8125,0,0},
    {0.75,0,0},
    {0.6875,0,0},
    {0.625,0,0},
    {0.5625,0,0},
    {0.5,0,0}
};

float bone_color_map[64][3] = 
{
    {0,0,0.0052},
    {0.0139,0.0139,0.0243},
    {0.0278,0.0278,0.0434},
    {0.0417,0.0417,0.0625},
    {0.0556,0.0556,0.0816},
    {0.0694,0.0694,0.1007},
    {0.0833,0.0833,0.1198},
    {0.0972,0.0972,0.1389},
    {0.1111,0.1111,0.158},
    {0.125,0.125,0.1771},
    {0.1389,0.1389,0.1962},
    {0.1528,0.1528,0.2153},
    {0.1667,0.1667,0.2344},
    {0.1806,0.1806,0.2535},
    {0.1944,0.1944,0.2726},
    {0.2083,0.2083,0.2917},
    {0.2222,0.2222,0.3108},
    {0.2361,0.2361,0.3299},
    {0.25,0.25,0.349},
    {0.2639,0.2639,0.3681},
    {0.2778,0.2778,0.3872},
    {0.2917,0.2917,0.4062},
    {0.3056,0.3056,0.4253},
    {0.3194,0.3194,0.4444},
    {0.3333,0.3385,0.4583},
    {0.3472,0.3576,0.4722},
    {0.3611,0.3767,0.4861},
    {0.375,0.3958,0.5},
    {0.3889,0.4149,0.5139},
    {0.4028,0.434,0.5278},
    {0.4167,0.4531,0.5417},
    {0.4306,0.4722,0.5556},
    {0.4444,0.4913,0.5694},
    {0.4583,0.5104,0.5833},
    {0.4722,0.5295,0.5972},
    {0.4861,0.5486,0.6111},
    {0.5,0.5677,0.625},
    {0.5139,0.5868,0.6389},
    {0.5278,0.6059,0.6528},
    {0.5417,0.625,0.6667},
    {0.5556,0.6441,0.6806},
    {0.5694,0.6632,0.6944},
    {0.5833,0.6823,0.7083},
    {0.5972,0.7014,0.7222},
    {0.6111,0.7205,0.7361},
    {0.625,0.7396,0.75},
    {0.6389,0.7587,0.7639},
    {0.6528,0.7778,0.7778},
    {0.6745,0.7917,0.7917},
    {0.6962,0.8056,0.8056},
    {0.7179,0.8194,0.8194},
    {0.7396,0.8333,0.8333},
    {0.7613,0.8472,0.8472},
    {0.783,0.8611,0.8611},
    {0.8047,0.875,0.875},
    {0.8264,0.8889,0.8889},
    {0.8481,0.9028,0.9028},
    {0.8698,0.9167,0.9167},
    {0.8915,0.9306,0.9306},
    {0.9132,0.9444,0.9444},
    {0.9349,0.9583,0.9583},
    {0.9566,0.9722,0.9722},
    {0.9783,0.9861,0.9861},
    {1,1,1}
};

float spring_color_map[64][3] = 
{
    {1.000000 ,0.000000 ,1.000000 },
    {1.000000 ,0.015873 ,0.984127 },
    {1.000000 ,0.031746 ,0.968254 },
    {1.000000 ,0.047619 ,0.952381 },
    {1.000000 ,0.063492 ,0.936508 },
    {1.000000 ,0.079365 ,0.920635 },
    {1.000000 ,0.095238 ,0.904762 },
    {1.000000 ,0.111111 ,0.888889 },
    {1.000000 ,0.126984 ,0.873016 },
    {1.000000 ,0.142857 ,0.857143 },
    {1.000000 ,0.158730 ,0.841270 },
    {1.000000 ,0.174603 ,0.825397 },
    {1.000000 ,0.190476 ,0.809524 },
    {1.000000 ,0.206349 ,0.793651 },
    {1.000000 ,0.222222 ,0.777778 },
    {1.000000 ,0.238095 ,0.761905 },
    {1.000000 ,0.253968 ,0.746032 },
    {1.000000 ,0.269841 ,0.730159 },
    {1.000000 ,0.285714 ,0.714286 },
    {1.000000 ,0.301587 ,0.698413 },
    {1.000000 ,0.317460 ,0.682540 },
    {1.000000 ,0.333333 ,0.666667 },
    {1.000000 ,0.349206 ,0.650794 },
    {1.000000 ,0.365079 ,0.634921 },
    {1.000000 ,0.380952 ,0.619048 },
    {1.000000 ,0.396825 ,0.603175 },
    {1.000000 ,0.412698 ,0.587302 },
    {1.000000 ,0.428571 ,0.571429 },
    {1.000000 ,0.444444 ,0.555556 },
    {1.000000 ,0.460317 ,0.539683 },
    {1.000000 ,0.476190 ,0.523810 },
    {1.000000 ,0.492063 ,0.507937 },
    {1.000000 ,0.507937 ,0.492063 },
    {1.000000 ,0.523810 ,0.476190 },
    {1.000000 ,0.539683 ,0.460317 },
    {1.000000 ,0.555556 ,0.444444 },
    {1.000000 ,0.571429 ,0.428571 },
    {1.000000 ,0.587302 ,0.412698 },
    {1.000000 ,0.603175 ,0.396825 },
    {1.000000 ,0.619048 ,0.380952 },
    {1.000000 ,0.634921 ,0.365079 },
    {1.000000 ,0.650794 ,0.349206 },
    {1.000000 ,0.666667 ,0.333333 },
    {1.000000 ,0.682540 ,0.317460 },
    {1.000000 ,0.698413 ,0.301587 },
    {1.000000 ,0.714286 ,0.285714 },
    {1.000000 ,0.730159 ,0.269841 },
    {1.000000 ,0.746032 ,0.253968 },
    {1.000000 ,0.761905 ,0.238095 },
    {1.000000 ,0.777778 ,0.222222 },
    {1.000000 ,0.793651 ,0.206349 },
    {1.000000 ,0.809524 ,0.190476 },
    {1.000000 ,0.825397 ,0.174603 },
    {1.000000 ,0.841270 ,0.158730 },
    {1.000000 ,0.857143 ,0.142857 },
    {1.000000 ,0.873016 ,0.126984 },
    {1.000000 ,0.888889 ,0.111111 },
    {1.000000 ,0.904762 ,0.095238 },
    {1.000000 ,0.920635 ,0.079365 },
    {1.000000 ,0.936508 ,0.063492 },
    {1.000000 ,0.952381 ,0.047619 },
    {1.000000 ,0.968254 ,0.031746 },
    {1.000000 ,0.984127 ,0.015873 },
    {1.000000 ,1.000000 ,0.000000 },
};

#ifdef _WIN32
#if _MSC_VER >= 0
	// enable the trunction warning
    #pragma warning ( pop )
#endif // _MSC_VER >= 0
#endif

} // end spy
} // end opengl
} // end cusp
