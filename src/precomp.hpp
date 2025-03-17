

#ifndef __OPENCV_SURFACE_MATCHING_PRECOMP_HPP__
#define __OPENCV_SURFACE_MATCHING_PRECOMP_HPP__

#include "surface_matching/ppf_match_3d.hpp"
#include "surface_matching/icp.hpp"
#include "surface_matching/ppf_helpers.hpp"

#include <string>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <ctime>

#include <fstream>
#include <iostream>
#include <algorithm>

#if defined (_OPENMP)
#include<omp.h>
#endif

#include <sstream>  // flann dependency, needed in precomp now
#include "opencv2/flann.hpp"

#include "c_utils.hpp"

#endif /* __OPENCV_SURFACE_MATCHING_PRECOMP_HPP__ */
