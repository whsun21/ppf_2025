

/** @file
@author Tolga Birdal <tbirdal AT gmail.com>
*/

#ifndef __OPENCV_SURFACE_MATCHING_HELPERS_HPP__
#define __OPENCV_SURFACE_MATCHING_HELPERS_HPP__

#include <opencv2/core.hpp>
#include "flann/flann.hpp"
#include "flann/algorithms/dist.h"

#include<vcg/complex/complex.h>

using namespace vcg;
using namespace std;

class MyEdge;
class MyFace;
class MyVertex;
struct MyUsedTypes : public UsedTypes<	Use<MyVertex>   ::AsVertexType,
                                        Use<MyEdge>     ::AsEdgeType,
                                        Use<MyFace>     ::AsFaceType>{};

class MyVertex  : public Vertex<MyUsedTypes, vertex::Coord3f, vertex::Normal3f, vertex::BitFlags  >{};
class MyFace    : public Face< MyUsedTypes, face::FFAdj,  face::Normal3f, face::VertexRef, face::BitFlags > {};
class MyEdge    : public Edge<MyUsedTypes>{};
class MyMesh    : public tri::TriMesh< vector<MyVertex>, vector<MyFace> , vector<MyEdge>  > {};


namespace kdtree {
	typedef flann::Index<flann::L2<float> > KDTree;

	void SearchKDTree(KDTree* tree, const cv::Mat& input, std::vector<std::vector<int>>& indices, std::vector<std::vector<float>>& dists, int nn);
	KDTree* BuildKDTree(const  cv::Mat& data);

}

namespace cv
{
namespace ppf_match_3d
{

//! @addtogroup surface_matching
//! @{

int writeMap(const map<string, vector<double>>& m, const string& outPath);
int writeMap(const map<string, vector<string>>& m, const string& outPath);
int writeMap(const map<string, double>& m, const string& outPath);
int readMap(map<string, vector<double>>& m2, const string& inPath);
int readMap(map<string, double>& m2, const string& inPath);
int readMap(map<string, vector<string>>& m2, const string& inPath);


/**
 *  @brief Load a PLY file
 *  @param [in] fileName The PLY model to read
 *  @param [in] withNormals Flag wheather the input PLY contains normal information,
 *  and whether it should be loaded or not
 *  @return Returns the matrix on successful load
 */
CV_EXPORTS_W Mat loadPLYSimple(const char* fileName, int withNormals = 0);

CV_EXPORTS_W Mat loadPLYSimple_bin(const char* fileName, int withNormals = 0);

void cvMat2vcgMesh(const Mat& pc, MyMesh& m);
void vcgMesh2cvMat(const MyMesh& m, Mat& pc);

/**
 *  @brief Write a point cloud to PLY file
 *  @param [in] PC Input point cloud
 *  @param [in] fileName The PLY model file to write
*/
CV_EXPORTS_W void writePLY(Mat PC, const char* fileName);

/**
*  @brief Used for debbuging pruposes, writes a point cloud to a PLY file with the tip
*  of the normal vectors as visible red points
*  @param [in] PC Input point cloud
*  @param [in] fileName The PLY model file to write
*/
CV_EXPORTS_W void writePLYVisibleNormals(Mat PC, const char* fileName);

Mat samplePCUniform(Mat PC, int sampleStep);
Mat samplePCUniformInd(Mat PC, int sampleStep, std::vector<int>& indices);

/**
 *  Sample a point cloud using uniform steps
 *  @param [in] pc Input point cloud
 *  @param [in] xrange X components (min and max) of the bounding box of the model
 *  @param [in] yrange Y components (min and max) of the bounding box of the model
 *  @param [in] zrange Z components (min and max) of the bounding box of the model
 *  @param [in] sample_step_relative The point cloud is sampled such that all points
 *  have a certain minimum distance. This minimum distance is determined relatively using
 *  the parameter sample_step_relative.
 *  @param [in] weightByCenter The contribution of the quantized data points can be weighted
 *  by the distance to the origin. This parameter enables/disables the use of weighting.
 *  @return Sampled point cloud
*/
CV_EXPORTS_W Mat samplePCByQuantization(Mat pc, Vec2f& xrange, Vec2f& yrange, Vec2f& zrange, float sample_step_relative, int weightByCenter=0);

CV_EXPORTS_W Mat samplePCByQuantization_cube(Mat pc, Vec2f& xrange, Vec2f& yrange, Vec2f& zrange, float sample_step, int weightByCenter = 0);

Mat samplePCByQuantization_normal(Mat pc, Vec2f& xrange, Vec2f& yrange, Vec2f& zrange, float sampleStep, float anglethreshold, int level);

void computeBboxStd(Mat pc, Vec2f& xRange, Vec2f& yRange, Vec2f& zRange);

void* indexPCFlann(Mat pc);
void destroyFlann(void* flannIndex);
void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances);
void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances, const int numNeighbors);
void queryPCFlannRadius(void* flannIndex, Mat& pc, Mat& indices, Mat& distances, double radius);

Mat normalizePCCoeff(Mat pc, float scale, float* Cx, float* Cy, float* Cz, float* MinVal, float* MaxVal);
Mat transPCCoeff(Mat pc, float scale, float Cx, float Cy, float Cz, float MinVal, float MaxVal);

/**
 *  Transforms the point cloud with a given a homogeneous 4x4 pose matrix (in double precision)
 *  @param [in] pc Input point cloud (CV_32F family). Point clouds with 3 or 6 elements per
 *  row are expected. In the case where the normals are provided, they are also rotated to be
 *  compatible with the entire transformation
 *  @param [in] Pose 4x4 pose matrix, but linearized in row-major form.
 *  @return Transformed point cloud
*/
CV_EXPORTS_W Mat transformPCPose(Mat pc, const Matx44d& Pose);

/**
 *  Generate a random 4x4 pose matrix
 *  @param [out] Pose The random pose
*/
CV_EXPORTS_W void getRandomPose(Matx44d& Pose);

/**
 *  Adds a uniform noise in the given scale to the input point cloud
 *  @param [in] pc Input point cloud (CV_32F family).
 *  @param [in] scale Input scale of the noise. The larger the scale, the more noisy the output
*/
CV_EXPORTS_W Mat addNoisePC(Mat pc, double scale);

/**
 *  @brief Compute the normals of an arbitrary point cloud
 *  computeNormalsPC3d uses a plane fitting approach to smoothly compute
 *  local normals. Normals are obtained through the eigenvector of the covariance
 *  matrix, corresponding to the smallest eigen value.
 *  If PCNormals is provided to be an Nx6 matrix, then no new allocation
 *  is made, instead the existing memory is overwritten.
 *  @param [in] PC Input point cloud to compute the normals for.
 *  @param [out] PCNormals Output point cloud
 *  @param [in] NumNeighbors Number of neighbors to take into account in a local region
 *  @param [in] FlipViewpoint Should normals be flipped to a viewing direction?
 *  @param [in] viewpoint
 *  @return Returns 0 on success
 */
CV_EXPORTS_W int computeNormalsPC3d(const Mat& PC, CV_OUT Mat& PCNormals, const int NumNeighbors, const bool FlipViewpoint, const Vec3f& viewpoint);

//! @}

} // namespace ppf_match_3d
} // namespace cv

#endif
