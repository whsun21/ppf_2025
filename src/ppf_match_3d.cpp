
#include "precomp.hpp"
#include "hash_murmur.hpp"

#include <opencv2/core/eigen.hpp>
#include "IcpOptimizer.h"
//#include "surface_matching/poisson_disk_sampling.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/StdVector"
#include "Eigen/Dense"


using namespace Eigen;
using namespace std;

namespace cv
{
namespace ppf_match_3d
{

static const size_t PPF_LENGTH = 6;

static void sort_vec(const VectorXd& vec, VectorXd& sorted_vec, VectorXi& ind) {
    ind = VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);//[0 1 2 3 ... N-1]
    auto rule = [vec](int i, int j)->bool {
        return vec(i) > vec(j);
        };//正则表达式，作为sort的谓词
    std::sort(ind.data(), ind.data() + ind.size(), rule);
    //data成员函数返回VectorXd的第一个元素的指针，类似于begin()
    sorted_vec.resize(vec.size());
    for (int i = 0; i < vec.size(); i++) {
        sorted_vec(i) = vec(ind(i));
    }
}

// routines for assisting sort
static bool pose3DPtrCompare(const Pose3DPtr& a, const Pose3DPtr& b)
{
  CV_Assert(!a.empty() && !b.empty());
  return ( a->numVotes > b->numVotes );
}

static bool pose3DPtrCompareOverlap(const Pose3DPtr& a, const Pose3DPtr& b)
{
    CV_Assert(!a.empty() && !b.empty()); 
    return (a->overlap > b->overlap);
}

static bool pose3DPtrCompareClusterCount(const Pose3DPtr& a, const Pose3DPtr& b)
{
    CV_Assert(!a.empty() && !b.empty());
    return (a->cluster_count > b->cluster_count);
}

static bool pose3DPtrCompareIntersecNum(const Pose3DPtr& a, const Pose3DPtr& b)
{
    CV_Assert(!a.empty() && !b.empty());
    return (a->freespaceIntersec> b->freespaceIntersec);
}


static int sortPoseClusters(const PoseCluster3DPtr& a, const PoseCluster3DPtr& b)
{
  CV_Assert(!a.empty() && !b.empty());
  return ( a->numVotes > b->numVotes );
}

// simple hashing
/*static int hashPPFSimple(const Vec4d& f, const double AngleStep, const double DistanceStep)
{
  Vec4i key(
      (int)(f[0] / AngleStep),
      (int)(f[1] / AngleStep),
      (int)(f[2] / AngleStep),
      (int)(f[3] / DistanceStep));

  int hashKey = d.val[0] | (d.val[1] << 8) | (d.val[2] << 16) | (d.val[3] << 24);
  return hashKey;
}*/

// quantize ppf and hash it for proper indexing
static KeyType hashPPF(const Vec4d& f, const double AngleStep, const double DistanceStep)
{
  Vec4i key(
      (int)(f[0] / AngleStep),
      (int)(f[1] / AngleStep),
      (int)(f[2] / AngleStep),
      (int)(f[3] / DistanceStep));
  KeyType hashKey[2] = {0, 0};  // hashMurmurx64() fills two values

  murmurHash(key.val, 4*sizeof(int), 42, &hashKey[0]);
  return hashKey[0];
}

static KeyType hashPPF2(const Vec4d& f, Vec4i& fd, const double AngleStep, const double DistanceStep, const double OrienDiffAngleStep)
{
    fd[0] = (int)(f[0] / AngleStep);
    fd[1] = (int)(f[1] / AngleStep);
    fd[2] = (int)(f[2] / AngleStep);
    fd[3] = (int)(f[3] / DistanceStep);
    KeyType hashKey[2] = { 0, 0 };  // hashMurmurx64() fills two values

    murmurHash(fd.val, 4 * sizeof(int), 42, &hashKey[0]); // 5 int
    return hashKey[0];
}

static KeyType hashPPF3(const Vec4d& f, const double theta, Vec<int, 5>& fd, const double AngleStep, const double DistanceStep, const double OrienDiffAngleStep)
{
    fd[0] = (int)(f[0] / AngleStep);
    fd[1] = (int)(f[1] / AngleStep);
    fd[2] = (int)(f[2] / AngleStep);
    fd[3] = (int)(f[3] / DistanceStep);
    fd[4] = (int)(theta / OrienDiffAngleStep);

    KeyType hashKey[2] = { 0, 0 };  // hashMurmurx64() fills two values

    murmurHash(fd.val, 5 * sizeof(int), 42, &hashKey[0]); // 5 int
    return hashKey[0];
    //const std::size_t h1 = std::hash<int>{}(fd[0]); // pcl 
    //const std::size_t h2 = std::hash<int>{}(fd[1]);
    //const std::size_t h3 = std::hash<int>{}(fd[2]);
    //const std::size_t h4 = std::hash<int>{}(fd[3]);
    //const std::size_t h5 = std::hash<int>{}(fd[4]);
    //return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);

}

/*static size_t hashMurmur(uint key)
{
  size_t hashKey=0;
  hashMurmurx86((void*)&key, 4, 42, &hashKey);
  return hashKey;
}*/

static double computeAlpha(const Vec3d& p1, const Vec3d& n1, const Vec3d& p2)
{
  Vec3d Tmg, mpt;
  Matx33d R;
  double alpha;

  computeTransformRT(p1, n1, R, Tmg);
  mpt = Tmg + R * p2;
  alpha=atan2(-mpt[2], mpt[1]);

  if ( alpha != alpha)
  {
    return 0;
  }

  if (sin(alpha)*mpt[2]<0.0)
    alpha=-alpha;

  return (-alpha);
}

static double computeTheta(const Vec3d& p1, const Vec3d& n1, const Vec3d& p2, const Vec3d& n2)
{
    Vec3d dp, o;
    double theta;

    dp = p2 - p1;
    TNormalize3(dp);
    o = n1.cross(dp);
    TNormalize3(o);
    theta = TAngle3Normalized(o, n2);

    return theta;
}

PPF3DDetector::PPF3DDetector()
{
  sampling_step_relative = 0.05;
  distance_step_relative = 0.05;
  scene_sample_step = (int)(1/0.04);
  angle_step_relative = 30;
  angle_step_radians = (360.0/angle_step_relative)*M_PI/180.0;
  angle_step = angle_step_radians;
  trained = false;

  hash_table = NULL;
  hash_nodes = NULL;

  setSearchParams();
}

PPF3DDetector::PPF3DDetector(const double RelativeSamplingStep, const double RelativeDistanceStep, const double NumAngles)
{
  sampling_step_relative = RelativeSamplingStep;
  distance_step_relative = RelativeDistanceStep; // 用不到
  angle_step_relative = NumAngles;
  angle_step_radians = (360.0/angle_step_relative)*M_PI/180.0;
  //SceneSampleStep = 1.0/RelativeSceneSampleStep;
  angle_step = angle_step_radians;
  trained = false;

  hash_table = NULL;
  hash_nodes = NULL;

  enableDebug(false);
  setSamplingMethod((string)"cube");
  //setOrientationDiffThreshold(2. / 180. * M_PI);  // 不要在这里设置,错误的根源
  setNMSThreshold(0.5);
  enableFreeSpaceConstraint(true);
  //setSearchParams();
}

void PPF3DDetector::setSearchParams(const double positionThreshold, const double rotationThreshold, const bool useWeightedClustering)
{
  if (positionThreshold<0)
    position_threshold = sampling_step_relative; // 有问题
  else
    position_threshold = positionThreshold;

  if (rotationThreshold<0)
    rotation_threshold = ((360 / angle_step) / 180.0 * M_PI); // 有问题
  else
    rotation_threshold = rotationThreshold;

  use_weighted_avg = useWeightedClustering;
}

Mat PPF3DDetector::getSampledModel()
{
    if (!trained) {
        throw cv::Exception(cv::Error::StsError, "The model is not trained. Cannot get SampledModel without training", __FUNCTION__, __FILE__, __LINE__);
    }
    return sampled_pc;
}

void PPF3DDetector::enableDebug(bool Debug) {
    debug = Debug;
}
void PPF3DDetector::enableFreeSpaceConstraint(bool usfs) {
    useFreeSpaceConstraint = usfs;
}
void PPF3DDetector::setSceneKeypointForDebug(Mat& PC) {
    debug_sampled_scene_ref = PC;
}

void PPF3DDetector::setGtPose(Matx44d& Pose) {
    if (!trained)
    {
        throw cv::Exception(cv::Error::StsError, "The model is not trained. Cannot setGtPose without training", __FUNCTION__, __FILE__, __LINE__);
    }

    gtPose = Pose;
    gtPoseModel = transformPCPose(sampled_pc, Pose);
}

void PPF3DDetector::setMultiGtPose(std::vector<Matx44d>& Poses, std::string & debugGTPoseModelPath) {
    if (!trained)
    {
        throw cv::Exception(cv::Error::StsError, "The model is not trained. Cannot setGtPose without training", __FUNCTION__, __FILE__, __LINE__);
    }

    multi_gt_pose = Poses;
    use_multi_gt_pose = true;
    Mat center = Mat(1, model_center.rows, CV_32F);
    float* pcData = center.ptr<float>(0);
    pcData[0] = (float)model_center[0];
    pcData[1] = (float)model_center[1];
    pcData[2] = (float)model_center[2];
    multi_gt_pose_model.resize(Poses.size());
    multi_gt_pose_center.resize(Poses.size());
    for (int i = 0; i < Poses.size(); i++) {
        multi_gt_pose_model[i] = transformPCPose(sampled_pc, Poses[i]);
        multi_gt_pose_center[i] = transformPCPose(center, Poses[i]);
        writePLY(multi_gt_pose_model[i], (debugGTPoseModelPath+"debug_gt_pose_"+to_string(i)+".ply").c_str());
        
    }
}
void PPF3DDetector::saveGTPoseModel(std::string& path) {
    if (!trained)
    {
        throw cv::Exception(cv::Error::StsError, "The model is not trained. Cannot saveGTPoseModel without training", __FUNCTION__, __FILE__, __LINE__);
    }

    writePLY(gtPoseModel, path.c_str());
}
void PPF3DDetector::setDebugFolderName(std::string& path) {
    debug_folder_name = path;
}
void PPF3DDetector::setSamplingMethod(std::string& Method) {
    samplingMethod = Method;
}

void PPF3DDetector::setNMSThreshold(double th) {
    nmsThreshold = th;
}

void PPF3DDetector::setClusterThreshold(double th) {
    clusterThreshold = th;
}

void PPF3DDetector::setOrientationDiffThreshold(double th) {
    orientation_diff_threshold = th;
}
void PPF3DDetector::setSDF(std::shared_ptr < Discregrid::DiscreteGrid>& sdfPtr) {
    sdf_ptr = sdfPtr;
}

void PPF3DDetector::setPPFAngleConstraint(double& angle) {
    ppf_angle_constraint = angle;
}
void PPF3DDetector::setPPFDistanceConstraint(double& dist_th) {
    ppf_distance_constraint = dist_th;
}

void PPF3DDetector::setSpectral(bool spectral_) {
    spectral = spectral_;
}

// compute per point PPF as in paper
void PPF3DDetector::computePPFFeatures(const Vec3d& p1, const Vec3d& n1,
                                       const Vec3d& p2, const Vec3d& n2,
                                       Vec4d& f)
{
  Vec3d d(p2 - p1);
  f[3] = cv::norm(d);
  if (f[3] <= EPS)
    return;
  d *= 1.0 / f[3];

  f[0] = TAngle3Normalized(n1, d);
  f[1] = TAngle3Normalized(n2, d);
  f[2] = TAngle3Normalized(n1, n2);
}

void PPF3DDetector::clearTrainingModels()
{
  if (this->hash_nodes)
  {
    free(this->hash_nodes);
    this->hash_nodes=0;
  }

  if (this->hash_table)
  {
    hashtableDestroy(this->hash_table);
    this->hash_table=0;
  }
}

PPF3DDetector::~PPF3DDetector()
{
  clearTrainingModels();
}


void PPF3DDetector::setModelTriplet(Vec3f& center, float diameter) {
    model_triplet = Mat(3, 3, CV_32F);
    float* pcData = model_triplet.ptr<float>(0);
    pcData[0] = (float)center[0];
    pcData[1] = (float)center[1];
    pcData[2] = (float)center[2];
    pcData = model_triplet.ptr<float>(1);
    pcData[0] = (float)center[0]+ diameter;
    pcData[1] = (float)center[1];
    pcData[2] = (float)center[2];
    pcData = model_triplet.ptr<float>(2);
    pcData[0] = (float)center[0];
    pcData[1] = (float)center[1] + diameter;
    pcData[2] = (float)center[2];

}


// TODO: Check all step sizes to be positive
void PPF3DDetector::trainModel(const Mat &PC)
{
  CV_Assert(PC.type() == CV_32F || PC.type() == CV_32FC1);

  // compute bbox
  Vec2f xRange, yRange, zRange;
  computeBboxStd(PC, xRange, yRange, zRange);

  // compute sampling step from diameter of bbox
  float dx = xRange[1] - xRange[0];
  float dy = yRange[1] - yRange[0];
  float dz = zRange[1] - zRange[0];
  float diameter = sqrt ( dx * dx + dy * dy + dz * dz );

  Vec3f modelCenter;
  modelCenter(0) = (xRange[1] + xRange[0]) / 2;
  modelCenter(1) = (yRange[1] + yRange[0]) / 2;
  modelCenter(2) = (zRange[1] + zRange[0]) / 2;
  model_center = modelCenter;
  model_minimum_length = std::min({ dx, dy, dz });
  model_bbox = Mat(8, 3, CV_32F);
  for (int i = 0; i< 2; i++) {
      for (int j = 0; j < 2; j++) {
          for (int k = 0; k < 2; k++) {
              int ind = i * 2 * 2 + j * 2 + k;
              float* pcData = model_bbox.ptr<float>(ind);
              pcData[0] = (float)xRange[i];
              pcData[1] = (float)yRange[j];
              pcData[2] = (float)zRange[k];
          }
      }
  }
  setModelTriplet(modelCenter, diameter);

  float distanceStep = (float)(diameter * sampling_step_relative);
  //float distanceStepFeature = (float)(diameter * distance_step_relative);
  //distanceStep /= 2;
  float distanceStepRefine = (float)(diameter * 0.01);

  //Mat sampled0 = samplePCByQuantization(PC, xRange, yRange, zRange, (float)sampling_step_relative,0);
  //writePLY(sampled0, "../samples/data/results/sampled_model0.ply");
  Mat sampled, sampledRefinement;
  if (samplingMethod == "cube") {
      sampled = samplePCByQuantization_cube(PC, xRange, yRange, zRange, distanceStep, 0);
      sampledRefinement = samplePCByQuantization_cube(PC, xRange, yRange, zRange, distanceStepRefine, 0);

  }
  else if (samplingMethod == "normal") {
      sampled = samplePCByQuantization_normal(PC, xRange, yRange, zRange, distanceStep, 15.0 / 180 * M_PI, 3);
      sampledRefinement = samplePCByQuantization_normal(PC, xRange, yRange, zRange, distanceStepRefine, 15.0 / 180 * M_PI, 3);
  }
  else { cout << "Unkown samplingMethod: " << samplingMethod << endl; exit(1); }

  if (debug) {
      string name1 = "../samples/data/results/" + debug_folder_name + "/sampled_model.ply";
      string name2 = "../samples/data/results/" + debug_folder_name + "/sampled_model_refine.ply";
      writePLY(sampled, name1.c_str());
      writePLY(sampledRefinement, name2.c_str());
  }

  int size = sampled.rows*sampled.rows;

  hashtable_int* hashTable = hashtableCreate(size, NULL);

  int numPPF = sampled.rows*sampled.rows;
  // 特征矩阵ppf：[|M|^2, 5], 前4维存ppf特征 F(mr, mi)，
  // 最后一个存alpha_model，这是将mr、n_mr和原点O、x正向对齐之后，将mi绕x轴旋转到XO+Y平面上，所需的旋转角
  ppf = Mat(numPPF, PPF_LENGTH, CV_32FC1);

  // TODO: Maybe I could sample 1/5th of them here. Check the performance later.
  int numRefPoints = sampled.rows;

  // pre-allocate the hash nodes
  hash_nodes = (THash*)calloc(numRefPoints*numRefPoints, sizeof(THash));

  // TODO : This can easily be parallelized. But we have to lock hashtable_insert.
  // I realized that performance drops when this loop is parallelized (unordered
  // inserts into the hashtable
  // But it is still there to be investigated. For now, I leave this unparallelized
  // since this is just a training part.
#if defined _OPENMP
#pragma omp parallel for
#endif
  for (int i=0; i<numRefPoints; i++)
  {
    const Vec3f p1(sampled.ptr<float>(i));
    const Vec3f n1(sampled.ptr<float>(i) + 3);

    //printf("///////////////////// NEW REFERENCE ////////////////////////\n");
    for (int j=0; j<numRefPoints; j++)
    {
      // cannot compute the ppf with myself
      if (i!=j)
      {
        const Vec3f p2(sampled.ptr<float>(j));
        const Vec3f n2(sampled.ptr<float>(j) + 3);

        Vec4d f1_4 = Vec4d::all(0);
        computePPFFeatures(p1, n1, p2, n2, f1_4);

        if (abs(f1_4[0] - M_PI/2) < ppf_angle_constraint) continue;
        if (f1_4[3] < ppf_distance_constraint * distanceStep) continue;

        double f5theta = computeTheta(p1, n1, p2, n2);

        KeyType hashValue = hashPPF(f1_4, angle_step_radians, distanceStep);
        Vec<int, 5> fd;
        //KeyType hashValue = hashPPF3(f1_4, f5theta, fd, angle_step_radians, distanceStep, orientation_diff_threshold);
        
        double alpha = computeAlpha(p1, n1, p2);
        // 模型点对(mr, mi)的特征在矩阵ppf中的位置
        uint ppfInd = i*numRefPoints+j;

        THash* hashNode = &hash_nodes[i*numRefPoints+j];
        hashNode->fd = fd; //
        hashNode->i = i;
        hashNode->ppfInd = ppfInd;

        hashtableInsertHashed_2(hashTable, hashValue, (void*)hashNode); // hashtableInsertHashed_2

        Mat(f1_4).reshape(1, 1).convertTo(ppf.row(ppfInd).colRange(0, 4), CV_32F);
        ppf.ptr<float>(ppfInd)[4] = (float)f5theta;
        ppf.ptr<float>(ppfInd)[5] = (float)alpha;
      }
    }
  }
  setSearchParams(diameter / 5.0, angle_step_radians*10, false);  //angle_step_radians, 作者建议位移阈值diam（M ）/10
  //setSearchParams(distanceStep, angle_step_radians, false);  //作者建议位移阈值diam（M ）/10

  angle_step = angle_step_radians;
  distance_step = distanceStep;
  hash_table = hashTable;
  num_ref_points = numRefPoints;
  sampled_pc = sampled;
  trained = true;

  //
  model_diameter = diameter;
  sampled_pc_refinement = sampledRefinement;
}



///////////////////////// MATCHING ////////////////////////////////////////

/*
void PPF3DDetector::overlapRatio(std::vector<Pose3DPtr>& results, double threshold, double  anglethreshold)
{
    double thresholdlen = threshold * model_diameter;
    MatrixXf samplecood(sampled_pc.rows(), 3);//sampled_pc采样后的模型
    samplecood = sampled_pc.block(0, 0, sampled_pc.rows(), 3);



#ifdef _OPENMP
#pragma omp parallel for schedule(static) 
#endif
    for (int i = 0; i < results.size(); i++)
    {
        MatrixXf pct = transformPose3f(samplecood, results[i].Pose);
        std::vector<std::vector<int>> indices;
        std::vector<std::vector<float>> dists;
        SearchKDTree(scene_knn_tree, pct, indices, dists, 1);
        //int pointnum = 0;
        int pointnum2 = 0;
        //VectorXi num(sampled_scene.rows());
        //VectorXi num2(sampled_scene.rows());
        //num.setZero();
        //num2.setZero();
        //MatrixXf curv(sampled_pc.rows(), 2);

        for (int j = 0; j < sampled_pc.rows(); j++)
        {

            if (dists[j][0] < thresholdlen)
            {
                //num2(indices[j][0]) = 1;
                pointnum2++;


            }

        }
        //results[i].overlap = num.sum();
        //results[i].overnum = pointnum2 ;

        results[i].overlap = 1.0 * pointnum2 / sampled_pc.rows();
        //	cout << pointnum << "  " << pointnum*1.0/pointnum2  << "  " << 1.0*pointnum2 / sampled_pc.rows() << endl;

    }

    //std::sort(results.begin(),results.end(), pose3DPtrCompareoverlapS);//排序算法 对姿态按投票数排序 测试用

}*/

void PPF3DDetector::overlapRatio(const Mat& srcPC, const Mat& dstPC, void* dstFlann, std::vector<Pose3DPtr>& resultsS, double threshold, double anglethreshold, int num_neighbors)
{
    double thresholdlen = threshold ;
    thresholdlen = thresholdlen * thresholdlen;

    double cos_angle_th = -1.01;
    if (anglethreshold < 180)
        cos_angle_th = cos(anglethreshold / 180 * M_PI);

    int NumNeighbors = num_neighbors;


#ifdef _OPENMP
#pragma omp parallel for 
#endif

    for (int i = 0; i < resultsS.size(); i++) {
        Mat srcMoved = transformPCPose(srcPC, resultsS[i]->pose);

        size_t numElSrc = (size_t)srcPC.rows;

        //int sizesResult[2] = { (int)numElSrc, 1 };
        //float* distances = new float[numElSrc];
        //int* indices = new int[numElSrc];
        //Mat Indices(2, sizesResult, CV_32S, indices, 0);
        //Mat Distances(2, sizesResult, CV_32F, distances, 0);
        //queryPCFlann(dstFlann, srcMoved, Indices, Distances);

        Mat Distances(srcPC.rows, NumNeighbors, CV_32F);
        Mat Indices(srcPC.rows, NumNeighbors, CV_32S);
        queryPCFlann(dstFlann, srcMoved, Indices, Distances, NumNeighbors);


        int pointNum = 0;
        for (int j = 0; j < numElSrc; j++)
        {
            //const Vec3f n1(srcMoved.ptr<float>(j) + 3);
            //int sceneInd = Indices.at<int>(j);
            //const Vec3f n2(dstPC.ptr<float>(sceneInd) + 3);
            //double a = n1.dot(n2);
            //if (Distances.at<float>(j) < thresholdlen)
            //{
            //    if (a > angle)
            //    {
            //        pointNum++;// sumnum_model(j);
            //    }
            //}
            
            const Vec3f n1(srcMoved.ptr<float>(j) + 3);
            const int* indLocal = Indices.ptr<int>(j);
            const float* distLocal = Distances.ptr<float>(j);
            for (int in = 0; in < NumNeighbors; ++in)
            {
                int sceneInd = indLocal[in];
                const Vec3f n2(dstPC.ptr<float>(sceneInd) + 3);
                double cosa = n1.dot(n2);
                float d = distLocal[in];
                if (d < thresholdlen)
                {
                    if (cosa > cos_angle_th)
                    {
#if defined (_OPENMP)
#pragma omp critical
#endif
                        {
                            pointNum++;
                        }
                        break; // 找到一个就可以
                    }
                }

            }


        }
        double Overlap = 1.0 * pointNum ; // / numElSrc
        resultsS[i]->updateOverlap(Overlap);

        //delete[] distances;
        //delete[] indices;
    }
}


void PPF3DDetector::generateFreeSpaceVolume(Mat& sampled_for_freespace, float resulotion) {
    // free space volume
    float step_size = 5 * resulotion;
    //float step_size = resulotion; // sampled scene resolution
    int step_num = 0;
    Eigen::Vector3f origin(0, 0, 0);
    Eigen::Vector3f scene_point;
    Eigen::Vector3f t;
    Eigen::Vector3f d;
    Eigen::Vector3f pj;
    vector<Eigen::Vector3f> all_pts;

    Eigen::MatrixXf scene_cloud;
    //cv::cv2eigen(sampled_for_freespace, scene_cloud);

    Mat small = samplePCUniform(sampled_for_freespace, 2);
    cv::cv2eigen(small, scene_cloud);

    //Mat small;
    //if (sampled_for_freespace.rows > 2000) {
    //    int size = sampled_for_freespace.rows;
    //    Mat cloud_copy;
    //    sampled_for_freespace.copyTo(cloud_copy);
    //    while (size > 2000) {
    //        Mat small = samplePCUniform(cloud_copy, 2);
    //        size = small.rows;
    //        cloud_copy = small;
    //    }
    //    cv::cv2eigen(cloud_copy, scene_cloud);
    //}
    //else {
    //    cv::cv2eigen(sampled_for_freespace, scene_cloud);
    //}


    for (size_t i = 0; i < scene_cloud.rows(); ++i) {
        scene_point = scene_cloud.row(i).leftCols(3);
        t = origin - scene_point;
        step_num = t.norm() / step_size;
        t.normalize();
        d = step_size * t;

        //pj = scene_point + resulotion * t; // first layer
        //all_pts.push_back(pj);

        for (size_t j = 1; j < step_num / 5; ++j) {
            pj = scene_point + j * d;
            all_pts.push_back(pj);
        }
    }

    Eigen::Matrix<float, -1, 3, 1> volume;
    volume.resize(all_pts.size(), 3);
    for (size_t k = 0; k < all_pts.size(); k++) {
        volume.row(k) = all_pts[k];
    }
    cv::eigen2cv(volume, scene_freespace);
    if (debug) {
        string prefix = "../samples/data/results/" + debug_folder_name;
        writePLY(scene_freespace, (prefix + "/fs.ply").c_str());
        //writePLY(scene_freespace, "fs.ply");
    }

}
/*
// free space intersection count
void PPF3DDetector::freespaceIntersectionCount( std::vector<Pose3DPtr>& resultsS, std::vector<Pose3DPtr>& finalPoses, const int& th)
{
    finalPoses.clear();
    finalPoses.reserve(resultsS.size());

    // sdf, inside is +, outside is -

#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for (int i = 0; i < resultsS.size(); i++) {
        Eigen::VectorXi verts_sdf_contains = MatrixXi::Zero(downsample_scene_freespace.rows, 1);
        // transpose freespace
        Mat freespaceMoved = transformPCPose(downsample_scene_freespace, resultsS[i]->pose.inv());
        for (int k = 0; k < downsample_scene_freespace.rows; ++k)
        {
            float* sample = freespaceMoved.ptr<float>(k);
            double d = sdf_ptr->interpolate(0, { sample[0], sample[1], sample[2]});
            if (d == std::numeric_limits<double>::max()) { d = -1.; }
            if (d > 0) { verts_sdf_contains(k) = 1; }
        }
        resultsS[i]->updatefreespaceIntersec(verts_sdf_contains.sum());
        if (verts_sdf_contains.sum() < th) {
#if defined (_OPENMP)
#pragma omp critical
#endif
            { finalPoses.push_back(resultsS[i]); }
            
        }
        // 
        //if (verts_sdf_contains.sum() == 0) {
        //    cout << i << endl;
        //    cout << resultsS[i]->pose << endl;
        //}
        //cout << verts_sdf_contains.sum() <<endl;
    }
}
*/
void PPF3DDetector::freespaceRatio(std::vector<Pose3DPtr>& resultsS, const double& th)
{
    double thresholdlen = th;


    // sdf, inside is +, outside is -
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for (int i = 0; i < resultsS.size(); i++) {
        //Eigen::VectorXi verts_sdf_overlap = MatrixXi::Zero(scene_freespace.rows, 1);
        Eigen::VectorXi verts_sdf_beyond = MatrixXi::Zero(scene_freespace.rows, 1);
        //Eigen::VectorXi verts_sdf_beyond  = MatrixXi::Zero(scene_freespace.rows, 1);
        // transpose freespace
        Mat freespaceMoved = transformPCPose(scene_freespace, resultsS[i]->pose.inv());

        for (int k = 0; k < scene_freespace.rows; ++k)
        {
            float* sample = freespaceMoved.ptr<float>(k);
            double d = sdf_ptr->interpolate(0, { sample[0], sample[1], sample[2] });
            if (d == std::numeric_limits<double>::max()) { continue; }
            if (d > thresholdlen) { verts_sdf_beyond(k) = 1; }
            //if (abs(d) <= thresholdlen) { verts_sdf_overlap(k) = 1; }
        }
        //cout << "verts_sdf_overlap " << verts_sdf_overlap.sum() << endl;
        //cout << "verts_sdf_beyond  "<< verts_sdf_beyond.sum() << endl;
        //double Overlap = verts_sdf_overlap.sum() - 0.1 * verts_sdf_beyond.sum();
        //double Overlap = verts_sdf_overlap.sum() ;
        //double Overlap = verts_sdf_overlap.sum() ;
        //resultsS[i]->updateOverlap(Overlap);
        double beyond = verts_sdf_beyond.sum();
        resultsS[i]->overlap -= 100* beyond;
        resultsS[i]->freespaceIntersec = beyond;


    }
}

void PPF3DDetector::sdfRatio(std::vector<Pose3DPtr>& resultsS, const double& th)
{
    double thresholdlen = th;

    int numPoints = downsample_scene_dense_refinement.rows;
    // sdf, inside is +, outside is -
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for (int i = 0; i < resultsS.size(); i++) {
        Eigen::VectorXi verts_sdf_overlap = MatrixXi::Zero(numPoints, 1);
        // transpose freespace
        Mat surfaceMoved = transformPCPose(downsample_scene_dense_refinement, resultsS[i]->pose.inv());

        for (int k = 0; k < numPoints; ++k)
        {
            float* sample = surfaceMoved.ptr<float>(k);
            double d = sdf_ptr->interpolate(0, { sample[0], sample[1], sample[2] });
            if (d == std::numeric_limits<double>::max()) { continue; }
            //if (d > thresholdlen) { verts_sdf_beyond(k) = 1; }
            if (abs(d) <= thresholdlen) { verts_sdf_overlap(k) = 1; }
        }
        double Overlap = verts_sdf_overlap.sum() ;
        resultsS[i]->updateOverlap(Overlap);
    }
}

/*
void  PPF3DDetector::SparseICP(std::vector<Pose3DPtr>& resultsS, Mat& srcPC, Mat&dstPC, const int ICP_nbIterations)
{

    double thresholdlen = 0.1 * model_diameter;

    int nbIterations = ICP_nbIterations; //30
    int nbIterationsIn = 3;
    double mu = 8;
    int nbIterShrink = 3;
    double p = 0.5;
    IcpMethod method = pointToPlane;
    bool verbose = false;
    size_t kNormals = 8;

#ifdef _OPENMP
#pragma omp parallel for 
#endif

    for (int i = 0; i < resultsS.size(); i++) {

        Mat pct_cv = transformPCPose(srcPC, resultsS[i]->pose);
        MatrixXf pct;
        cv::cv2eigen(pct_cv, pct);
        MatrixXf range(2, 3);//sampled_pc采样后的模型
        range(0, 0) = pct.col(0).minCoeff() - thresholdlen;
        range(0, 1) = pct.col(1).minCoeff() - thresholdlen;
        range(0, 2) = pct.col(2).minCoeff() - thresholdlen;
        range(1, 0) = pct.col(0).maxCoeff() + thresholdlen;
        range(1, 1) = pct.col(1).maxCoeff() + thresholdlen;
        range(1, 2) = pct.col(2).maxCoeff() + thresholdlen;
        //cout << range << endl;
        //cout << scene_range1.transpose() << endl.
        //cout << scene_range2.transpose() << endl;

        MatrixXf downsampleScene;
        cv::cv2eigen(dstPC, downsampleScene);
        MatrixXf modelmove(downsampleScene.rows(), 6);
        int size = 0;
        for (int j = 0; j < downsampleScene.rows(); j++)
        {
            float min = (downsampleScene.row(j).head(3) - range.row(0)).minCoeff();
            float max = (downsampleScene.row(j).head(3) - range.row(1)).maxCoeff();
            //cout << min << "  " << max << endl;
            if (min > 0 && max < 0)
            {
                modelmove.row(size) = downsampleScene.row(j);
                size++;
            }
        }
        //cout << sampled_pc.rows() <<" "<< size << endl;
        if (size > 100)
        {
            IcpOptimizer myIcpOptimizer(pct.cast<double>(), modelmove.block(0, 0, size, 6).cast<double>(), kNormals, nbIterations, nbIterationsIn, mu, nbIterShrink, p, method, verbose, 2);
            int hasIcpFailed = myIcpOptimizer.performSparceICP();
            if (hasIcpFailed)
            {
                cerr << "Failed to load the point clouds. Check the paths." << endl;
            }
            RigidTransfo resultingTransfo = myIcpOptimizer.getComputedTransfo();
            //cout << "Computed Rotation : " << endl << resultingTransfo.first << endl << "Computed Translation : " << endl << resultingTransfo.second << endl;
            MatrixXd P(3, 4);
            P << resultingTransfo.first, resultingTransfo.second;//横行向合并
            Matrix4d Pose;
            Pose << P,
                RowVector4d(0, 0, 0, 1);
            //cout << "Computed Rotation : " << endl << Pose << endl;
            //cout << "Rotation : " << endl << resultsS[i].Pose << endl;
            Matx44d pose;
            cv::eigen2cv(Pose, pose);
            resultsS[i]->pose = pose * resultsS[i]->pose;
        }

    }
}
*/
bool PPF3DDetector::matchPose(const Pose3D& sourcePose, const Pose3D& targetPose)
{
  // translational difference
  Vec3d dv = targetPose.t - sourcePose.t; 
  double dNorm = cv::norm(dv);

  const double phi = fabs ( sourcePose.angle - targetPose.angle ); // 有问题

  //Eigen::Matrix<double, 4, 4> sMatrix; 
  //cv::cv2eigen(sourcePose.pose, sMatrix); // cv::Mat 转换成 Eigen::Matrix
  //Eigen::Affine3f sPose;
  //sPose.matrix() = sMatrix.cast<float>();

  //Eigen::Matrix<double, 4, 4> tMatrix;
  //cv::cv2eigen(targetPose.pose, tMatrix); // cv::Mat 转换成 Eigen::Matrix
  //Eigen::Affine3f tPose;
  //tPose.matrix() = tMatrix.cast<float>();

  //Eigen::Matrix3f RtRsInv(sPose.rotation().inverse().lazyProduct(tPose.rotation()).eval());
  //Eigen::AngleAxisf rotation_diff_mat(RtRsInv); //tr(Rs.inv * Rt) = tr(Rt * Rs.inv)
  //const double phi = std::abs(rotation_diff_mat.angle());
  //
  bool clustering = (phi < this->rotation_threshold && dNorm < this->position_threshold);
  //if (clustering) {
  //    std::cout << "phi   " << phi << " rot-thshd " << this->rotation_threshold << " 小于阈值吗：" << (phi < this->rotation_threshold) << std::endl;
  //    std::cout << "dNorm " << dNorm << " pst-thshd " << this->position_threshold << " 小于阈值吗：" << (dNorm < this->position_threshold)  << std::endl << std::endl;
  //}
  return clustering;
}

void PPF3DDetector::clusterPoses(std::vector<Pose3DPtr>& poseList, int numPoses, std::vector<Pose3DPtr> &finalPoses)
{
  std::vector<PoseCluster3DPtr> poseClusters;

  finalPoses.clear();

  // sort the poses for stability
  //std::sort(poseList.begin(), poseList.end(), pose3DPtrCompare);

  for (int i=0; i<numPoses; i++)
  {
    Pose3DPtr pose = poseList[i];
    bool assigned = false;

    // search all clusters
    for (size_t j=0; j<poseClusters.size() && !assigned; j++)
    {
      const Pose3DPtr poseCenter = poseClusters[j]->poseList[0];
      if (matchPose(*pose, *poseCenter))
      {
        poseClusters[j]->addPose(pose);
        assigned = true;
      }
    }

    if (!assigned)
    {
      poseClusters.push_back(PoseCluster3DPtr(new PoseCluster3D(pose)));
    }
  }

  // sort the clusters so that we could output multiple hypothesis
  std::sort(poseClusters.begin(), poseClusters.end(), sortPoseClusters);

  finalPoses.resize(poseClusters.size());

  // TODO: Use MinMatchScore

  if (use_weighted_avg)
  {
#if defined _OPENMP
#pragma omp parallel for
#endif
    // uses weighting by the number of votes
    for (int i=0; i<static_cast<int>(poseClusters.size()); i++)
    {
      // We could only average the quaternions. So I will make use of them here
      Vec4d qAvg = Vec4d::all(0);
      Vec3d tAvg = Vec3d::all(0);

      // Perform the final averaging
      PoseCluster3DPtr curCluster = poseClusters[i];
      std::vector<Pose3DPtr> curPoses = curCluster->poseList;
      int curSize = (int)curPoses.size();
      size_t numTotalVotes = 0;

      for (int j=0; j<curSize; j++)
        numTotalVotes += curPoses[j]->numVotes;

      double wSum=0;

      for (int j=0; j<curSize; j++)
      {
        const double w = (double)curPoses[j]->numVotes / (double)numTotalVotes;

        qAvg += w * curPoses[j]->q;
        tAvg += w * curPoses[j]->t;
        wSum += w;
      }

      tAvg *= 1.0 / wSum;
      qAvg *= 1.0 / wSum;

      curPoses[0]->updatePoseQuat(qAvg, tAvg);
      curPoses[0]->numVotes=curCluster->numVotes;

      finalPoses[i]=curPoses[0]->clone();
    }
  }
  else
  {
#if defined _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<static_cast<int>(poseClusters.size()); i++)
    {
      // We could only average the quaternions. So I will make use of them here
      Vec4d qAvg = Vec4d::all(0);
      Vec3d tAvg = Vec3d::all(0);

      // Perform the final averaging
      PoseCluster3DPtr curCluster = poseClusters[i];
      std::vector<Pose3DPtr> curPoses = curCluster->poseList;
      const int curSize = (int)curPoses.size();

      //debug
      //if (curSize > 1) {
      //    std::cout << "cluster " << i << std::endl;
      //    for (Pose3DPtr Pose : curPoses) {
      //        std::cout << Pose->pose << std::endl;
      //    }
      //}

      for (int j=0; j<curSize; j++)
      {
        qAvg += curPoses[j]->q;
        tAvg += curPoses[j]->t;
      }

      tAvg *= 1.0 / curSize;
      qAvg *= 1.0 / curSize;
      // normalize quantonion
      const double qNorm = cv::norm(qAvg);
      if (qNorm > EPS)
      {
          qAvg *= 1.0 / qNorm;
      }

      curPoses[0]->updatePoseQuat(qAvg, tAvg);
      curPoses[0]->numVotes=curCluster->numVotes;

      //debug
      //if (curSize > 1) {
      //    std::cout << "cluster center:" << std::endl;
      //    std::cout << curPoses[0]->pose << std::endl;
      //    std::cout << std::endl;
      //}


      finalPoses[i]=curPoses[0]->clone();
    }
  }

  poseClusters.clear();
}

bool PPF3DDetector::matchPoseInR3(const Pose3D& sourcePose, const Pose3D& targetPose)
{
    Mat tpl_s = transformPCPose(model_triplet, sourcePose.pose);
    Mat tpl_t = transformPCPose(model_triplet, targetPose.pose);
    double th = clusterThreshold * model_diameter;
    bool clustering  = true;
    for (int ii = 0; ii < tpl_s.rows; ii++)
    {
        Vec3f v1(tpl_s.ptr<float>(ii));
        //const Vec3f n1(pct_gt.ptr<float>(ii) + 3);
        Vec3f v2(tpl_t.ptr<float>(ii));
        v1 = v1 - v2;
        if (cv::norm(v1) > th) {
            clustering = false;
        }
    }
    return clustering;
}


void PPF3DDetector::clusterPosesInR3(std::vector<Pose3DPtr>& poseList, int numPoses, std::vector<Pose3DPtr>& finalPoses)
{
    std::vector<PoseCluster3DPtr> poseClusters;

    finalPoses.clear();

    vector<Mat> tripletTransformed(numPoses);

#if defined _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < numPoses; i++) {
        tripletTransformed[i] = transformPCPose(model_triplet, poseList[i]->pose);
    }


    for (int i = 0; i < numPoses; i++)
    {
        Pose3DPtr pose = poseList[i];
        bool assigned = false;

        // search all clusters
        for (size_t j = 0; j < poseClusters.size() && !assigned; j++)
        {
            const Pose3DPtr poseCenter = poseClusters[j]->poseList[0];
            if (matchPoseInR3(*pose, *poseCenter))
            {
                poseClusters[j]->addPose(pose);
                assigned = true;
            }
        }

        if (!assigned)
        {
            poseClusters.push_back(PoseCluster3DPtr(new PoseCluster3D(pose)));
        }
    }

    // sort the clusters so that we could output multiple hypothesis
    std::sort(poseClusters.begin(), poseClusters.end(), sortPoseClusters);

    finalPoses.resize(poseClusters.size());

    // TODO: Use MinMatchScore


#if defined _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < static_cast<int>(poseClusters.size()); i++)
    {
        // We could only average the quaternions. So I will make use of them here
        Vec4d qAvg = Vec4d::all(0);
        Vec3d tAvg = Vec3d::all(0);

        // Perform the final averaging
        PoseCluster3DPtr curCluster = poseClusters[i];
        std::vector<Pose3DPtr> curPoses = curCluster->poseList;
        const int curSize = (int)curPoses.size();

        //debug
        //if (curSize > 1) {
        //    std::cout << "cluster " << i << std::endl;
        //    for (Pose3DPtr Pose : curPoses) {
        //        std::cout << Pose->pose << std::endl;
        //    }
        //}

        for (int j = 0; j < curSize; j++)
        {
            qAvg += curPoses[j]->q;
            tAvg += curPoses[j]->t;
        }

        tAvg *= 1.0 / curSize;
        qAvg *= 1.0 / curSize;
        // normalize quantonion
        const double qNorm = cv::norm(qAvg);
        if (qNorm > EPS)
        {
            qAvg *= 1.0 / qNorm;
        }

        curPoses[0]->updatePoseQuat(qAvg, tAvg);
        curPoses[0]->numVotes = curCluster->numVotes;

        //debug
        //if (curSize > 1) {
        //    std::cout << "cluster center:" << std::endl;
        //    std::cout << curPoses[0]->pose << std::endl;
        //    std::cout << std::endl;
        //}


        finalPoses[i] = curPoses[0]->clone();
    }

    poseClusters.clear();
}

//https://zhuanlan.zhihu.com/p/78504109
//https://blog.csdn.net/m0_45388819/article/details/117217322
//https://blog.csdn.net/qq_31638535/article/details/143980041
void PPF3DDetector::NMScenter(std::vector<Pose3DPtr>& poseList, double Threshold, std::vector<Pose3DPtr>& finalPoses)
{
    finalPoses.clear();

    int i, j;
    int numInputPose = poseList.size();
    vector<bool> is_suppressed(numInputPose);
    //vector<map<string, Vec2f> > bboxList(numInputPose);
    //vector<float> bboxVolumes(numInputPose);
    vector<Mat> centerTransformed(numInputPose);

#if defined _OPENMP
#pragma omp parallel for
#endif

    for (i = 0; i < numInputPose; i++) {
        is_suppressed[i] = 0;

        //Mat pcTemp = transformPCPose(sampled_pc, poseList[i]->pose);
        //Vec2f xRange, yRange, zRange;
        //computeBboxStd(pcTemp, xRange, yRange, zRange);
        //bboxList[i]["xRange"] = xRange;
        //bboxList[i]["yRange"] = yRange;
        //bboxList[i]["zRange"] = zRange;

        //float dx = xRange[1] - xRange[0];
        //float dy = yRange[1] - yRange[0];
        //float dz = zRange[1] - zRange[0];
        //bboxVolumes[i] = dx * dy * dz;

        Mat mc = Mat(model_center).reshape(1, 1);
        Mat ctT = transformPCPose(mc, poseList[i]->pose);
        centerTransformed[i] = ctT;
        //cout << ctT.row(0) << endl;

    }

    double th = Threshold * model_diameter;
    //double th = Threshold * model_minimum_length;

    for (i = 0; i < numInputPose; i++)                // 循环所有窗口   
    {
        if (!is_suppressed[i])           // 判断窗口是否被抑制   
        {
            /*map<string, Vec2f>& bboxKeeped = bboxList[i];
            float& bboxKeepedVolume = bboxVolumes[i];*/
            Mat& centerKeeped = centerTransformed[i];
            //cout << centerKeeped.row(0) << endl;

            for (j = i + 1; j < numInputPose; j++)
            {
                if (!is_suppressed[j])   // 判断窗口是否被抑制   
                {
                    Mat& centerCurrent = centerTransformed[j];
                    //cout << centerCurrent.row(0) << endl;

                    double centerDistance = cv::norm(centerCurrent.row(0) - centerKeeped.row(0));
                    if (centerDistance < th) //0.5
                    {
                        is_suppressed[j] = 1;           // 将窗口j标记为抑制  
                    }

                    //map<string, Vec2f>& bboxCurrent= bboxList[j];

                    //float inter_x = min(bboxCurrent["xRange"][1], bboxKeeped["xRange"][1]) - max(bboxCurrent["xRange"][0], bboxKeeped["xRange"][0]);
                    //float inter_y = min(bboxCurrent["yRange"][1], bboxKeeped["yRange"][1]) - max(bboxCurrent["yRange"][0], bboxKeeped["yRange"][0]);
                    //float inter_z = min(bboxCurrent["zRange"][1], bboxKeeped["zRange"][1]) - max(bboxCurrent["zRange"][0], bboxKeeped["zRange"][0]);

                    //if (inter_x > 0 && inter_y > 0 && inter_z > 0)
                    //{
                    //    float inter = inter_x * inter_y * inter_z;
                    //    float& bboxCurrentVolume = bboxVolumes[j];
                    //    float Union = bboxKeepedVolume + bboxCurrentVolume - inter;
                    //    float iou = inter / Union;
                    //    if (iou > Threshold)          // 判断重叠比率是否超过重叠阈值   0.4
                    //    {
                    //        is_suppressed[j] = 1;           // 将窗口j标记为抑制   
                    //    }
                    //}
                }
            }
        }
    }

    for (i = 0; i < numInputPose; i++)                  // 遍历所有输入窗口   
    {
        if (!is_suppressed[i])             // 将未发生抑制的窗口信息保存到输出信息中   
        {
            finalPoses.push_back(poseList[i]->clone()); //clone overlap
        }
    }

}


void PPF3DDetector::avg_quaternion_markley(Vec4d& qs_avg_cv, std::vector< Vec4d> qs_cv) {


    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    int M = qs_cv.size();
    for (int i = 0; i < M; i++) {
        //Eigen::Vector4f qs_temp = qs[i];
        Eigen::Vector4d qs_temp;
        cv::cv2eigen(qs_cv[i], qs_temp);
        A = qs_temp * qs_temp.transpose() + A;
    }
    A = ((float)1.0 / (float)M) * A;

    Eigen::EigenSolver<Eigen::Matrix4d> es(A);
    Eigen::MatrixXcd evecs = es.eigenvectors();//获取矩阵特征向量4*4，这里定义的MatrixXcd必须有c，表示获得的是complex复数矩阵
    Eigen::MatrixXcd evals = es.eigenvalues();//获取矩阵特征值 4*1
    Eigen::MatrixXd evalsReal;//注意这里定义的MatrixXd里没有c
    evalsReal = evals.real();//获取特征值实数部分
    Eigen::MatrixXd::Index evalsMax;
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);//得到最大特征值的位置
    Eigen::Vector4d q;
    q << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax), evecs.real()(3, evalsMax);//得到对应特征向量
    Eigen::Vector4d qs_avg = q;
    cv::eigen2cv(qs_avg, qs_avg_cv);

}

void PPF3DDetector::NMSbbox(std::vector<Pose3DPtr>& poseList, double Threshold, std::vector<Pose3DPtr>& finalPoses)
{
    finalPoses.clear();

    int i, j;
    int numInputPose = poseList.size();
    vector<bool> is_suppressed(numInputPose);
    vector<Mat> bboxTransformed(numInputPose);

#if defined _OPENMP
#pragma omp parallel for
#endif
    for (i = 0; i < numInputPose; i++) {
        is_suppressed[i] = 0;
        bboxTransformed[i] = transformPCPose(model_bbox, poseList[i]->pose);
    }

    double th = Threshold * model_diameter;

    for (i = 0; i < numInputPose; i++)                // 循环所有窗口   
    {
        if (!is_suppressed[i])           // 判断窗口是否被抑制   
        {

            Mat& bboxKeeped = bboxTransformed[i];
            poseList[i]->cluster_count = 1;
            poseList[i]->qs.push_back(poseList[i]->q);

            for (j = i + 1; j < numInputPose; j++)
            {
                if (!is_suppressed[j])   // 判断窗口是否被抑制   
                {
                    Mat& bboxCurrent = bboxTransformed[j];

                    //double totalD_ADD = 0;
                    double max_d = 0;
                    for (int ii = 0; ii < bboxKeeped.rows; ii++)
                    {
                        Vec3f v1(bboxKeeped.ptr<float>(ii));
                        //const Vec3f n1(pct_gt.ptr<float>(ii) + 3);
                        Vec3f v2(bboxCurrent.ptr<float>(ii));
                        v1 = v1 - v2;
                        double tmp_d = cv::norm(v1);
                        if (tmp_d > max_d) max_d = tmp_d;
                        //totalD_ADD += cv::norm(v1);
                    }
                    //totalD_ADD /= bboxKeeped.rows;

                    if (max_d < th) //0.5
                    {
                        is_suppressed[j] = 1;           // 将窗口j标记为抑制  
                        poseList[i]->numVotes += poseList[j]->numVotes;
                        //poseList[i]->q += poseList[j]->q;
                        poseList[i]->t += poseList[j]->t;
                        poseList[i]->cluster_count += 1;
                        poseList[i]->qs.push_back(poseList[j]->q);
                    }

                }

            }
        }
    }

    for (i = 0; i < numInputPose; i++)                  // 遍历所有输入窗口   
    {
        if (!is_suppressed[i])             // 将未发生抑制的窗口信息保存到输出信息中   
        {
            poseList[i]->t *= 1.0 / poseList[i]->cluster_count;
            //poseList[i]->q *= 1.0 / poseList[i]->cluster_count;
            //// normalize quantonion
            //const double qNorm = cv::norm(poseList[i]->q);
            //if (qNorm > EPS)
            //{
            //    poseList[i]->q *= 1.0 / qNorm;
            //}
            Vec4d qs_avg;
            avg_quaternion_markley(qs_avg, poseList[i]->qs);
            poseList[i]->q = qs_avg;

            finalPoses.push_back(poseList[i]->clone()); //clone overlap
        }
    }

}


//
//int nonMaximumSuppression(int numBoxes, const CvPoint* points,
//    const CvPoint* oppositePoints, const float* score,
//    float overlapThreshold,
//    int* numBoxesOut, CvPoint** pointsOut,
//    CvPoint** oppositePointsOut, float** scoreOut)
//{
//
//    // numBoxes：窗口数目// points：窗口左上角坐标点// oppositePoints：窗口右下角坐标点  
//    // score：窗口得分// overlapThreshold：重叠阈值控制// numBoxesOut：输出窗口数目  
//    // pointsOut：输出窗口左上角坐标点// oppositePoints：输出窗口右下角坐标点  
//    // scoreOut：输出窗口得分  
//    int i, j, index;
//    float* box_area = (float*)malloc(numBoxes * sizeof(float));    // 定义窗口面积变量并分配空间   
//    int* indices = (int*)malloc(numBoxes * sizeof(int));          // 定义窗口索引并分配空间   
//    int* is_suppressed = (int*)malloc(numBoxes * sizeof(int));    // 定义是否抑制表标志并分配空间   
//    // 初始化indices、is_supperssed、box_area信息   
//    for (i = 0; i < numBoxes; i++)
//    {
//        indices[i] = i;
//        is_suppressed[i] = 0;
//        box_area[i] = (float)((oppositePoints[i].x - points[i].x + 1) *
//            (oppositePoints[i].y - points[i].y + 1));
//    }
//    // 对输入窗口按照分数比值进行排序，排序后的编号放在indices中   
//    sort(numBoxes, score, indices);
//    for (i = 0; i < numBoxes; i++)                // 循环所有窗口   
//    {
//        if (!is_suppressed[indices[i]])           // 判断窗口是否被抑制   
//        {
//            for (j = i + 1; j < numBoxes; j++)    // 循环当前窗口之后的窗口   
//            {
//                if (!is_suppressed[indices[j]])   // 判断窗口是否被抑制   
//                {
//                    int x1max = max(points[indices[i]].x, points[indices[j]].x);                     // 求两个窗口左上角x坐标最大值   
//                    int x2min = min(oppositePoints[indices[i]].x, oppositePoints[indices[j]].x);     // 求两个窗口右下角x坐标最小值   
//                    int y1max = max(points[indices[i]].y, points[indices[j]].y);                     // 求两个窗口左上角y坐标最大值   
//                    int y2min = min(oppositePoints[indices[i]].y, oppositePoints[indices[j]].y);     // 求两个窗口右下角y坐标最小值   
//                    int overlapWidth = x2min - x1max + 1;            // 计算两矩形重叠的宽度   
//                    int overlapHeight = y2min - y1max + 1;           // 计算两矩形重叠的高度   
//                    if (overlapWidth > 0 && overlapHeight > 0)
//                    {
//                        float overlapPart = (overlapWidth * overlapHeight) / box_area[indices[j]];    // 计算重叠的比率   
//                        if (overlapPart > overlapThreshold)          // 判断重叠比率是否超过重叠阈值   
//                        {
//                            is_suppressed[indices[j]] = 1;           // 将窗口j标记为抑制   
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    *numBoxesOut = 0;    // 初始化输出窗口数目0   
//    for (i = 0; i < numBoxes; i++)
//    {
//        if (!is_suppressed[i]) (*numBoxesOut)++;    // 统计输出窗口数目   
//    }
//
//    *pointsOut = (CvPoint*)malloc((*numBoxesOut) * sizeof(CvPoint));           // 分配输出窗口左上角坐标空间   
//    *oppositePointsOut = (CvPoint*)malloc((*numBoxesOut) * sizeof(CvPoint));   // 分配输出窗口右下角坐标空间   
//    *scoreOut = (float*)malloc((*numBoxesOut) * sizeof(float));                // 分配输出窗口得分空间   
//    index = 0;
//    for (i = 0; i < numBoxes; i++)                  // 遍历所有输入窗口   
//    {
//        if (!is_suppressed[indices[i]])             // 将未发生抑制的窗口信息保存到输出信息中   
//        {
//            (*pointsOut)[index].x = points[indices[i]].x;
//            (*pointsOut)[index].y = points[indices[i]].y;
//            (*oppositePointsOut)[index].x = oppositePoints[indices[i]].x;
//            (*oppositePointsOut)[index].y = oppositePoints[indices[i]].y;
//            (*scoreOut)[index] = score[indices[i]];
//            index++;
//        }
//
//    }
//
//    free(indices);          // 释放indices空间   
//    free(box_area);         // 释放box_area空间   
//    free(is_suppressed);    // 释放is_suppressed空间   
//
//    return LATENT_SVM_OK;
//}

Mat PPF3DDetector::SubtractPlane(const Mat& scenePC, float dist) {
    // convert cv mat to pcl 
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn(new pcl::PointCloud<pcl::PointNormal>());

    cloud_pn->resize(scenePC.rows);
//#if defined _OPENMP
//#pragma omp parallel for
//#endif
    for (int k = 0; k < scenePC.rows; k++)
    {
        const float* data = scenePC.ptr<float>(k);
        pcl::PointNormal pt;
        pt.x = data[0];
        pt.y = data[1];
        pt.z = data[2];
        pt.normal_x = data[3];
        pt.normal_y = data[4];
        pt.normal_z = data[5];
        cloud_pn->at(k) = pt;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud_pn, *cloud_in);

    pcl::SACSegmentation<pcl::PointXYZ> seg(true);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC); //SAC_RANSAC
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(dist);

    pcl::ExtractIndices<pcl::PointNormal> extract;

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients coeff;

    pcl::PointCloud<pcl::PointNormal>::Ptr plane(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr keeped(new pcl::PointCloud<pcl::PointNormal>);
    seg.setInputCloud(cloud_in);
    seg.segment(*inliers, coeff);

    extract.setInputCloud(cloud_pn);
    extract.setIndices(inliers);
    //extract.setNegative(false);
    //extract.filter(*plane);
    extract.setNegative(true);
    extract.filter(*keeped);

    cv::Mat keeped_cv = cv::Mat(keeped->size(), 6, CV_32FC1);
//#if defined _OPENMP
//#pragma omp parallel for
//#endif
    for (int i = 0; i < keeped->size(); i++)
    {
        float* data = keeped_cv.ptr<float>(i);
        data[0] = keeped->at(i).x;
        data[1] = keeped->at(i).y;
        data[2] = keeped->at(i).z;
        data[3] = keeped->at(i).normal_x;
        data[4] = keeped->at(i).normal_y;
        data[5] = keeped->at(i).normal_z;
    }

    return keeped_cv;
}

void PPF3DDetector::match(const Mat& pc, std::vector<Pose3DPtr>& results, const double relativeSceneSampleStep, const double relativeSceneDistance)
{
  if (!trained)
  {
    throw cv::Exception(cv::Error::StsError, "The model is not trained. Cannot match without training", __FUNCTION__, __FILE__, __LINE__);
  }

  CV_Assert(pc.type() == CV_32F || pc.type() == CV_32FC1);
  CV_Assert(relativeSceneSampleStep<=1 && relativeSceneSampleStep>0);

  scene_sample_step = (int)(1.0/relativeSceneSampleStep);

  //int numNeighbors = 10;
  int numAngles = (int) (floor (2 * M_PI / angle_step));
  float distanceStep = (float)distance_step;
  //float distanceStepFeature = (float)(model_diameter * distance_step_relative);
  uint n = num_ref_points;
  std::vector<Pose3DPtr> poseList;
  int sceneSamplingStep = scene_sample_step;

  // 计算包围框
  Vec2f xRange, yRange, zRange;
  computeBboxStd(pc, xRange, yRange, zRange);

  // 场景点云采样
  /*float dx = xRange[1] - xRange[0];
  float dy = yRange[1] - yRange[0];
  float dz = zRange[1] - zRange[0];
  float diameter = sqrt ( dx * dx + dy * dy + dz * dz );
  float distanceSampleStep = diameter * RelativeSceneDistance;*/
  //Mat sampled = samplePCByQuantization(pc, xRange, yRange, zRange, (float)relativeSceneDistance, 0);

  float distanceSampleStep = relativeSceneDistance * model_diameter;
  float distanceSampleStepRefine = distanceSampleStep / 4;
  Mat sampled, sampled_for_freespace, sampledDenseRefinement, sampledSparseRefinement, pcRemained;
  sampled_for_freespace = samplePCByQuantization_cube(pc, xRange, yRange, zRange, distanceSampleStep, 0);
  sampledDenseRefinement = samplePCUniform(pc, 2);
  //sampledSparseRefinement = samplePCUniform(pc, 4);
  //sampledSparseRefinement = samplePCByQuantization_cube(pc, xRange, yRange, zRange, distanceSampleStepRefine, 0);

  pcRemained = SubtractPlane(pc, distanceSampleStep * 4);
  if (samplingMethod == "cube") {
      //sampled = samplePCByQuantization_cube(pc, xRange, yRange, zRange, distanceSampleStep, 0);
      sampled = samplePCByQuantization_cube(pcRemained, xRange, yRange, zRange, distanceSampleStep, 0);
  }
  else if (samplingMethod == "normal") {
      //sampled = samplePCByQuantization_normal(pc, xRange, yRange, zRange, distanceSampleStep, 15.0 / 180 * M_PI, 3);
      sampled = samplePCByQuantization_normal(pcRemained, xRange, yRange, zRange, distanceSampleStep, 15.0 / 180 * M_PI, 3);
  }
  //writePLY(sampled, "sampled_scene.ply");

  //Mat sampledDenseRefinement= samplePCUniform(pc, 8);
  relative_scene_distance = relativeSceneDistance;
  //downsample_scene_sparse_refinement = sampled;
  //downsample_scene_sparse_refinement_flannIndex = indexPCFlann(downsample_scene_sparse_refinement);
  downsample_scene_dense_refinement = sampledDenseRefinement;
  downsample_scene_dense_refinement_flannIndex = indexPCFlann(downsample_scene_dense_refinement);
  
  if(useFreeSpaceConstraint) generateFreeSpaceVolume(sampled_for_freespace, (float)distance_step);  // set downsample_scene_freespace

  // 将参考点或者说关键点、采样的场景点云存下来，用来debug
  if (debug) {
      Mat sampled_ref0 = Mat((sampled.rows / sceneSamplingStep) + 1, pc.cols, CV_32F);
      int c = 0;
      for (int i = 0; i < sampled.rows; i += sceneSamplingStep)
      {
          sampled.row(i).copyTo(sampled_ref0.row(c));
          c += 1;
      }
      Mat sampled_ref;
      sampled_ref0.rowRange(0, c).copyTo(sampled_ref);
      int refPointNum = c;

      std::cout << "参考点数：" << refPointNum << std::endl;
      string name1 = "../samples/data/results/" + debug_folder_name + "/sampled_scene.ply";
      string name2 = "../samples/data/results/" + debug_folder_name + "/sampled_scene_ref.ply";
      string name3 = "../samples/data/results/" + debug_folder_name + "/pcRemained.ply";
      string name4 = "../samples/data/results/" + debug_folder_name + "/sampled_scene_dense.ply";
      string name5 = "../samples/data/results/" + debug_folder_name + "/sampled_scene_sparse.ply";
      writePLY(sampled, name1.c_str());
      writePLY(sampled_ref, name2.c_str());
      writePLY(pcRemained, name3.c_str()); 
      writePLY(sampledDenseRefinement, name4.c_str());
      //writePLY(sampledSparseRefinement, name5.c_str());
  }

  // radius tree
  Mat dest_32f;
  sampled.colRange(0, 3).copyTo(dest_32f);
  cv::flann::Index downsampleSceneRadiusTree(dest_32f, cv::flann::KDTreeIndexParams(1));
  unsigned int max_neighbours = num_ref_points * 2;
  double model_max_dist = model_diameter * model_diameter;

  // 创建一个列表，用来存预测的位姿，这些位姿被算法认为是Positive的
  poseList.reserve((sampled.rows / sceneSamplingStep) +4);

#if defined _OPENMP
#pragma omp parallel for
#endif
  // 遍历所有的参考点 sr
  for (int i = 0; i < sampled.rows; i += sceneSamplingStep)
  {

    uint refIndMax = 0, alphaIndMax = 0;
    uint maxVotes = 0;

    //给定一个参考点sr
    const Vec3f p1(sampled.ptr<float>(i));
    const Vec3f n1(sampled.ptr<float>(i) + 3);
    // 计算将sr、n_sr和原点、x正方向对齐所需的变换Rsg, tsg
    Vec3d tsg = Vec3d::all(0);
    Matx33d Rsg = Matx33d::all(0), RInv = Matx33d::all(0);
    computeTransformRT(p1, n1, Rsg, tsg);
    // 创建一个累加器，用来存hough voting的votes，矩阵：[30*|M|, 1]或者说[|M|, 30]，其中|M|是模型采样点数
    uint* accumulator = (uint*)calloc(numAngles * n, sizeof(uint));

    // saveVoters
    vector<Mat> coordAccumulator;
    if (debug) coordAccumulator.resize(numAngles * n);

    std::vector<float> vecQuery{ p1[0], p1[1], p1[2] };
    std::vector<int> vecIndex;
    std::vector<float> vecDist;
    downsampleSceneRadiusTree.radiusSearch(vecQuery, vecIndex, vecDist, model_max_dist, max_neighbours, cv::flann::SearchParams(max_neighbours));

    // Tolga Birdal's notice:
    // As a later update, we might want to look into a local neighborhood only
    // To do this, simply search the local neighborhood by radius look up
    // and collect the neighbors to compute the relative pose

    // 遍历所有的场景点si
    //for (int j = 0; j < sampled.rows; j ++)
    //{
    //  if (i!=j)
    //  {
    //    // 对于一个si，它和sr构成点对
    //    const Vec3f p2(sampled.ptr<float>(j));
    //    const Vec3f n2(sampled.ptr<float>(j) + 3);

    for (int j = 1; j < vecIndex.size(); j++)
    {
      int pInd = vecIndex[j];
      if ((pInd == 0) && (vecIndex[j - 1] == 0))
      {
            break;
      }
      if (i != pInd)
      {
        const Vec3f p2(sampled.ptr<float>(pInd));
        const Vec3f n2(sampled.ptr<float>(pInd) + 3);

        // 计算 F(sr, si)，离散化之后计算哈希值
        Vec4d f1_4 = Vec4d::all(0);
        computePPFFeatures(p1, n1, p2, n2, f1_4);
        /**** 计算 theta_s ****/
        double f5theta = computeTheta(p1, n1, p2, n2);

        KeyType hashValue = hashPPF(f1_4, angle_step, distanceStep);

        //Vec<int, 5> fd;
        //KeyType hashValue = hashPPF3(f1_4, f5theta, fd, angle_step, distanceStep, orientation_diff_threshold);

        // 计算 alpha_s
        Vec3d p2t;
        double alpha_scene;
        p2t = tsg + Rsg * Vec3d(p2);
        alpha_scene=atan2(-p2t[2], p2t[1]);
        if ( alpha_scene != alpha_scene)
        {
          continue;
        }
        if (sin(alpha_scene)*p2t[2]<0.0)
          alpha_scene=-alpha_scene;
        alpha_scene=-alpha_scene;


        // 根据哈希值索引具有相似特征的模型点对(mr, mi)，因为离散化，F(sr, si) 约等于 F(mr, mi)，这有助于抵抗噪声
        // 得到模型点对集合 A = {(mr, mi), (mr', mi'), ...}
        hashnode_i* node = hashtableGetBucketHashed(hash_table, (hashValue));
        // 遍历模型点对集合 A
        while (node)
        {
            // 对于匹配到的一个模型点对(mr, mi)
            THash* tData = (THash*) node->data;

            //if (!(tData->fd == fd)) { 
            //    node = node->next; 
            //    continue; 
            //}

            // mr的下标r, r属于[0, |M|-1]
            int corrI = (int)tData->i;
            // 模型点对(mr, mi)的特征在矩阵ppf中的位置
            int ppfInd = (int)tData->ppfInd;
            float* ppfCorrScene = ppf.ptr<float>(ppfInd);
            
            /**** 比较theta_m, theta_s，若 theta_s - theta_m \in [-10deg, +10deg]，则认为他们匹配 ****/
            //double theta_m = (double)ppfCorrScene[PPF_LENGTH - 2];
            //double delta_theta = f5theta - theta_m;
            //////double angle_m = theta_m / M_PI * 180;
            //////double angle_s = f5theta / M_PI * 180;
            //////double angle_delta = delta_theta / M_PI * 180;
            //////int bin_m = (int)(theta_m / orientation_diff_threshold);
            //////int bin_s = (int)(f5theta / orientation_diff_threshold);
            //////if (bin_m != bin_s) {
            //////    node = node->next;
            //////    continue;
            //////}
            //if (delta_theta < -orientation_diff_threshold || delta_theta > orientation_diff_threshold) {
            //    //cout << delta_theta / M_PI * 180 << endl;
            //    node = node->next;
            //    continue;
            //}


            // 取出(mr, mi)对应的alpha_model
            double alpha_model = (double)ppfCorrScene[PPF_LENGTH - 1];
            // 将(sr, n_sr), (mr, n_mr)和(O, +x)对齐之后，再把mi绕x轴旋转到si所需的角度alpha，
            double alpha = alpha_model - alpha_scene;

            /*  Tolga Birdal's note: Map alpha to the indices:
                    atan2 generates results in (-pi pi]
                    That's why alpha should be in range [-2pi 2pi]
                    So the quantization would be :
                    numAngles * (alpha+2pi)/(4pi)
                    */                    //printf("%f\n", alpha);

            // 将[-2pi, 2pi]分成30个区间，计算alpha在第几个区间
            int alpha_index = (int)(numAngles * (alpha + 2 * M_PI) / (4 * M_PI));
            /* 在累加器矩阵[| M | , 30]的(r, alpha_index)处 + 1，
            
            也就是，将(sr, si)和(mr, mi)的这次匹配记录下来，顺便将这两个有向point pair对齐所需的变换角alpha也记录了下来。
            注意，点对特征 F(mr, mi) 是非对称的，mr是模型中的关键点，这里累加器矩阵只关注关键点mr。
            或者说，累加器矩阵只记录了场景关键点sr和模型的哪个关键点mr是匹配的，匹配的时候两个全局描述符中有多少邻域点是重合的。
            
            有向点拥有法向，有向点的两组匹配关系可以确定一个刚体变换。下面用集合的语言说。
            
            使用两点hough投票做误匹配去除。这是在求解一个优化问题，即
            
            
            初步的做法是，由点描述符匹配生成一组domain到codomain的点映射，即点匹配关系。我们要去除其中的错误匹配。
            我们在domain穷举抽样两点，两点对应domain到codomain的两个匹配关系，拟合一个刚体变换。
            然后验证这个刚体变换是否能让两点坐标、法向重合，即是否能保持距离和法向。不能，则重新抽样两点。若能，则计算这个刚体变换
            在离散的刚体变换空间中的网格索引，让这个网格中的数加一。
            
            巧妙的做法是，通过匹配domain上的点对特征和codomain上的点对特征，得到两个匹配关系，他们能保持距离、法向、弯曲程度。
            我们在domain上取一个参考点sr，围绕这个参考点遍历抽样点对，计算点对特征，通过匹配这些点对特征生成点映射。
            （若参考点sr在domain的物体ROI上，生成的点映射在domain的物体ROI上一定是局部单射。）
            若参考点sr在domain的物体ROI上，因为点对特征的重复性很好，
            所以生成的点映射一定包含所有的正确点映射，他们从domain的物体ROI映到domain的物体ROI上。
            当然也会包含一些多余的错的点映射，因为点对特征区分能力有限。
            接着对这些遍历抽样的所有点对，根据匹配关系计算刚体变换，然后在离散刚体变换空间做投票。
            
            而这种在domain上围绕一个参考点sr遍历抽样点对的做法，使得一组domain到codomain的点映射拥有一个相同的domain参考点sr。
            若他们满足一个相同的刚体变换，则等价于他们拥有一个相同的codomain参考点mr和旋转角alpha。
            这种等价让hough投票更容易。因为，刚体变换空间是6维的，而(mr, alpha)所属的空间 M X [-2pi, 2pi]是二维的。
            如果假设刚体变换空间每一维离散成30个区间，那么一共有30^6=729000000个区间。
            而 M X [-2pi, 2pi]，M是模型曲面流形，它可以离散成1000*30=30,000个区间，空间大大减少。
            
            若在domain的物体ROI上取第二个参考点，再围绕这个参考点抽样点对，那么这些点对是新的，对比两点hough投票的穷举做法，可以发现
            这不过是向穷举抽样两点靠近一小步罢了。这一次同样也生成了所有正确的点匹配和一些错误匹配，所以这一次只是让正确刚体变换和错误刚体变换的
            得票数都翻一倍罢了。得票数就是拥有一致的刚体变换的匹配关系的个数。
            
            注意，将刚体变换空间离散成网格，再计算一个刚体变换的网格索引，让这个网格中的数加一，这其实是在离散的流形上做聚类，
            因为流形距离小的两个刚体变换才能放到一个网格中。两个参考点分别hough投票，得到离散刚体变换空间中的两个票数极大值点，
            如果极大值点相同，就将该网格处的票数值相加。这其实对应了ppf最后对刚体变换做聚类的环节，区别在于，他是在连续的流形上聚类。
            */
            uint accIndex = corrI * numAngles + alpha_index;
            accumulator[accIndex]++;

            // saveVoters
            if (spectral) {
                Mat corres = (cv::Mat_<int>(1, 3) << j, ppfInd, corrI);
                coordAccumulator[accIndex].push_back(corres);
            }
            
            node = node->next;
        }
      }
    }

     
    // the paper says: "For stability reasons, all peaks that received a certain amount
    // of votes relative to the maximum peak are used." No specific value is mentioned,
    // but 90% seems good

    for (uint k = 0; k < n; k++)
    {
        for (int j = 0; j < numAngles; j++)
        {
            const uint accInd = k * numAngles + j;
            const uint accVal = accumulator[accInd];
            if (accVal > maxVotes)
                maxVotes = accVal;
        }
    }

    // visualize accumulator
    //Mat accum = cv::Mat(numAngles * n, 1, CV_8U, accumulator) * (255./maxVotes);
    //cv::Mat result1 = accum.reshape(0, n);
    //if (maxVotes < 50) continue;
    //if (maxVotes < 1) {
    /*if (0) {
        const uint accVal = 0;
        refIndMax = 0;
        alphaIndMax = 0;

        // invert Tsg : Luckily rotation is orthogonal: Inverse = Transpose.
        // We are not required to invert.
        Vec3d tInv, tmg;
        Matx33d Rmg;
        RInv = Rsg.t();
        tInv = -RInv * tsg;

        Matx44d TsgInv;
        rtToPose(RInv, tInv, TsgInv);

        // TODO : Compute pose
        const Vec3f pMax(sampled_pc.ptr<float>(refIndMax));
        const Vec3f nMax(sampled_pc.ptr<float>(refIndMax) + 3);

        computeTransformRT(pMax, nMax, Rmg, tmg);

        Matx44d Tmg;
        rtToPose(Rmg, tmg, Tmg);

        // convert alpha_index to alpha
        int alpha_index = alphaIndMax;
        double alpha = (alpha_index * (4 * M_PI)) / numAngles - 2 * M_PI;

        // Equation 2:
        Matx44d Talpha;
        Matx33d R;
        Vec3d t = Vec3d::all(0);
        getUnitXRotation(alpha, R);
        rtToPose(R, t, Talpha);

        Matx44d rawPose = TsgInv * (Talpha * Tmg);

        Pose3DPtr pose(new Pose3D(alpha, refIndMax, accVal));  //modelIndex
        pose->updatePose(rawPose);

        #if defined (_OPENMP)
        #pragma omp critical
        #endif
        {
            poseList.push_back(pose);
        }
        free(accumulator);
        continue;
    } */

    float vote_th = maxVotes * 0.95;
    for (uint k = 0; k < n; k++)
    {
        for (int j = 0; j < numAngles; j++)
        {
            const uint accInd = k * numAngles + j;
            const uint accVal = accumulator[accInd];
            if (accVal >= vote_th)
            {
                refIndMax = k;
                alphaIndMax = j;

                // invert Tsg : Luckily rotation is orthogonal: Inverse = Transpose.
                // We are not required to invert.
                Vec3d tInv, tmg;
                Matx33d Rmg;
                RInv = Rsg.t();
                tInv = -RInv * tsg;

                Matx44d TsgInv;
                rtToPose(RInv, tInv, TsgInv);

                // TODO : Compute pose
                const Vec3f pMax(sampled_pc.ptr<float>(refIndMax));
                const Vec3f nMax(sampled_pc.ptr<float>(refIndMax) + 3);

                computeTransformRT(pMax, nMax, Rmg, tmg);

                Matx44d Tmg;
                rtToPose(Rmg, tmg, Tmg);

                // convert alpha_index to alpha
                int alpha_index = alphaIndMax;
                double alpha = (alpha_index * (4 * M_PI)) / numAngles - 2 * M_PI;

                // Equation 2:
                Matx44d Talpha;
                Matx33d R;
                Vec3d t = Vec3d::all(0);
                getUnitXRotation(alpha, R);
                rtToPose(R, t, Talpha);

                Matx44d rawPose = TsgInv * (Talpha * Tmg);

                Pose3DPtr pose(new Pose3D(alpha, refIndMax, accVal));  //modelIndex
                pose->updatePose(rawPose);

                if (spectral) {
                    //spectral matching
                    // form corres
                    Mat voted = coordAccumulator[accInd];
                    Mat modelI = voted.col(2);
                    Mat modelJ = voted.col(1) - modelI * n;
                    Mat sceneJ = voted.col(0);
                    int corres_n = voted.rows + 1;

                    vector<vector <Mat>> corr_list(corres_n); // (1, 6, CV_32F)
                    vector<vector <int>> corr_ptind_list(corres_n); // (1, 6, CV_32F)

                    corr_list[0].resize(2);
                    corr_ptind_list[0].resize(2);
                    sampled.row(i).copyTo(corr_list[0][0]); // si
                    sampled_pc.row(refIndMax).copyTo(corr_list[0][1]); //mi
                    corr_ptind_list[0][0] = i;// si
                    corr_ptind_list[0][1] = refIndMax;//mi

                    int sj, mj;
#if defined _OPENMP
#pragma omp parallel for
#endif

                    for (int ri = 0; ri < voted.rows; ri++) {
                        corr_list[ri + 1].resize(2);
                        corr_ptind_list[ri + 1].resize(2);
                        sj = sceneJ.at<int>(ri);
                        sampled.row(sj).copyTo(corr_list[ri + 1][0]);
                        mj = modelJ.at<int>(ri);
                        sampled_pc.row(mj).copyTo(corr_list[ri + 1][1]);
                        corr_ptind_list[ri + 1][0] = sj;// sj
                        corr_ptind_list[ri + 1][1] = mj;//mj
                    }

                    spectralMatch(pose, corres_n, corr_list, corr_ptind_list);
                    
                }
                // compute coordinates of the voters
                // i scene  j scene  i model j model
                //if (0) {
                //    Mat voted = coordAccumulator[accInd];
                //    Mat modelI = voted.col(2);
                //    Mat modelJ = voted.col(1) - modelI * n;
                //    Mat sceneJ = voted.col(0);
                //    Mat votes = Mat(voted.rows * 2 + 2, 6, CV_32F);
                //    sampled.row(i).copyTo(votes.row(0)); // si
                //    sampled_pc.row(refIndMax).copyTo(votes.row(1)); //mi
                //    int sj, mj;
                //    for (int ri = 0; ri < voted.rows; ri++) {
                //        sj = sceneJ.at<int>(ri);
                //        sampled.row(sj).copyTo(votes.row(2 + ri));
                //        mj = modelJ.at<int>(ri);
                //        sampled_pc.row(mj).copyTo(votes.row(2 + voted.rows + ri));
                //    }
                //    pose->addVoter(votes);
                //}

                #if defined (_OPENMP)
                #pragma omp critical
                #endif
                {
                    poseList.push_back(pose);
                }
            }
        }
    }
    free(accumulator);

/*
    // Maximize the accumulator
    for (uint k = 0; k < n; k++)
    {
      for (int j = 0; j < numAngles; j++)
      {
        const uint accInd = k*numAngles + j;
        const uint accVal = accumulator[ accInd ];
        if (accVal > maxVotes)
        {
          maxVotes = accVal;
          refIndMax = k;
          alphaIndMax = j;
        }

#if !defined (_OPENMP)
        accumulator[accInd ] = 0;
#endif
      }
    }

    // invert Tsg : Luckily rotation is orthogonal: Inverse = Transpose.
    // We are not required to invert.
    Vec3d tInv, tmg;
    Matx33d Rmg;
    RInv = Rsg.t();
    tInv = -RInv * tsg;

    Matx44d TsgInv;
    rtToPose(RInv, tInv, TsgInv);

    // TODO : Compute pose
    const Vec3f pMax(sampled_pc.ptr<float>(refIndMax));
    const Vec3f nMax(sampled_pc.ptr<float>(refIndMax) + 3);

    computeTransformRT(pMax, nMax, Rmg, tmg);

    Matx44d Tmg;
    rtToPose(Rmg, tmg, Tmg);

    // convert alpha_index to alpha
    int alpha_index = alphaIndMax;
    double alpha = (alpha_index*(4*M_PI))/numAngles-2*M_PI;

    // Equation 2:
    Matx44d Talpha;
    Matx33d R;
    Vec3d t = Vec3d::all(0);
    getUnitXRotation(alpha, R);
    rtToPose(R, t, Talpha);

    Matx44d rawPose = TsgInv * (Talpha * Tmg);

    Pose3DPtr pose(new Pose3D(alpha, refIndMax, maxVotes));
    pose->updatePose(rawPose);

    if (spectral) {
        //spectral matching
        // form corres
        const uint accInd = refIndMax * numAngles + alphaIndMax;
        Mat voted = coordAccumulator[accInd];
        Mat modelI = voted.col(2);
        Mat modelJ = voted.col(1) - modelI * n;
        Mat sceneJ = voted.col(0);
        int corres_n = voted.rows + 1;

        vector<vector <Mat>> corr_list(corres_n); // (1, 6, CV_32F)
        vector<vector <int>> corr_ptind_list(corres_n); // (1, 6, CV_32F)

        corr_list[0].resize(2);
        corr_ptind_list[0].resize(2);
        sampled.row(i).copyTo(corr_list[0][0]); // si
        sampled_pc.row(refIndMax).copyTo(corr_list[0][1]); //mi
        corr_ptind_list[0][0] = i;// si
        corr_ptind_list[0][1] = refIndMax;//mi

        int sj, mj;
#if defined _OPENMP
#pragma omp parallel for
#endif

        for (int ri = 0; ri < voted.rows; ri++) {
            corr_list[ri + 1].resize(2);
            corr_ptind_list[ri + 1].resize(2);
            sj = sceneJ.at<int>(ri);
            sampled.row(sj).copyTo(corr_list[ri + 1][0]);
            mj = modelJ.at<int>(ri);
            sampled_pc.row(mj).copyTo(corr_list[ri + 1][1]);
            corr_ptind_list[ri + 1][0] = sj;// sj
            corr_ptind_list[ri + 1][1] = mj;//mj
        }

        spectralMatch(pose, corres_n, corr_list, corr_ptind_list);

    }


    #if defined (_OPENMP)
    #pragma omp critical
    #endif
    {
      poseList.push_back(pose);
    }

    free(accumulator);*/
  }

  // TODO : Make the parameters relative if not arguments.
  //double MinMatchScore = 0.5;

  int numPosesAdded = poseList.size();
  if (1) {
      std::sort(poseList.begin(), poseList.end(), pose3DPtrCompare); // 将clusterPoses中对poseList的排序放到这里
      //if (debug) debugPose(poseList, "numVotes", "afterHoughVot", true); // false
      results = poseList;

        //clusterPoses(poseList, numPosesAdded, results);
        //if (debug) debugPose(results, "numVotes", "afterCluster", true);
        //int postPoseNum = 10000;
        //if (results.size() < postPoseNum) postPoseNum = results.size();
        //vector<Pose3DPtr> resultsPost(results.begin(), results.begin() + postPoseNum);
        //results = resultsPost;
  }



  /*
  if(0)
  {
      if (debug) debugPose(poseList, "numVotes", "afterHoughVot", true);
      clusterPoses(poseList, std::min(2000, numPosesAdded), results);
      if (debug) debugPose(results, "numVotes", "afterCluster", true);
      //results = poseList;
      
      if (useFreeSpaceConstraint) {
          std::vector<Pose3DPtr> filterRes;
          //std::vector<Pose3DPtr> smallRes;
          //smallRes.assign(results.begin(), results.begin() + 150);
          freespaceIntersectionCount(results, filterRes, 5); // uwa 100
          if (filterRes.size() > 0) results = filterRes;
          std::sort(results.begin(), results.end(), pose3DPtrCompare);
          if (debug) debugPose(results, "numVotes", "afterFreespace", true);
      }

      overlapRatio(sampled_pc, downsample_scene, downsample_scene_flannIndex, results, model_diameter * 0.05, 25); //uwa 0.01, 25
      std::sort(results.begin(), results.end(), pose3DPtrCompareOverlap);
      if (debug) debugPose(results, "overlap", "afterOverlapRatio", true);

  }*/

  /*
  if (0) {
      std::sort(poseList.begin(), poseList.end(), pose3DPtrCompare);
      std::vector<Pose3DPtr> finalPoses; NMScenter(poseList, nmsThreshold, finalPoses);
      debugPose(finalPoses, "numVotes", "afterHoughVot", true);

      clusterPoses(poseList, std::min(2000, numPosesAdded), results);
      NMScenter(results, nmsThreshold, finalPoses);
      debugPose(finalPoses, "numVotes", "afterCluster", true);
      //results = poseList;

      if (useFreeSpaceConstraint) {
          std::vector<Pose3DPtr> filterRes;
          //std::vector<Pose3DPtr> smallRes;
          //smallRes.assign(results.begin(), results.begin() + 150);
          freespaceIntersectionCount(results, filterRes, 5); // uwa 100
          if (filterRes.size() > 0) results = filterRes;
          std::sort(results.begin(), results.end(), pose3DPtrCompare);
          NMScenter(results, nmsThreshold, finalPoses);
          debugPose(finalPoses, "numVotes", "afterFreespace", true);
      }


      overlapRatio(sampled_pc, downsample_scene, downsample_scene_flannIndex, results, model_diameter * 0.05, 25); //uwa 0.01, 25
      std::sort(results.begin(), results.end(), pose3DPtrCompareOverlap);
      NMScenter(results, nmsThreshold, finalPoses);
      debugPose(finalPoses, "overlap", "afterOverlapRatio", true);
  }*/



  //std::vector<Pose3DPtr> filterRes;
  //std::vector<Pose3DPtr> smallRes;
  //smallRes.assign(results.begin(), results.begin() + 100);
  //freespaceIntersectionCount(smallRes, filterRes, 100);
  //results = filterRes;
  //std::sort(results.begin(), results.end(), pose3DPtrCompareOverlap);
  //if (debug) debugPose(results, "overlap", "afterFreespace", true);

}

void PPF3DDetector::spectralMatch(Pose3DPtr& pose, int corres_n, std::vector<std::vector<cv::Mat>>& corr_list, std::vector<std::vector<int>>& corr_ptind_list)
{
    //build consistency matrix
    float sigmaDist = 0.04; // controls Gaussian kernel width
    float sigmaAngle = 0.2615;
    Eigen::MatrixXd A;
    A.resize(corres_n, corres_n);
    A.diagonal().setOnes();

#if defined _OPENMP
#pragma omp parallel for
#endif
    for (int r = 0; r < corres_n; r++) {
        vector<Mat>& corr_1 = corr_list[r];
        const Vec3f sp1(corr_1[0].ptr<float>(0));
        const Vec3f sn1(corr_1[0].ptr<float>(0) + 3);
        const Vec3f mp1(corr_1[1].ptr<float>(0));
        const Vec3f mn1(corr_1[1].ptr<float>(0) + 3);

        for (int c = r + 1; c < corres_n; c++) {
            vector<Mat>& corr_2 = corr_list[c];
            const Vec3f sp2(corr_2[0].ptr<float>(0));
            const Vec3f sn2(corr_2[0].ptr<float>(0) + 3);
            const Vec3f mp2(corr_2[1].ptr<float>(0));
            const Vec3f mn2(corr_2[1].ptr<float>(0) + 3);

            // for every two corres, comput five consistency measures
            Vec4d sf1_4 = Vec4d::all(0); // agl, a, a, dist
            computePPFFeatures(sp1, sn1, sp2, sn2, sf1_4);
            double sf5theta = computeTheta(sp1, sn1, sp2, sn2);

            Vec4d mf1_4 = Vec4d::all(0);
            computePPFFeatures(mp1, mn1, mp2, mn2, mf1_4);
            double mf5theta = computeTheta(mp1, mn1, mp2, mn2);

            double da1 = sf1_4[0] - mf1_4[0];
            double da2 = sf1_4[1] - mf1_4[1];
            double da3 = sf1_4[2] - mf1_4[2];
            double dd = sf1_4[3] - mf1_4[3];
            double da4 = sf5theta - mf5theta;
            double ta1 = -0.5 * (pow(da1, 2) / pow(sigmaAngle, 2));
            double ta2 = -0.5 * (pow(da2, 2) / pow(sigmaAngle, 2));
            double ta3 = -0.5 * (pow(da3, 2) / pow(sigmaAngle, 2));
            double td = -0.5 * (pow(dd, 2) / pow(sigmaDist, 2));
            double ta4 = -0.5 * (pow(da1, 2) / pow(sigmaAngle, 2));

            double wc1c2 = exp(ta1 + ta2 + ta3 + td + ta4);

            A(r, c) = wc1c2;
            A(c, r) = wc1c2;
        }
    }


    // eigen decomposition
    Eigen::EigenSolver<Eigen::MatrixXd> es(A);
    Eigen::MatrixXcd evecs = es.eigenvectors();//获取矩阵特征向量4*4，这里定义的MatrixXcd必须有c，表示获得的是complex复数矩阵
    Eigen::MatrixXcd evals = es.eigenvalues();//获取矩阵特征值 4*1
    Eigen::MatrixXd evalsReal;//注意这里定义的MatrixXd里没有c
    evalsReal = evals.real();//获取特征值实数部分
    Eigen::MatrixXd::Index evalsMax;
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);//得到最大特征值的位置
    double maxev = evalsReal.rowwise().sum().maxCoeff();
    Eigen::Matrix<double, -1, 1> vm;
    vm = evecs.real().col(evalsMax);

    //cout << "A: " << endl << A << endl;
    //cout << "evalsReal: " << endl << evalsReal << endl;
    //cout << "evecs.real(): " << endl << evecs.real() << endl;

    // eigen vector nms
    VectorXi sorted_vm_corrid;
    VectorXd sorted_vm; // 6,5,4,3,2,1
    sort_vec(vm, sorted_vm, sorted_vm_corrid); // ind = 3,0,4,1...
    vector<int> corrid_remained;
    
    //cout << "原始向量: " << endl;
    //cout << vm << endl << endl;
    //cout << "排序后: " << endl;
    //cout << sorted_vm << endl << endl;
    //cout << "排序后向量各元素对应的原始向量中的位置" << endl;
    //cout << sorted_vm_corrid << endl;
    
    int i, j;
    VectorXi is_suppressed(corres_n);
    is_suppressed.setZero();
    //vector<bool> is_suppressed(corres_n);

//#if defined _OPENMP
//#pragma omp parallel for
//#endif
//
//    for (i = 0; i < corres_n; i++) {
//        is_suppressed[i] = 0;
//    }


    for (i = 0; i < corres_n; i++)                // 循环所有窗口   
    {
        if (!is_suppressed[i])           // 判断窗口是否被抑制   
        {
            if (sorted_vm[i] < 1e-5) { 
                is_suppressed[i] = 1; 
                continue; 
            }
            // keeped corres mapping 
            int corr_id_kpd = sorted_vm_corrid[i];
            std::vector<int >& corr_ptind_keepd = corr_ptind_list[corr_id_kpd];
            for (j = i + 1; j < corres_n; j++)
            {
                if (!is_suppressed[j])   // 判断窗口是否被抑制   
                {
                    if (sorted_vm[j] < 1e-5) {
                        is_suppressed[j] = 1;
                        continue;
                    }
                    // current corres mapping
                    int corr_id_crt = sorted_vm_corrid[j];
                    std::vector<int >& corr_ptind_current = corr_ptind_list[corr_id_crt]; // size 2
                    // 判断是否被抑制
                    for (int ikp = 0; ikp < 2; ikp++) {
                        for (int ic = 0; ic < 2; ic++) {
                            bool violate = (corr_ptind_keepd[ikp] == corr_ptind_current[ic]);
                            if (violate) {
                                is_suppressed[j] = 1;           // 将窗口j标记为抑制  

                            }
                        }
                    }

                }
            }
        }
    }

    for (i = 0; i < corres_n; i++)                  // 遍历所有输入窗口   
    {
        if (!is_suppressed[i])             // 将未发生抑制的窗口信息保存到输出信息中   
        {
            corrid_remained.push_back( sorted_vm_corrid[i]);
        }
    }

    pose->updateVotes(corrid_remained.size());
    pose->updateEigenv(maxev);

    // update pose
    //Matx44d newPose = Matx44d::eye();
    //int selInd = corrid_remained.size();
    //bool valid = false;
    //if (selInd >= 6)
    //{

    //    Mat Src_Match = Mat(selInd, 6, CV_64F);
    //    Mat Dst_Match = Mat(selInd, 6, CV_64F);

    //    for (int di = 0; di < selInd; di++)
    //    {
    //        int corrid = corrid_remained[di];
    //        vector<Mat>& corr_ = corr_list[corrid];

    //        const float* srcPt = corr_[0].ptr<float>(0);
    //        const float* dstPt = corr_[1].ptr<float>(0);
    //        double* srcMatchPt = Src_Match.ptr<double>(di);
    //        double* dstMatchPt = Dst_Match.ptr<double>(di);
    //        int ci = 0;

    //        for (ci = 0; ci < 6; ci++)
    //        {
    //            srcMatchPt[ci] = (double)srcPt[ci];
    //            dstMatchPt[ci] = (double)dstPt[ci];
    //        }
    //    }

    //    Vec3d rpy, t;
    //    minimizePointToPlaneMetric(Src_Match, Dst_Match, rpy, t);
    //    if (cvIsNaN(cv::trace(rpy)) || cvIsNaN(cv::norm(t)))
    //        valid = false;
    //    else {
    //        valid = true;
    //        getTransformMat(rpy, t, newPose);
    //    }
    //}
    //else { valid = false; }

    //if (valid)
    //    pose->updatePose(newPose);


}

void PPF3DDetector::debugMatch(const Mat& pc, std::vector<Pose3DPtr>& results, const double relativeSceneSampleStep, const double relativeSceneDistance)
{
    if (!trained)
    {
        throw cv::Exception(cv::Error::StsError, "The model is not trained. Cannot match without training", __FUNCTION__, __FILE__, __LINE__);
    }

    CV_Assert(pc.type() == CV_32F || pc.type() == CV_32FC1);
    CV_Assert(relativeSceneSampleStep <= 1 && relativeSceneSampleStep > 0);

    scene_sample_step = (int)(1.0 / relativeSceneSampleStep);

    //int numNeighbors = 10;
    int numAngles = (int)(floor(2 * M_PI / angle_step));
    float distanceStep = (float)distance_step;
    uint n = num_ref_points;
    std::vector<Pose3DPtr> poseList;
    int sceneSamplingStep = scene_sample_step;

    // compute bbox
    Vec2f xRange, yRange, zRange;
    computeBboxStd(pc, xRange, yRange, zRange);

    // sample the point cloud
    /*float dx = xRange[1] - xRange[0];
    float dy = yRange[1] - yRange[0];
    float dz = zRange[1] - zRange[0];
    float diameter = sqrt ( dx * dx + dy * dy + dz * dz );
    float distanceSampleStep = diameter * RelativeSceneDistance;*/

    //Mat sampled = samplePCByQuantization(pc, xRange, yRange, zRange, (float)relativeSceneDistance, 0);

    float distanceSampleStep = relativeSceneDistance * model_diameter;
    float distanceSampleStepRefine = distanceSampleStep / 2;

    Mat sampled, sampledDenseRefinement;
    if (samplingMethod == "cube") {
        sampled = samplePCByQuantization_cube(pc, xRange, yRange, zRange, distanceSampleStep, 0);
        sampledDenseRefinement = samplePCByQuantization_cube(pc, xRange, yRange, zRange, distanceSampleStepRefine, 0);
    }
    else if (samplingMethod == "normal") {
        sampled = samplePCByQuantization_normal(pc, xRange, yRange, zRange, distanceSampleStep, 15.0 / 180 * M_PI, 3);
        sampledDenseRefinement = samplePCByQuantization_normal(pc, xRange, yRange, zRange, distanceSampleStepRefine, 15.0 / 180 * M_PI, 3);

    }


    //Mat sampledDenseRefinement= samplePCUniform(pc, 8);
    relative_scene_distance = relativeSceneDistance;

    //downsample_scene_dense_refinement = sampledDenseRefinement;
    //downsample_scene = sampled;
    //downsample_scene_flannIndex = indexPCFlann(downsample_scene);
    //downsample_scene_dense_refinement_flannIndex = indexPCFlann(downsample_scene_dense_refinement);

    // allocate the accumulator : Moved this to the inside of the loop
    /*#if !defined (_OPENMP)
       uint* accumulator = (uint*)calloc(numAngles*n, sizeof(uint));
    #endif*/

    poseList.reserve((sampled.rows / sceneSamplingStep) + 4);

    #if defined _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < debug_sampled_scene_ref.rows; i++)
    {
        // 给定一个sr
        //uint refIndMax = 0, alphaIndMax = 0;
        //uint maxVotes = 0;

        const Vec3f p1(debug_sampled_scene_ref.ptr<float>(i));
        const Vec3f n1(debug_sampled_scene_ref.ptr<float>(i) + 3);
        Vec3d tsg = Vec3d::all(0);
        Matx33d Rsg = Matx33d::all(0), RInv = Matx33d::all(0);

        uint* accumulator = (uint*)calloc(numAngles * n, sizeof(uint));
        computeTransformRT(p1, n1, Rsg, tsg);


        // Tolga Birdal's notice:
        // As a later update, we might want to look into a local neighborhood only
        // To do this, simply search the local neighborhood by radius look up
        // and collect the neighbors to compute the relative pose

        for (int j = 0; j < sampled.rows; j++)
        {
            if (i != j)
            {
                const Vec3f p2(sampled.ptr<float>(j));
                const Vec3f n2(sampled.ptr<float>(j) + 3);

                Vec3d p2t;
                double alpha_scene;

                Vec4d f = Vec4d::all(0);
                computePPFFeatures(p1, n1, p2, n2, f);
                KeyType hashValue = hashPPF(f, angle_step, distanceStep);
                p2t = tsg + Rsg * Vec3d(p2);

                alpha_scene = atan2(-p2t[2], p2t[1]);

                if (alpha_scene != alpha_scene)
                {
                    continue;
                }

                if (sin(alpha_scene) * p2t[2] < 0.0)
                    alpha_scene = -alpha_scene;

                alpha_scene = -alpha_scene;

                hashnode_i* node = hashtableGetBucketHashed(hash_table, (hashValue));

                while (node)
                {
                    THash* tData = (THash*)node->data;
                    int corrI = (int)tData->i;
                    int ppfInd = (int)tData->ppfInd;
                    float* ppfCorrScene = ppf.ptr<float>(ppfInd);
                    double alpha_model = (double)ppfCorrScene[PPF_LENGTH - 1];
                    double alpha = alpha_model - alpha_scene;

                    /*  Tolga Birdal's note: Map alpha to the indices:
                            atan2 generates results in (-pi pi]
                            That's why alpha should be in range [-2pi 2pi]
                            So the quantization would be :
                            numAngles * (alpha+2pi)/(4pi)
                            */

                            //printf("%f\n", alpha);
                    int alpha_index = (int)(numAngles * (alpha + 2 * M_PI) / (4 * M_PI));

                    uint accIndex = corrI * numAngles + alpha_index;

                    accumulator[accIndex]++;
                    node = node->next;
                }
            }
        }

        // 对于一个给定的sr，查找accumulator中是否有tp pose，得分是多少
        // 遍历accumulator中每个格子，计算位姿
        std::vector<Pose3DPtr> srPoseList;
        srPoseList.reserve(numAngles * n);

        for (uint k = 0; k < n; k++)
        {
            for (int j = 0; j < numAngles; j++)
            {
                const uint accInd = k * numAngles + j;
                const uint accVal = accumulator[accInd];
                if (accVal == 0) continue;
                uint Votes = accVal;
                uint refInd = k;
                uint alphaInd = j;

                // invert Tsg : Luckily rotation is orthogonal: Inverse = Transpose.
                // We are not required to invert.
                Vec3d tInv, tmg;
                Matx33d Rmg;
                RInv = Rsg.t();
                tInv = -RInv * tsg;

                Matx44d TsgInv;
                rtToPose(RInv, tInv, TsgInv);

                // TODO : Compute pose
                const Vec3f pMax(sampled_pc.ptr<float>(refInd));
                const Vec3f nMax(sampled_pc.ptr<float>(refInd) + 3);

                computeTransformRT(pMax, nMax, Rmg, tmg);

                Matx44d Tmg;
                rtToPose(Rmg, tmg, Tmg);

                // convert alpha_index to alpha
                int alpha_index = alphaInd;
                double alpha = (alpha_index * (4 * M_PI)) / numAngles - 2 * M_PI;

                // Equation 2:
                Matx44d Talpha;
                Matx33d R;
                Vec3d t = Vec3d::all(0);
                getUnitXRotation(alpha, R);
                rtToPose(R, t, Talpha);

                Matx44d rawPose = TsgInv * (Talpha * Tmg);

                Pose3DPtr pose(new Pose3D(alpha, refInd, Votes));
                pose->updatePose(rawPose);

//#if defined (_OPENMP)
//#pragma omp critical
//#endif
                {
                    srPoseList.push_back(pose);
                }


            }
        }

        free(accumulator);

        std::sort(srPoseList.begin(), srPoseList.end(), pose3DPtrCompare);

        if (debug) {
            string stage = "after_Sr_HoughVot_" + to_string(i);
            debugPose(srPoseList, "numVotes", stage, true, "HoughVot/");
        }

        cout << "sr " << i << endl;
    }

    // TODO : Make the parameters relative if not arguments.
    //double MinMatchScore = 0.5;

    //int numPosesAdded = poseList.size();

    //std::sort(poseList.begin(), poseList.end(), pose3DPtrCompare); // 将clusterPoses中对poseList的排序放到这里

    //debugPose(poseList, "numVotes", "afterHoughVot");

    //clusterPoses(poseList, numPosesAdded, results);
    ////clusterPosesNMS(poseList, 0.5, results);

    //debugPose(results, "numVotes", "afterCluster");

    //overlapRatio(sampled_pc, downsample_scene, downsample_scene_flannIndex, results, model_diameter * 0.08, 25); //relativeSceneDistance *0.8
    //std::sort(results.begin(), results.end(), pose3DPtrCompareOverlap);

    //debugPose(results, "overlap", "afterOverlapRatio");

}

void PPF3DDetector::postProcessing(std::vector<Pose3DPtr>& results, ICP& icp, bool refineEnabled, bool nmsEnabled)
{
    // nms for multi instances
    // 注意，这里截取了前100个位姿做NMS
    if (nmsEnabled) {
    //if (0) {
        std::vector<Pose3DPtr> finalPoses;
        

        NMSbbox(results, nmsThreshold, finalPoses);
        //NMScenter(results, nmsThreshold, finalPoses);
        //std::sort(finalPoses.begin(), finalPoses.end(), pose3DPtrCompare);
        //std::sort(finalPoses.begin(), finalPoses.end(), pose3DPtrCompareClusterCount);
        //debugPose(finalPoses, "numVotes", "afterNMSall", true);
        cout << "NMS results num: " << finalPoses.size() << endl;
        results = finalPoses;

        // overlap
        //freespaceRatio(finalPoses, model_diameter * 0.005);
        //overlapRatio(sampled_pc_refinement, downsample_scene_dense_refinement, downsample_scene_dense_refinement_flannIndex, finalPoses, model_diameter * 0.02, 25);
        //std::sort(finalPoses.begin(), finalPoses.end(), pose3DPtrCompareOverlap);
        //debugPose(finalPoses, "overlap", "afterOverlapRatio", true);
        //results = finalPoses;

        //int postPoseNum = 300;
        //if (finalPoses.size() < postPoseNum) postPoseNum = finalPoses.size();
        //vector<Pose3DPtr> resultsPost(finalPoses.begin(), finalPoses.begin() + postPoseNum);
        //results = resultsPost;
        //cout << "NMS results num: " << results.size() << endl;

        if (debug) {
            //debugPose(results, "overlap", "afterNMS", true);
            //debugPose(results, "numVotes", "afterNMStop200", true);
        }
    }

    if (refineEnabled)
    {
        // sparse refine
        //SparseICP(results, sampled_pc_refinement, downsample_scene, 5);

        //icp.registerModelToScene(sampled_pc_refinement, downsample_scene_sparse_refinement, results);


        //overlapRatio(sampled_pc_refinement, downsample_scene_sparse_refinement, downsample_scene_sparse_refinement_flannIndex, results, model_diameter * 0.02, 25);//relative_scene_distance * 0.8
        //std::sort(results.begin(), results.end(), pose3DPtrCompareOverlap);

        // dense refine
        //SparseICP(results, sampled_pc_refinement, downsample_scene_dense_refinement, 5);

        icp.registerModelToScene(sampled_pc_refinement, downsample_scene_dense_refinement, results);
        double angle_th = 30;
        double dist_th = model_diameter * 0.02;
        overlapRatio(sampled_pc_refinement, downsample_scene_dense_refinement, downsample_scene_dense_refinement_flannIndex, results, dist_th, angle_th, 1); //model_diameter * 0.005, 25
        //double dist_th = model_diameter * 0.01;
        //sdfRatio(results, dist_th);
        //freespaceRatio(results, dist_th); // model_diameter * 0.01
        std::sort(results.begin(), results.end(), pose3DPtrCompareOverlap);
        //if (debug) debugPose(results, "overlap", "afterOverlapRatio", true);


        std::vector<Pose3DPtr> finalPoses;
        //NMSbbox(results, 0.5, finalPoses);
        //cout << "NMS results num: " << finalPoses.size() << endl;
        //results = finalPoses;
        NMScenter(results, 0.3, finalPoses); // 0.3
        cout << "NMS results num: " << finalPoses.size() << endl;
        results = finalPoses;
        //if (debug) debugPose(results, "overlap", "afterNMScenter", true);

        //icp.registerModelToScene(sampled_pc_refinement, downsample_scene_dense_refinement, results);
        //overlapRatio(sampled_pc_refinement, downsample_scene_dense_refinement, downsample_scene_dense_refinement_flannIndex, results, model_diameter * 0.03, 30, 1); //model_diameter * 0.03, 25
        ////freespaceRatio(results, model_diameter * 0.01); // model_diameter * 0.01
        //std::sort(results.begin(), results.end(), pose3DPtrCompareOverlap);
        ////if (debug) debugPose(results, "overlap", "afterRefinement", true);
        //NMScenter(results, 0.2, finalPoses);
        //cout << "NMS results num: " << finalPoses.size() << endl;
        //results = finalPoses;



    }

}

void PPF3DDetector::deleteFlannPtr() {
    destroyFlann(downsample_scene_sparse_refinement_flannIndex);
    destroyFlann(downsample_scene_dense_refinement_flannIndex);
}

/*
看看到底哪个位姿是准确的，假设已经对Poses做了排序
scoreType: numVotes or overlap
*/ 
void PPF3DDetector::debugPose(std::vector<Pose3DPtr>& Poses, std::string scoreType, std::string stage, bool save, std::string saveFolder){
    double th = 0.1 * model_diameter;
    int poseNum = Poses.size();
    vector<bool> whetherTP(poseNum, false);
    vector<double> scores(poseNum, 0);

    Mat center = Mat(1, model_center.rows, CV_32F);
    float* pcData = center.ptr<float>(0);
    pcData[0] = (float)model_center[0];
    pcData[1] = (float)model_center[1];
    pcData[2] = (float)model_center[2];

    // 用 TP 对应的位姿，对模型做变换，存下来，看看为啥得票数那么低

#if defined _OPENMP
#pragma omp parallel for
#endif

    for (int i = 0; i < poseNum; i++) {
        Pose3DPtr Pose = Poses[i];
        Mat pct_pred = transformPCPose(sampled_pc, Pose->pose);
        //score
        if (scoreType == "numVotes") {
            scores[i] = Pose->numVotes;
        }
        else if (scoreType == "overlap") {
            scores[i] = Pose->overlap;
        }
        //else if (scoreType == "freespace") {
        //    scores[i] = Pose->freespaceIntersec;
        //}
        else {
            cout << "wrong scoreType: " << scoreType << endl; exit(1);
        }

        // TP = 0, 1
        if (use_multi_gt_pose) {

            Mat features;
            pct_pred.colRange(0, 3).copyTo(features);
            kdtree::KDTree* model_tree = kdtree::BuildKDTree(features);
            Mat cp = transformPCPose(center, Pose->pose);

            // compare with all gt pose
            //vector<int> cases{ 0 ,4, 5 };
            //for (int igt = 0; igt < multi_gt_pose_model.size(); igt++) {
            vector<int> cases{ 4 };
            for (int igt : cases) {
                
                // prefilter with center distance
                Vec3f cd = multi_gt_pose_center[igt].at<Vec3f>(0, 0) - cp.at<Vec3f>(0, 0); //https://www.cnblogs.com/hsy1941/p/8298314.html
                if (cv::norm(cd) < 0.5 * model_diameter) {
                    // ADI

                    Mat queries;
                    multi_gt_pose_model[igt].colRange(0, 3).copyTo(queries);
                    //cout << pc.at<float>(0, 0) << pc.at<float>(0, 1) << pc.at<float>(0, 2) << endl;
                    std::vector<std::vector<int>> indices;
                    std::vector<std::vector<float>> dists;
                    kdtree::SearchKDTree(model_tree, queries, indices, dists, 1);
                    double totalD_ADI = 0;
                    for (int ii = 0; ii < queries.rows; ii++)
                    {
                        totalD_ADI += sqrt(dists[ii][0]);
                    }
                    totalD_ADI /= queries.rows;
                    if (totalD_ADI < th) {
                        whetherTP[i] = 1;
                        break;
                    }
                }
            }
            delete model_tree;
        }
        else {
            if (isTPUsingADD(gtPoseModel, pct_pred, th))
                whetherTP[i] = 1;
        }

        // 对于是TP的pose，以及排名前三的pose，将model根据他们做转换，保存下来
        //if (whetherTP[i] == 1 || i < 3) {
        if (stage == "afterNMScenter" && (whetherTP[i] == 1 || i < 3) || (whetherTP[i] == 1 || i < 3)) {
            if (save) {
                // saveFolder = "HoughVot/"
                string prefix = "../samples/data/results/" + debug_folder_name + "/debug_" + saveFolder + stage + "_" + to_string(i + 1);
                string debugName = prefix + "_" + scoreType + "_" + to_string(scores[i]);
                writePLY(pct_pred, (debugName + ".ply").c_str());

                //save votes' point cloud
               //if (stage == "afterHoughVot") {
                if (0) {
                    Mat votes = Pose->voters;
                    string votesNameSi = debugName + "_si" + ".ply";
                    string votesNameMi = debugName + "_mi" + ".ply";
                    string votesNameSj = debugName + "_sj" + ".ply";
                    string votesNameMj = debugName + "_mj" + ".ply";
                    Mat pct_pred_mi = transformPCPose(votes.row(1), Pose->pose);
                    Mat pct_pred_mj = transformPCPose(votes.rowRange(2 + (votes.rows - 2) / 2, votes.rows), Pose->pose);

                    writePLY(votes.row(0), votesNameSi.c_str());
                    writePLY(pct_pred_mi, votesNameMi.c_str());
                    writePLY(votes.rowRange(2, 2 + (votes.rows - 2) / 2), votesNameSj.c_str());
                    writePLY(pct_pred_mj, votesNameMj.c_str());
                }
            }
        }
    }


    // write vector to txt
    string whetherTPPath = "../samples/data/results/"+ debug_folder_name + "/debug_whetherTP_" +  stage+".txt";
    ofstream ofTP(whetherTPPath);
    for (const auto& i : whetherTP) {
        ofTP << i << ' ' << endl;
    }
    ofTP.close();
    string scoresPath = "../samples/data/results/" + debug_folder_name + "/debug_scores_" +  stage + ".txt";
    ofstream ofScr(scoresPath);
    for (const auto& i : scores) {
        ofScr << i << ' ' << endl;
    }
    ofScr.close();
}


} // namespace ppf_match_3d

} // namespace cv
