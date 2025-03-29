
// Author: Tolga Birdal <tbirdal AT gmail.com>

#include "precomp.hpp"
#include <boost/algorithm/string.hpp>

namespace kdtree {

    KDTree* BuildKDTree(const  cv::Mat& data)
    {
        int rows, dim;
        rows = (int)data.rows;
        dim = (int)data.cols;
        //std::cout << rows * dim  << " " << std::endl;
        float* temp = new float[rows * dim];

        flann::Matrix<float> dataset_mat(temp, rows, dim);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                dataset_mat[i][j] = data.at<float>(i, j);
                //std::cout << data(i, j) << "  ";
            }
            //std::cout << std::endl;
        }

        //KDTreeSingleIndexParams 为搜索最大叶子数
        KDTree* tree = new KDTree(dataset_mat, flann::KDTreeSingleIndexParams(15));
        tree->buildIndex();
        //std::cout << "test..." << tree->size() << std::endl;
        //tree = &temp_tree;
        delete[] temp;

        return tree;
    }

    void SearchKDTree(KDTree* tree, const cv::Mat& input,
        std::vector<std::vector<int>>& indices,
        std::vector<std::vector<float>>& dists, int nn)
    {
        int rows_t = input.rows;
        int dim = input.cols;

        float* temp = new float[rows_t * dim];
        flann::Matrix<float> query_mat(temp, rows_t, dim);
        for (int i = 0; i < rows_t; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                query_mat[i][j] = input.at<float>(i, j);
            }
        }

        indices.resize(rows_t);
        dists.resize(rows_t);

        for (int i = 0; i < rows_t; i++)
        {
            indices[i].resize(nn);
            dists[i].resize(nn);
        }

        tree->knnSearch(query_mat, indices, dists, nn, flann::SearchParams(128));
        delete[] temp;
    }
}


namespace cv
{
namespace ppf_match_3d
{

int writeMap(const map<string, vector<double>>& m, const string& outPath) {
    //map<string, vector<double>> m = { {"1th", vector<double>{0,1,2}} , {"2th", vector<double>{0,1,2}} };
    // 存入文件out.txt
    ofstream of(outPath);
    for (const auto& i : m) {
        of << i.first << ' ';
        for (auto& v : i.second)
            of << v << ' ';
        of << std::endl;
    }
    of.close();
    return 0;
}

int writeMap(const map<string, vector<string>>& m, const string& outPath) {
    //map<string, vector<double>> m = { {"1th", vector<double>{0,1,2}} , {"2th", vector<double>{0,1,2}} };
    // 存入文件out.txt
    ofstream of(outPath);
    for (const auto& i : m) {
        of << i.first << ' ';
        for (auto& v : i.second)
            of << v << ' ';
        of << std::endl;
    }
    of.close();
    return 0;
}

int writeMap(const map<string, double>& m, const string& outPath) {
    //map<string, vector<double>> m = { {"1th", vector<double>{0,1,2}} , {"2th", vector<double>{0,1,2}} };
    // 存入文件out.txt
    ofstream of(outPath);
    for (const auto& i : m) {
        of << i.first << ' ' << i.second << std::endl;
    }
    of.close();
    return 0;
}

int readMap(map<string, vector<double>>& m2, const string& inPath) {
    // 读取文件，存入map m2中
    //map<string, vector<double>> m2;
    ifstream iff(inPath);
    if (!iff.is_open()) { cout << "not open: " << inPath << endl; exit(1); }
    string keyval;
    while (getline(iff, keyval)) {
        std::vector<std::string> mStr;
        boost::split(mStr, keyval, boost::is_any_of(" "));
        for (int i = 1; i < mStr.size() - 1; i++)
            m2[mStr[0]].push_back(stod(mStr[i]));
    }
    iff.close();
    return 0;
}

int readMap(map<string, vector<string>>& m2, const string& inPath) {
    // 读取文件，存入map m2中
    //map<string, vector<double>> m2;
    ifstream iff(inPath);
    if (!iff.is_open()) { cout << "not open: " << inPath << endl; exit(1); }
    string keyval;
    while (getline(iff, keyval)) {
        std::vector<std::string> mStr;
        boost::split(mStr, keyval, boost::is_any_of(" "));
        for (int i = 1; i < mStr.size() - 1; i++)
            m2[mStr[0]].push_back(mStr[i]);
    }
    iff.close();
    return 0;
}

int readMap(map<string, double>& m2, const string& inPath) {
    // 读取文件，存入map m2中
    //map<string, vector<double>> m2;
    ifstream iff(inPath);
    if (!iff.is_open()) { cout << "not open: " << inPath << endl; exit(1); }
    string keyval;
    while (getline(iff, keyval)) {
        std::vector<std::string> mStr;
        boost::split(mStr, keyval, boost::is_any_of(" "));
        m2[mStr[0]] = stod(mStr[1]);
    }
    iff.close();
    return 0;
}


typedef cv::flann::L2<float> Distance_32F;
typedef cv::flann::GenericIndex< Distance_32F > FlannIndex;

void shuffle(int *array, size_t n);
Mat genRandomMat(int rows, int cols, double mean, double stddev, int type);
void getRandQuat(Vec4d& q);
void getRandomRotation(Matx33d& R);
void meanCovLocalPC(const Mat& pc, const int point_count, Matx33d& CovMat, Vec3d& Mean);
void meanCovLocalPCInd(const Mat& pc, const int* Indices, const int point_count, Matx33d& CovMat, Vec3d& Mean);

static std::vector<std::string> split(const std::string &text, char sep) {
  std::vector<std::string> tokens;
  std::size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != std::string::npos) {
    tokens.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  tokens.push_back(text.substr(start));
  return tokens;
}

void cvMat2vcgMesh(const Mat& pc, MyMesh& m)
{
    m.Clear();

    int vertCount = pc.rows;
    vcg::tri::Allocator<MyMesh>::AddVertices(m, vertCount);
    for (int i = 0; i < vertCount; ++i)
    {
        const float* data = pc.ptr<float>(i);
        m.vert[i].P() = vcg::Point3f(data[0], data[1], data[2]);
        m.vert[i].N() = vcg::Point3f(data[3], data[4], data[5]);
    }
}

void vcgMesh2cvMat(const MyMesh& m, Mat& pc)
{
    pc = Mat(m.vert.size(), 6, CV_32FC1);
    for (int i = 0; i < m.vert.size(); ++i)
    {
        float* data = pc.ptr<float>(i);
        vcg::Point3f n = m.vert[i].N();
        vcg::Point3f p = m.vert[i].P();
        //
        //cout << p[0] << " " << p[1] << " " << p[2] << " " << n[0] << " " << n[1] << " " << n[2] << " " <<  endl;

        data[0] = p[0];
        data[1] = p[1];
        data[2] = p[2];
        //data[3] = n[0];
        //data[4] = n[1];
        //data[5] = n[2];

        // normalize the normals
        double nx = n[0], ny = n[1], nz = n[2];
        double norm = sqrt(nx * nx + ny * ny + nz * nz);

        if (norm > EPS)
        {
            data[3] = (float)(nx / norm);
            data[4] = (float)(ny / norm);
            data[5] = (float)(nz / norm);
        }
        else
        {
            data[3] = 0.0f;
            data[4] = 0.0f;
            data[5] = 0.0f;
        }
    }
}

Mat loadPLYSimple_bin(const char* fileName, int withNormals)
{
    //读入时前面6列为坐标和法向量
    Mat cloud;

    int numVertices = 0;
    int numCols = 3;
    int has_normals = 0;


    std::ifstream ifs(fileName, std::ios::in | std::ios::binary);
    std::string str1;
    str1 = fileName;
    if (!ifs.is_open())
        std::cout << "Error opening input file: " + str1 + "\n";

    /* char   buffer[80];
    getcwd(buffer, 80);
    printf("The   current   directory   is:   %s ", buffer);*/

    std::string str;
    while (str.substr(0, 10) != "end_header")
    {
        std::vector<std::string> tokens = split(str, ' ');
        if (tokens.size() == 3)
        {
            if (tokens[0] == "element" && tokens[1] == "vertex")
            {
                numVertices = atoi(tokens[2].c_str());
            }
            else if (tokens[0] == "property")
            {
                if (tokens[2] == "nx" || tokens[2] == "normal_x")
                {
                    has_normals = -1;
                    numCols += 3;
                }
                else if (tokens[2] == "r" || tokens[2] == "red")
                {
                    //has_color = true;
                    numCols += 3;
                }
                else if (tokens[2] == "a" || tokens[2] == "alpha")
                {
                    //has_alpha = true;
                    numCols += 1;
                }
            }
        }
        else if (tokens.size() > 1 && tokens[0] == "format" && tokens[1] != "ascii")
            std::cout << "Cannot read file, only ascii ply format is currently supported...\n";
        std::getline(ifs, str);
    }
    withNormals &= has_normals;

    cloud = Mat(numVertices, withNormals ? 6 : 3, CV_32FC1);
    //Eigen::Matrix<float,numVertices, withNormals ? 6 : 3,RowMajor> eigMatRow;

    std::cout << "模型点数：" << numVertices << "  法向量状态：" << withNormals << std::endl;

    int n = 0;
    for (int i = 0; i < numVertices; )
    {
        float fea[6];
        if (ifs.read((char*)&fea[0], 6 * sizeof(float)))
        {

            for (int col = 0; col < 6; ++col)
            {
                cloud.at<float>(i, col) = fea[col];
            }

            if (withNormals)//模型归一化
            {
                // normalize to unit norm
                double norm = sqrt(cloud.at<float>(i, 3) * cloud.at<float>(i, 3) + cloud.at<float>(i, 4) * cloud.at<float>(i, 4) + cloud.at<float>(i, 5) * cloud.at<float>(i, 5));
                if (norm > 0.00001)
                {
                    cloud.at<float>(i, 3) /= static_cast<float>(norm);
                    cloud.at<float>(i, 4) /= static_cast<float>(norm);
                    cloud.at<float>(i, 5) /= static_cast<float>(norm);
                }
            }
            i++;
        }


    }

    //cloud *= 5.0f;
    return cloud;
}

Mat loadPLYSimple(const char* fileName, int withNormals)
{
  Mat cloud;
  int numVertices = 0;
  int numCols = 3;
  int has_normals = 0;

  std::ifstream ifs(fileName);

  if (!ifs.is_open())
    CV_Error(Error::StsError, String("Error opening input file: ") + String(fileName) + "\n");

  std::string str;
  while (str.substr(0, 10) != "end_header")
  {
    std::vector<std::string> tokens = split(str,' ');
    if (tokens.size() == 3)
    {
      if (tokens[0] == "element" && tokens[1] == "vertex")
      {
        numVertices = atoi(tokens[2].c_str());
      }
      else if (tokens[0] == "property")
      {
        if (tokens[2] == "nx" || tokens[2] == "normal_x")
        {
          has_normals = -1;
          numCols += 3;
        }
        else if (tokens[2] == "r" || tokens[2] == "red")
        {
          //has_color = true;
          numCols += 3;
        }
        else if (tokens[2] == "a" || tokens[2] == "alpha")
        {
          //has_alpha = true;
          numCols += 1;
        }
      }
    }
    else if (tokens.size() > 1 && tokens[0] == "format" && tokens[1] != "ascii")
      CV_Error(Error::StsBadArg, String("Cannot read file, only ascii ply format is currently supported..."));
    std::getline(ifs, str);
  }
  withNormals &= has_normals;

  cloud = Mat(numVertices, withNormals ? 6 : 3, CV_32FC1);

  for (int i = 0; i < numVertices; i++)
  {
    float* data = cloud.ptr<float>(i);
    int col = 0;
    for (; col < (withNormals ? 6 : 3); ++col)
    {
      ifs >> data[col];
    }
    for (; col < numCols; ++col)
    {
      float tmp;
      ifs >> tmp;
    }
    if (withNormals)
    {
      // normalize to unit norm
      double norm = sqrt(data[3]*data[3] + data[4]*data[4] + data[5]*data[5]);
      if (norm>0.00001)
      {
        data[3]/=static_cast<float>(norm);
        data[4]/=static_cast<float>(norm);
        data[5]/=static_cast<float>(norm);
      }
    }
  }

  //cloud *= 5.0f;
  return cloud;
}

void writePLY(Mat PC, const char* FileName)
{
  std::ofstream outFile( FileName );

  if ( !outFile.is_open() )
    CV_Error(Error::StsError, String("Error opening output file: ") + String(FileName) + "\n");

  ////
  // Header
  ////

  const int pointNum = ( int ) PC.rows;
  const int vertNum  = ( int ) PC.cols;

  outFile << "ply" << std::endl;
  outFile << "format ascii 1.0" << std::endl;
  outFile << "element vertex " << pointNum << std::endl;
  outFile << "property float x" << std::endl;
  outFile << "property float y" << std::endl;
  outFile << "property float z" << std::endl;
  if (vertNum==6)
  {
    outFile << "property float nx" << std::endl;
    outFile << "property float ny" << std::endl;
    outFile << "property float nz" << std::endl;
  }
  outFile << "end_header" << std::endl;

  ////
  // Points
  ////

  for ( int pi = 0; pi < pointNum; ++pi )
  {
    const float* point = PC.ptr<float>(pi);

    outFile << point[0] << " " << point[1] << " " << point[2];

    if (vertNum==6)
    {
      outFile<<" " << point[3] << " "<<point[4]<<" "<<point[5];
    }

    outFile << std::endl;
  }

  return;
}

void writePLYVisibleNormals(Mat PC, const char* FileName)
{
  std::ofstream outFile(FileName);

  if (!outFile.is_open())
    CV_Error(Error::StsError, String("Error opening output file: ") + String(FileName) + "\n");

  ////
  // Header
  ////

  const int pointNum = (int)PC.rows;
  const int vertNum = (int)PC.cols;
  const bool hasNormals = vertNum == 6;

  outFile << "ply" << std::endl;
  outFile << "format ascii 1.0" << std::endl;
  outFile << "element vertex " << (hasNormals? 2*pointNum:pointNum) << std::endl;
  outFile << "property float x" << std::endl;
  outFile << "property float y" << std::endl;
  outFile << "property float z" << std::endl;
  if (hasNormals)
  {
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
  }
  outFile << "end_header" << std::endl;

  ////
  // Points
  ////

  for (int pi = 0; pi < pointNum; ++pi)
  {
    const float* point = PC.ptr<float>(pi);

    outFile << point[0] << " " << point[1] << " " << point[2];

    if (hasNormals)
    {
      outFile << " 127 127 127" << std::endl;
      outFile << point[0] + point[3] << " " << point[1] + point[4] << " " << point[2] + point[5];
      outFile << " 255 0 0";
    }

    outFile << std::endl;
  }

  return;
}

Mat samplePCUniform(Mat PC, int sampleStep)
{
  int numRows = PC.rows/sampleStep;
  Mat sampledPC = Mat(numRows, PC.cols, PC.type());

  int c=0;
  for (int i=0; i<PC.rows && c<numRows; i+=sampleStep)
  {
    PC.row(i).copyTo(sampledPC.row(c++));
  }

  return sampledPC;
}

Mat samplePCUniformInd(Mat PC, int sampleStep, std::vector<int> &indices)
{
  int numRows = cvRound((double)PC.rows/(double)sampleStep);
  indices.resize(numRows);
  Mat sampledPC = Mat(numRows, PC.cols, PC.type());

  int c=0;
  for (int i=0; i<PC.rows && c<numRows; i+=sampleStep)
  {
    indices[c] = i;
    PC.row(i).copyTo(sampledPC.row(c++));
  }

  return sampledPC;
}

void* indexPCFlann(Mat pc)
{
  Mat dest_32f;
  pc.colRange(0,3).copyTo(dest_32f);
  return new FlannIndex(dest_32f, cvflann::KDTreeSingleIndexParams(8));
}

void destroyFlann(void* flannIndex)
{
  delete ((FlannIndex*)flannIndex);
}

// For speed purposes this function assumes that PC, Indices and Distances are created with continuous structures
void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances)
{
  queryPCFlann(flannIndex, pc, indices, distances, 1);
}

void queryPCFlann(void* flannIndex, Mat& pc, Mat& indices, Mat& distances, const int numNeighbors)
{
  Mat obj_32f;
  pc.colRange(0, 3).copyTo(obj_32f);
  ((FlannIndex*)flannIndex)->knnSearch(obj_32f, indices, distances, numNeighbors, cvflann::SearchParams(32));
}

void queryPCFlannRadius(void* flannIndex, Mat& pc, Mat& indices, Mat& distances, double radius)
{
    Mat obj_32f;
    pc.colRange(0, 3).copyTo(obj_32f);
    ((FlannIndex*)flannIndex)->radiusSearch(obj_32f, indices, distances, radius, cvflann::SearchParams(32));
}

Mat samplePCByQuantization_normal(Mat pc, Vec2f& xrange, Vec2f& yrange, Vec2f& zrange, float sampleStep, float anglethreshold, int level)
{

    //设置网格参数
    float xr = xrange[1] - xrange[0] + 0.001;//x的跨度
    float yr = yrange[1] - yrange[0] + 0.001;
    float zr = zrange[1] - zrange[0] + 0.001;

    //std::cout << xr << " " << xr << " " << xr << " " << sampleStep << std::endl;

    int numPoints = 0;


    int xnumSamplesDim = (int)(xr / sampleStep) + 1;//采样宽度数
    int ynumSamplesDim = (int)(yr / sampleStep) + 1;//采样宽度数
    int znumSamplesDim = (int)(zr / sampleStep) + 1;//采样宽度数
    std::vector< std::vector<int> > map;

    map.resize((xnumSamplesDim + 1) * (ynumSamplesDim + 1) * (znumSamplesDim + 1));//设置行数


    //std::cout << xnumSamplesDim << "  vvvv " << ynumSamplesDim << "  vvvv " << znumSamplesDim << "  vvvv " << map.size() << std::endl;


    for (int i = 0; i < pc.rows; i++)
    {
        const float* point = pc.ptr<float>(i);

        const int xCell = (int)((float)xnumSamplesDim * (point[0] - xrange[0]) / xr);//计算x轴索引下标
        const int yCell = (int)((float)ynumSamplesDim * (point[1] - yrange[0]) / yr);//计算y轴索引下标
        const int zCell = (int)((float)znumSamplesDim * (point[2] - zrange[0]) / zr);//计算z轴索引下标
        const int index = xCell * ynumSamplesDim * znumSamplesDim + yCell * znumSamplesDim + zCell;//计算在二维向量组中的下标
        map[index].push_back(i);//把下标压入二维向量组								  //  }
    }
    //下采样

    Mat ypc = pc;
    Mat dpc = Mat(pc.rows, pc.cols, CV_32F);

    int row = 0;
    int mapsize = map.size();
    double cosanglethreshold = cos(anglethreshold);

    for (int lev = 0; lev < level; lev++)
    {
        row = 0;
        int nn = 0;
        int span = pow(2, lev);
        //std::cout << lev << " " << span << std::endl;
        for (int i = 0; i < xnumSamplesDim; i += span)
        {
            for (int j = 0; j < ynumSamplesDim; j += span)
            {
                for (int k = 0; k < znumSamplesDim; k += span)
                {
                    //
                    std::vector<Vec3f> normal;
                    std::vector<Vec3f> clustnormal;
                    std::vector<Vec3f> clustcoord;
                    std::vector<int> count;
                    std::vector<int> num;
                    std::vector<int> total_num;
                    //聚类
                    for (int i1 = 0; i1 < span && i + i1 < xnumSamplesDim; i1++)
                    {
                        for (int j1 = 0; j1 < span && j + j1 < ynumSamplesDim; j1++)
                        {
                            for (int k1 = 0; k1 < span && k + k1 < znumSamplesDim; k1++)
                            {
                                int index = (i + i1) * ynumSamplesDim * znumSamplesDim + (j + j1) * znumSamplesDim + (k + k1);
                                nn += map[index].size();
                                for (int n = 0; n < map[index].size(); n++)
                                {
                                    int yn = true;
                                    int m = map[index][n];

                                    Vec3f a(ypc.at<float>(m, 3), ypc.at<float>(m, 4), ypc.at<float>(m, 5));
                                    Vec3f c(ypc.at<float>(m, 0), ypc.at<float>(m, 1), ypc.at<float>(m, 2));
                                    for (int m = 0; m < normal.size(); m++)
                                    {
                                        float acosz = (a.dot(normal[m]));
                                        if (acosz > cosanglethreshold)
                                        {
                                            yn = false;
                                            clustnormal[m] = clustnormal[m] + a;
                                            clustcoord[m] = clustcoord[m] + c;
                                            count[m]++;
                                            break;
                                        }
                                    }

                                    if (yn)
                                    {
                                        normal.push_back(a);
                                        clustnormal.push_back(a);
                                        clustcoord.push_back(c);
                                        count.push_back(1);
                                    }
                                }



                            }

                        }
                    }
                    //清空
                    for (int i1 = 0; i1 < span && i + i1 < xnumSamplesDim; i1++)
                    {
                        for (int j1 = 0; j1 < span && j + j1 < ynumSamplesDim; j1++)
                        {
                            for (int k1 = 0; k1 < span && k + k1 < znumSamplesDim; k1++)
                            {
                                int index = (i + i1) * ynumSamplesDim * znumSamplesDim + (j + j1) * znumSamplesDim + (k + k1);
                                map[index].clear();
                                //std::cout << map[index].size()<<std::endl;
                            }
                        }

                    }



                    //插入
                    int index = (i)*ynumSamplesDim * znumSamplesDim + (j)*znumSamplesDim + (k);
                    for (int i1 = 0; i1 < clustnormal.size(); i1++)
                    {
                        double norm = cv::norm(clustnormal[i1]);
                        if (norm > 0.000001) {
                            Vec3f a = clustnormal[i1] / norm;

                            dpc.at<float>(row, 0) = clustcoord[i1](0) / count[i1];
                            dpc.at<float>(row, 1) = clustcoord[i1](1) / count[i1];
                            dpc.at<float>(row, 2) = clustcoord[i1](2) / count[i1];
                            dpc.at<float>(row, 3) = a(0);
                            dpc.at<float>(row, 4) = a(1);
                            dpc.at<float>(row, 5) = a(2);
                            /*if (count[i1] != total_num[i1])
                            {
                                std::cout << count[i1] << "  " << total_num[i1] << std::endl;
                                system("pause");
                            }*/


                            map[index].push_back(row);
                            row++;
                        }
                        /*else
                        {
                            std::cout << total_num[i1] << std::endl;
                        }*/

                    }
                }

            }
        }


        //改写数据
        ypc = dpc.rowRange(0, row).colRange(0, 6);
    }

    return ypc;
}

Mat samplePCByQuantization_cube(Mat pc, Vec2f& xrange, Vec2f& yrange, Vec2f& zrange, float sampleStep, int weightByCenter)
{
    std::vector< std::vector<int> > map;

    //int numSamplesDim = (int)(1.0/sampleStep);

    float xr = xrange[1] - xrange[0];
    float yr = yrange[1] - yrange[0];
    float zr = zrange[1] - zrange[0];

    // 使用方格进行采样
    int xnumSamplesDim = (int)(xr / sampleStep);
    int ynumSamplesDim = (int)(yr / sampleStep);
    int znumSamplesDim = (int)(zr / sampleStep);

    int numPoints = 0;

    map.resize((xnumSamplesDim + 1) * (ynumSamplesDim + 1) * (znumSamplesDim + 1));

    // OpenMP might seem like a good idea, but it didn't speed this up for me
    //#pragma omp parallel for
    for (int i = 0; i < pc.rows; i++)
    {
        const float* point = pc.ptr<float>(i);

        // quantize a point
        const int xCell = (int)((float)xnumSamplesDim * (point[0] - xrange[0]) / xr);
        const int yCell = (int)((float)ynumSamplesDim * (point[1] - yrange[0]) / yr);
        const int zCell = (int)((float)znumSamplesDim * (point[2] - zrange[0]) / zr);
        const int index = xCell * ynumSamplesDim * znumSamplesDim + yCell * znumSamplesDim + zCell;

        /*#pragma omp critical
            {*/
        map[index].push_back(i);
        //  }
    }

    for (unsigned int i = 0; i < map.size(); i++)
    {
        numPoints += (map[i].size() > 0);
    }

    Mat pcSampled = Mat(numPoints, pc.cols, CV_32F);
    int c = 0;

    for (unsigned int i = 0; i < map.size(); i++)
    {
        double px = 0, py = 0, pz = 0;
        double nx = 0, ny = 0, nz = 0;

        std::vector<int> curCell = map[i];
        int cn = (int)curCell.size();
        if (cn > 0)
        {
            if (weightByCenter)
            {
                int xCell, yCell, zCell;
                double xc, yc, zc;
                double weightSum = 0;

                zCell = i % znumSamplesDim;//计算点云点的下标索引
                yCell = ((i - zCell) / znumSamplesDim) % ynumSamplesDim;
                xCell = ((i - zCell - yCell * ynumSamplesDim) / (ynumSamplesDim * znumSamplesDim));
                xc = ((double)xCell + 0.5) * (double)xr / xnumSamplesDim + (double)xrange[0];//计算单元格中心坐标
                yc = ((double)yCell + 0.5) * (double)yr / ynumSamplesDim + (double)yrange[0];
                zc = ((double)zCell + 0.5) * (double)zr / znumSamplesDim + (double)zrange[0];

                for (int j = 0; j < cn; j++)
                {
                    const int ptInd = curCell[j];
                    float* point = pc.ptr<float>(ptInd);
                    const double dx = point[0] - xc;
                    const double dy = point[1] - yc;
                    const double dz = point[2] - zc;
                    const double d = sqrt(dx * dx + dy * dy + dz * dz);
                    double w = 0;

                    if (d > EPS)
                    {
                        // it is possible to use different weighting schemes.
                        // inverse weigthing was just good for me
                        // exp( - (distance/h)**2 )
                        //const double w = exp(-d*d);
                        w = 1.0 / d;
                    }

                    //float weights[3]={1,1,1};
                    px += w * (double)point[0];
                    py += w * (double)point[1];
                    pz += w * (double)point[2];
                    nx += w * (double)point[3];
                    ny += w * (double)point[4];
                    nz += w * (double)point[5];

                    weightSum += w;
                }
                px /= (double)weightSum;
                py /= (double)weightSum;
                pz /= (double)weightSum;
                nx /= (double)weightSum;
                ny /= (double)weightSum;
                nz /= (double)weightSum;
            }
            else
            {
                for (int j = 0; j < cn; j++)
                {
                    const int ptInd = curCell[j];
                    float* point = pc.ptr<float>(ptInd);

                    px += (double)point[0];
                    py += (double)point[1];
                    pz += (double)point[2];
                    nx += (double)point[3];
                    ny += (double)point[4];
                    nz += (double)point[5];
                }

                px /= (double)cn;
                py /= (double)cn;
                pz /= (double)cn;
                nx /= (double)cn;
                ny /= (double)cn;
                nz /= (double)cn;

            }

            float* pcData = pcSampled.ptr<float>(c);
            pcData[0] = (float)px;
            pcData[1] = (float)py;
            pcData[2] = (float)pz;

            // normalize the normals
            double norm = sqrt(nx * nx + ny * ny + nz * nz);

            if (norm > EPS)
            {
                pcData[3] = (float)(nx / norm);
                pcData[4] = (float)(ny / norm);
                pcData[5] = (float)(nz / norm);
            }
            else
            {
                pcData[3] = 0.0f;
                pcData[4] = 0.0f;
                pcData[5] = 0.0f;
            }
            //#pragma omp atomic
            c++;

            curCell.clear();
        }
    }

    map.clear();
    return pcSampled;
}

// uses a volume instead of an octree
// TODO: Right now normals are required.
// This is much faster than sample_pc_octree
Mat samplePCByQuantization(Mat pc, Vec2f& xrange, Vec2f& yrange, Vec2f& zrange, float sampleStep, int weightByCenter)
{
  std::vector< std::vector<int> > map;

  int numSamplesDim = (int)(1.0/sampleStep);

  float xr = xrange[1] - xrange[0];
  float yr = yrange[1] - yrange[0];
  float zr = zrange[1] - zrange[0];

  int numPoints = 0;

  map.resize((numSamplesDim+1)*(numSamplesDim+1)*(numSamplesDim+1));

  // OpenMP might seem like a good idea, but it didn't speed this up for me
  //#pragma omp parallel for
  for (int i=0; i<pc.rows; i++)
  {
    const float* point = pc.ptr<float>(i);

    // quantize a point
    const int xCell =(int) ((float)numSamplesDim*(point[0]-xrange[0])/xr);
    const int yCell =(int) ((float)numSamplesDim*(point[1]-yrange[0])/yr);
    const int zCell =(int) ((float)numSamplesDim*(point[2]-zrange[0])/zr);
    const int index = xCell*numSamplesDim*numSamplesDim+yCell*numSamplesDim+zCell;

    /*#pragma omp critical
        {*/
    map[index].push_back(i);
    //  }
  }

  for (unsigned int i=0; i<map.size(); i++)
  {
    numPoints += (map[i].size()>0);
  }

  Mat pcSampled = Mat(numPoints, pc.cols, CV_32F);
  int c = 0;

  for (unsigned int i=0; i<map.size(); i++)
  {
    double px=0, py=0, pz=0;
    double nx=0, ny=0, nz=0;

    std::vector<int> curCell = map[i];
    int cn = (int)curCell.size();
    if (cn>0)
    {
      if (weightByCenter)
      {
        int xCell, yCell, zCell;
        double xc, yc, zc;
        double weightSum = 0 ;
        zCell = i % numSamplesDim;
        yCell = ((i-zCell)/numSamplesDim) % numSamplesDim;
        xCell = ((i-zCell-yCell*numSamplesDim)/(numSamplesDim*numSamplesDim));

        xc = ((double)xCell+0.5) * (double)xr/numSamplesDim + (double)xrange[0];
        yc = ((double)yCell+0.5) * (double)yr/numSamplesDim + (double)yrange[0];
        zc = ((double)zCell+0.5) * (double)zr/numSamplesDim + (double)zrange[0];

        for (int j=0; j<cn; j++)
        {
          const int ptInd = curCell[j];
          float* point = pc.ptr<float>(ptInd);
          const double dx = point[0]-xc;
          const double dy = point[1]-yc;
          const double dz = point[2]-zc;
          const double d = sqrt(dx*dx+dy*dy+dz*dz);
          double w = 0;

          if (d>EPS)
          {
            // it is possible to use different weighting schemes.
            // inverse weigthing was just good for me
            // exp( - (distance/h)**2 )
            //const double w = exp(-d*d);
            w = 1.0/d;
          }

          //float weights[3]={1,1,1};
          px += w*(double)point[0];
          py += w*(double)point[1];
          pz += w*(double)point[2];
          nx += w*(double)point[3];
          ny += w*(double)point[4];
          nz += w*(double)point[5];

          weightSum+=w;
        }
        px/=(double)weightSum;
        py/=(double)weightSum;
        pz/=(double)weightSum;
        nx/=(double)weightSum;
        ny/=(double)weightSum;
        nz/=(double)weightSum;
      }
      else
      {
        for (int j=0; j<cn; j++)
        {
          const int ptInd = curCell[j];
          float* point = pc.ptr<float>(ptInd);

          px += (double)point[0];
          py += (double)point[1];
          pz += (double)point[2];
          nx += (double)point[3];
          ny += (double)point[4];
          nz += (double)point[5];
        }

        px/=(double)cn;
        py/=(double)cn;
        pz/=(double)cn;
        nx/=(double)cn;
        ny/=(double)cn;
        nz/=(double)cn;

      }

      float *pcData = pcSampled.ptr<float>(c);
      pcData[0]=(float)px;
      pcData[1]=(float)py;
      pcData[2]=(float)pz;

      // normalize the normals
      double norm = sqrt(nx*nx+ny*ny+nz*nz);

      if (norm>EPS)
      {
        pcData[3]=(float)(nx/norm);
        pcData[4]=(float)(ny/norm);
        pcData[5]=(float)(nz/norm);
      }
      else
      {
        pcData[3]=0.0f;
        pcData[4]=0.0f;
        pcData[5]=0.0f;
      }
      //#pragma omp atomic
      c++;

      curCell.clear();
    }
  }

  map.clear();
  return pcSampled;
}

void shuffle(int *array, size_t n)
{
  size_t i;
  for (i = 0; i < n - 1; i++)
  {
    size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
    int t = array[j];
    array[j] = array[i];
    array[i] = t;
  }
}

// compute the standard bounding box
void computeBboxStd(Mat pc, Vec2f& xRange, Vec2f& yRange, Vec2f& zRange)
{
  Mat pcPts = pc.colRange(0, 3);
  int num = pcPts.rows;

  float* points = (float*)pcPts.data;

  xRange[0] = points[0];
  xRange[1] = points[0];
  yRange[0] = points[1];
  yRange[1] = points[1];
  zRange[0] = points[2];
  zRange[1] = points[2];

  for  ( int  ind = 0; ind < num; ind++ )
  {
    const float* row = (float*)(pcPts.data + (ind * pcPts.step));
    const float x = row[0];
    const float y = row[1];
    const float z = row[2];

    if (x<xRange[0])
      xRange[0]=x;
    if (x>xRange[1])
      xRange[1]=x;

    if (y<yRange[0])
      yRange[0]=y;
    if (y>yRange[1])
      yRange[1]=y;

    if (z<zRange[0])
      zRange[0]=z;
    if (z>zRange[1])
      zRange[1]=z;
  }
}

Mat normalizePCCoeff(Mat pc, float scale, float* Cx, float* Cy, float* Cz, float* MinVal, float* MaxVal)
{
  double minVal=0, maxVal=0;

  Mat x,y,z, pcn;
  pc.col(0).copyTo(x);
  pc.col(1).copyTo(y);
  pc.col(2).copyTo(z);

  float cx = (float) cv::mean(x)[0];
  float cy = (float) cv::mean(y)[0];
  float cz = (float) cv::mean(z)[0];

  cv::minMaxIdx(pc, &minVal, &maxVal);

  x=x-cx;
  y=y-cy;
  z=z-cz;
  pcn.create(pc.rows, 3, CV_32FC1);
  x.copyTo(pcn.col(0));
  y.copyTo(pcn.col(1));
  z.copyTo(pcn.col(2));

  cv::minMaxIdx(pcn, &minVal, &maxVal);
  pcn=(float)scale*(pcn)/((float)maxVal-(float)minVal);

  *MinVal=(float)minVal;
  *MaxVal=(float)maxVal;
  *Cx=(float)cx;
  *Cy=(float)cy;
  *Cz=(float)cz;

  return pcn;
}

Mat transPCCoeff(Mat pc, float scale, float Cx, float Cy, float Cz, float MinVal, float MaxVal)
{
  Mat x,y,z, pcn;
  pc.col(0).copyTo(x);
  pc.col(1).copyTo(y);
  pc.col(2).copyTo(z);

  x=x-Cx;
  y=y-Cy;
  z=z-Cz;
  pcn.create(pc.rows, 3, CV_32FC1);
  x.copyTo(pcn.col(0));
  y.copyTo(pcn.col(1));
  z.copyTo(pcn.col(2));

  pcn=(float)scale*(pcn)/((float)MaxVal-(float)MinVal);

  return pcn;
}

Mat transformPCPose(Mat pc, const Matx44d& Pose)
{
  Mat pct = Mat(pc.rows, pc.cols, CV_32F);

  Matx33d R;
  Vec3d t;
  poseToRT(Pose, R, t);

#if defined _OPENMP
#pragma omp parallel for
#endif
  for (int i=0; i<pc.rows; i++)
  {
    const float *pcData = pc.ptr<float>(i);
    const Vec3f n1(&pcData[3]);

    Vec4d p = Pose * Vec4d(pcData[0], pcData[1], pcData[2], 1);
    Vec3d p2(p.val);

    // p2[3] should normally be 1
    if (fabs(p[3]) > EPS)
    {
      Mat((1.0 / p[3]) * p2).reshape(1, 1).convertTo(pct.row(i).colRange(0, 3), CV_32F);
    }

    // If the point cloud has normals,
    // then rotate them as well
    if (pc.cols == 6)
    {
      Vec3d n(n1), n2;

      n2 = R * n;
      double nNorm = cv::norm(n2);

      if (nNorm > EPS)
      {
        Mat((1.0 / nNorm) * n2).reshape(1, 1).convertTo(pct.row(i).colRange(3, 6), CV_32F);
      }
    }
  }

  return pct;
}

Mat genRandomMat(int rows, int cols, double mean, double stddev, int type)
{
  Mat meanMat = mean*Mat::ones(1,1,type);
  Mat sigmaMat= stddev*Mat::ones(1,1,type);
  RNG rng(time(0));
  Mat matr(rows, cols,type);
  rng.fill(matr, RNG::NORMAL, meanMat, sigmaMat);

  return matr;
}

void getRandQuat(Vec4d& q)
{
  q[0] = (float)rand()/(float)(RAND_MAX);
  q[1] = (float)rand()/(float)(RAND_MAX);
  q[2] = (float)rand()/(float)(RAND_MAX);
  q[3] = (float)rand()/(float)(RAND_MAX);

  q *= 1.0 / cv::norm(q);
  q[0]=fabs(q[0]);
}

void getRandomRotation(Matx33d& R)
{
  Vec4d q;
  getRandQuat(q);
  quatToDCM(q, R);
}

void getRandomPose(Matx44d& Pose)
{
  Matx33d R;
  Vec3d t;

  srand((unsigned int)time(0));
  getRandomRotation(R);

  t[0] = (float)rand()/(float)(RAND_MAX);
  t[1] = (float)rand()/(float)(RAND_MAX);
  t[2] = (float)rand()/(float)(RAND_MAX);

  rtToPose(R,t,Pose);
}

Mat addNoisePC(Mat pc, double scale)
{
  Mat randT = genRandomMat(pc.rows,pc.cols,0,scale,CV_32FC1);
  return randT + pc;
}

/*
The routines below use the eigenvectors of the local covariance matrix
to compute the normals of a point cloud.
The algorithm uses FLANN and Joachim Kopp's fast 3x3 eigenvector computations
to improve accuracy and increase speed
Also, view point flipping as in point cloud library is implemented
*/

void meanCovLocalPC(const Mat& pc, const int point_count, Matx33d& CovMat, Vec3d& Mean)
{
  cv::calcCovarMatrix(pc.rowRange(0, point_count), CovMat, Mean, cv::COVAR_NORMAL | cv::COVAR_ROWS);
  CovMat *= 1.0 / (point_count - 1);
}

void meanCovLocalPCInd(const Mat& pc, const int* Indices, const int point_count, Matx33d& CovMat, Vec3d& Mean)
{
  int i, j, k;

  CovMat = Matx33d::all(0);
  Mean = Vec3d::all(0);
  for (i = 0; i < point_count; ++i)
  {
    const float* cloud = pc.ptr<float>(Indices[i]);
    for (j = 0; j < 3; ++j)
    {
      for (k = 0; k < 3; ++k)
        CovMat(j, k) += cloud[j] * cloud[k];
      Mean[j] += cloud[j];
    }
  }
  Mean *= 1.0 / point_count;
  CovMat *= 1.0 / point_count;

  for (j = 0; j < 3; ++j)
    for (k = 0; k < 3; ++k)
      CovMat(j, k) -= Mean[j] * Mean[k];
}

int computeNormalsPC3d(const Mat& PC, Mat& PCNormals, const int NumNeighbors, const bool FlipViewpoint, const Vec3f& viewpoint)
{
  int i;

  if (PC.cols!=3 && PC.cols!=6) // 3d data is expected
  {
    //return -1;
    CV_Error(cv::Error::BadImageSize, "PC should have 3 or 6 elements in its columns");
  }

  PCNormals.create(PC.rows, 6, CV_32F);
  Mat PCInput = PCNormals.colRange(0, 3);
  Mat Distances(PC.rows, NumNeighbors, CV_32F);
  Mat Indices(PC.rows, NumNeighbors, CV_32S);

  PC.rowRange(0, PC.rows).colRange(0, 3).copyTo(PCNormals.rowRange(0, PC.rows).colRange(0, 3));

  void* flannIndex = indexPCFlann(PCInput);

  queryPCFlann(flannIndex, PCInput, Indices, Distances, NumNeighbors);
  destroyFlann(flannIndex);
  flannIndex = 0;

#if defined _OPENMP
#pragma omp parallel for
#endif
  for (i=0; i<PC.rows; i++)
  {
    Matx33d C;
    Vec3d mu;
    const int* indLocal = Indices.ptr<int>(i);

    // compute covariance matrix
    meanCovLocalPCInd(PCNormals, indLocal, NumNeighbors, C, mu);

    // eigenvectors of covariance matrix
    Mat eigVect, eigVal;
    eigen(C, eigVal, eigVect);
    eigVect.row(2).convertTo(PCNormals.row(i).colRange(3, 6), CV_32F);

    if (FlipViewpoint)
    {
      Vec3f nr(PCNormals.ptr<float>(i) + 3);
      Vec3f pci(PCNormals.ptr<float>(i));
      flipNormalViewpoint(pci, viewpoint, nr);
      Mat(nr).reshape(1, 1).copyTo(PCNormals.row(i).colRange(3, 6));
    }
  }

  return 1;
}

} // namespace ppf_match_3d

} // namespace cv
