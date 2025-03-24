

#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"
#include "src/c_utils.hpp"

#include <Windows.h>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include "_deps/Eigen/Dense"
#include <opencv2/core/eigen.hpp>
#include <map>

using namespace std;
using namespace cv;
using namespace ppf_match_3d;

int evalUwa() {
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/UWA/";
    string configPath = "3D models/Mian/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/Halcon_benchmark/halconResults/uwa/";



    //  
    string cfgPath0 = rootPath + configPath + "ConfigScene1.ini";//cfgName
    LPCTSTR cfgPath = cfgPath0.c_str();
    
    // 多个模型
    vector<string> uwaModelName{ "parasaurolophus_high", "cheff", "chicken_high", "T-rex_high" };
    //cout << uwaModelName[0] << endl;
    map< string, vector<double>> ADDMap;
    map< string, vector<double>> phiMap;
    map< string, vector<double>> dNormMap;

    LPSTR  modelNumCh = new char[1024];
    GetPrivateProfileString("MODELS", "NUMBER", "NULL", modelNumCh, 512, cfgPath);
    int modelNum = atoi(modelNumCh);
    string modelKey, modelGTKey;
    for (int i = 0; i < modelNum; i++) {
        modelKey = "MODEL_" + to_string(i);
        modelGTKey = modelKey + "_GROUNDTRUTH";  
    }

    // 单个模型
    // read mdoel
    LPSTR  modelPath0 = new char[1024];
    GetPrivateProfileString("MODELS", modelKey.c_str(), "NULL", modelPath0, 512, cfgPath);
    string mPath1 = rootPath + modelPath0;

    string mPath = mPath1.substr(0, mPath1.length()-4) + "_0.ply";
    Mat pc = loadPLYSimple(mPath.c_str(), 1);
    Vec3f p1(pc.ptr<float>(0));
    

    // read gt 
    LPSTR  gtPath0 = new char[1024];
    GetPrivateProfileString("MODELS", modelGTKey.c_str(), "NULL", gtPath0, 512, cfgPath);
    string gtPath = rootPath + gtPath0;

    std::vector<std::string> mStr, mn;
    boost::split(mStr, gtPath0, boost::is_any_of("/"));
    boost::split(mn, *(mStr.end() - 1), boost::is_any_of("."));
    string gtName = mn[0];

    ifstream gt_ifs(gtPath);
    if (!gt_ifs.is_open()) exit(1);
    Matx44d gt_pose;
    for (int ii = 0; ii < 4; ii++)
        for (int jj = 0; jj < 4; jj++)
        {
            gt_ifs >> gt_pose(ii, jj);
        }

    // read pred
    string predFilePath = predPath + gtName + ".txt";
    ifstream pred_ifs(predFilePath);
    if (!pred_ifs.is_open()) exit(1);

    string modelName;
    getline(pred_ifs, modelName);
    string timeStr;
    getline(pred_ifs, timeStr);
    int n = timeStr.find("=");
    string timeStr2 = timeStr.substr(n + 1, timeStr.length() - n);
    cout << timeStr2 << endl;
    double time = stod(timeStr2, 0);
    Matx44d pred_pose;
    for (int ii = 0; ii < 4; ii++)
        for (int jj = 0; jj < 4; jj++)
        {
            pred_ifs >> pred_pose(ii, jj);
        }


    // ADD
    Mat pct_gt = transformPCPose(pc, gt_pose); //pc是原始模型
    Mat pct_pred = transformPCPose(pc, pred_pose); //pc是原始模型

    double totalD = 0;
    for (int ii = 0; ii < pct_gt.rows; ii++)
    {
        Vec3f v1(pct_gt.ptr<float>(ii));
        //const Vec3f n1(pct_gt.ptr<float>(ii) + 3);
        Vec3f v2(pct_pred.ptr<float>(ii));
        v1 = v1 - v2;
        totalD += cv::norm(v1);
    }
    totalD /= pct_gt.rows;

    // manifold 
    Eigen::Matrix<double, 4, 4> gtMatrix;
    cv::cv2eigen(gt_pose, gtMatrix); // cv::Mat 转换成 Eigen::Matrix
    Eigen::Affine3f gtPose;
    gtPose.matrix() = gtMatrix.cast<float>();
    //cout << gtPose.rotation() << endl;
    //cout << gtPose.translation() << endl;

    

    Eigen::Matrix<double, 4, 4> predMatrix;
    cv::cv2eigen(pred_pose, predMatrix); // cv::Mat 转换成 Eigen::Matrix
    Eigen::Affine3f predPose;
    predPose.matrix() = predMatrix.cast<float>();

    Eigen::Matrix3f RtRsInv(gtPose.rotation().inverse().lazyProduct(predPose.rotation()).eval());
    Eigen::AngleAxisf rotation_diff_mat(RtRsInv); //tr(Rs.inv * Rt) = tr(Rt * Rs.inv)
    double phi = std::abs(rotation_diff_mat.angle());
    float dNorm = (gtPose.translation() - predPose.translation()).norm();


    return 0;
}

void evalKinect() {
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/Kinect/";
    string configPath = "/3D models/CVLab/Kinect/ObjectRecognition/Scenes/2011_06_27/configwithbestview/";
    string rePath = "D:/wenhao.sun/Documents/GitHub/1-project/Halcon_benchmark/halconResults/kinect/";


}

int main() {
    #if (defined __x86_64__ || defined _M_X64)
        cout << "Running on 64 bits" << endl;
    #else
        cout << "Running on 32 bits" << endl;
    #endif

    #ifdef _OPENMP
        cout << "Running with OpenMP" << endl;
    #else
        cout << "Running without OpenMP and without TBB" << endl;
    #endif

    evalUwa();
}

int main2(){
    string path = "D:\\wenhao.sun\\Documents\\GitHub\\1-project\\ppf_2025\\test.txt";
    ifstream infile;
    infile.open(path);
    vector<int> pose;
    pose.resize(16);
    for (int i = 0; i < 10; i++) {
        string str;
        getline(infile, str);
        for (int j = 0; j < 16; j++) {
            infile >> pose[j];
        }
        getline(infile, str);
        cout << str;
    }
    return 0;
}


int main1(int argc, char** argv)
{
    // 计算距离
    Vec3f a0(-14.92602151836492, -650.8811157495577, -966.8551517443179);
    Vec3f a1(56.88654298232103, -132.3572823089681, -1263.999500987719);
    Vec3f a2(-64.14969939427944, -502.0570831202151, -1155.696617291158);
    float dist = cv::norm(a0 - a2);



    //cv::Mat mat3 = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9); //枚举赋值
    cv::Mat Pose0 = (cv::Mat_<double>(4, 4) << 
        0.9875361520229935, -0.0499953598989327, 0.1492407867715634, -14.92602151836492,
        0.1054792992986845, -0.4935452838505936, -0.8633001622890474, -650.8811157495577,
        0.1168180887837051, 0.8682819339232005, -0.4821203349325862, -966.8551517443179,
        0, 0, 0, 1);
    cv::Mat Pose2 = (cv::Mat_<double>(4, 4) <<
        0.9954723249325371, -0.06285915580300847, 0.07129920634300013, -64.14969939427944,
        -0.004839312671306641, -0.7826525632536144, -0.6224399941242654, -502.0570831202151,
        0.09492855917097834, 0.6192767489291676, -0.7794131618656707, -1155.696617291158,
        0, 0, 0, 1);
    //daltaT = T2 * inverse(T1), p->T1*p, p->T2*p, T1*p->T2*p, daltaT*T1*p = T2*p
    Mat daltaT = Pose2 * Pose0.inv();
    cv::Matx44d dt;
    daltaT.copyTo(dt);
    cout << dt << endl;

    Matx33d R;
    Vec3d t;
    poseToRT(dt, R, t);
    Matx33d rtr = R.t() * R;
    cout << rtr << endl;

    cout << t << endl;
    cout << cv::norm(t) << endl;

    return 0;
    
}
