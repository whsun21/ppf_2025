

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
#include <iostream>

using namespace std;
using namespace cv;
using namespace ppf_match_3d;

int writeMap(const map<string, vector<double>> & m, const string & outPath);
int readMap(map<string, vector<double>>& m2, const string& inPath);

int evalUwa() {
    cout << "eval UWA" << endl;
    string dataName = "UWA";

    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/UWA/";
    string configPath = "3D models/Mian/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/Halcon_benchmark/halconResults/uwa/";

    vector<string> uwaModelName{ "parasaurolophus_high", "cheff", "chicken_high", "T-rex_high" };
    //cout << uwaModelName[0] << endl;
    map< string, vector<double>> ADDMap;
    map< string, vector<double>> phiMap;
    map< string, vector<double>> dNormMap;
    map< string, vector<double>> timeMap;
    for (int i = 0; i < uwaModelName.size(); i++) {
        ADDMap[uwaModelName[i]] = vector<double>();
        phiMap[uwaModelName[i]] = vector<double>();
        dNormMap[uwaModelName[i]] = vector<double>();
        timeMap[uwaModelName[i]] = vector<double>();
    }
    string ADDName = "../eval_"+ dataName + (string)"/eval_ADD.txt";
    string phiName = "../eval_" + dataName + (string)"/eval_phi.txt";
    string dNormName = "../eval_" + dataName + (string)"/eval_dNorm.txt";
    string timeName = "../eval_" + dataName + (string)"/eval_time.txt";

    // 所有场景
    string cfgsFile = rootPath + configPath + "configFilesList.txt";
    vector<string> cfgNameAll;
    ifstream cfg_ifs(cfgsFile);
    string cfgName0;
    while (getline(cfg_ifs, cfgName0)) {
        cfgNameAll.push_back(cfgName0);
    }
    cfg_ifs.close();

    for (auto & cfgName : cfgNameAll) {
        // 单个场景
        string cfgPath0 = rootPath + configPath + cfgName;//cfgName
        LPCTSTR cfgPath = cfgPath0.c_str();

        // 多个模型

        LPSTR  modelNumCh = new char[1024];
        GetPrivateProfileString("MODELS", "NUMBER", "NULL", modelNumCh, 512, cfgPath);
        int modelNum = atoi(modelNumCh);
        string modelKey, modelGTKey;
        for (int i = 0; i < modelNum; i++) {
            modelKey = "MODEL_" + to_string(i);
            modelGTKey = modelKey + "_GROUNDTRUTH";


            // 单个模型
            // read mdoel
            LPSTR  modelPath0 = new char[1024];
            GetPrivateProfileString("MODELS", modelKey.c_str(), "NULL", modelPath0, 512, cfgPath);
            string mPath1 = rootPath + modelPath0;

            string modelNameInCfg;
            {
                std::vector<std::string> mStr, mn;
                boost::split(mStr, mPath1, boost::is_any_of("/"));
                boost::split(mn, *(mStr.end() - 1), boost::is_any_of("."));
                modelNameInCfg = mn[0];
            }


            string mPath = mPath1.substr(0, mPath1.length() - 4) + "_0.ply";
            Mat pc = loadPLYSimple(mPath.c_str(), 1);
            Vec3f p1(pc.ptr<float>(0));


            // read gt 
            LPSTR  gtPath0 = new char[1024];
            GetPrivateProfileString("MODELS", modelGTKey.c_str(), "NULL", gtPath0, 512, cfgPath);
            string gtPath = rootPath + gtPath0;
            string gtName;
            {
                std::vector<std::string> mStr, mn;
                boost::split(mStr, gtPath0, boost::is_any_of("/"));
                boost::split(mn, *(mStr.end() - 1), boost::is_any_of("."));
                gtName = mn[0];
            }

            ifstream gt_ifs(gtPath);
            if (!gt_ifs.is_open()) { cout << "not open: " << gtPath << endl; exit(1); }
            Matx44d gt_pose;
            for (int ii = 0; ii < 4; ii++)
                for (int jj = 0; jj < 4; jj++)
                {
                    gt_ifs >> gt_pose(ii, jj);
                }
            gt_ifs.close();

            // read pred
            string predFilePath = predPath + gtName + ".txt";
            ifstream pred_ifs(predFilePath);
            if (!pred_ifs.is_open()) exit(1);

            string modelNameInPred;
            getline(pred_ifs, modelNameInPred); ////////
            if (modelNameInPred != modelNameInCfg) { cout << "not same: " << "gt " << gtPath << "pred " << predFilePath << endl; exit(1); }
            string timeStr;
            getline(pred_ifs, timeStr);
            int n = timeStr.find("=");
            string timeStr2 = timeStr.substr(n + 1, timeStr.length() - n);
            //cout << timeStr2 << endl;
            double time = stod(timeStr2, 0);
            timeMap[modelNameInPred].push_back(time);

            Matx44d pred_pose;
            for (int ii = 0; ii < 4; ii++)
                for (int jj = 0; jj < 4; jj++)
                {
                    pred_ifs >> pred_pose(ii, jj);
                }
            pred_ifs.close();

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
            ADDMap[modelNameInCfg].push_back(totalD);

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
            phiMap[modelNameInCfg].push_back(phi);
            dNormMap[modelNameInCfg].push_back(dNorm);
        }
    }


    // 保存误差结果
    writeMap(ADDMap, ADDName);
    writeMap(phiMap, phiName);
    writeMap(dNormMap, dNormName);
    writeMap(timeMap, timeName);
    
    int instancesNum(0);
    for (auto it = ADDMap.begin(); it != ADDMap.end(); ++it) {
        instancesNum += it->second.size();
        cout << it->first << ": " << it->second.size() << endl;
    }
    cout <<  "Total: " << instancesNum << endl;  //188


    return 0;
}

void evalKinect() {
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/Kinect/";
    string configPath = "/3D models/CVLab/Kinect/ObjectRecognition/Scenes/2011_06_27/configwithbestview/";
    string rePath = "D:/wenhao.sun/Documents/GitHub/1-project/Halcon_benchmark/halconResults/kinect/";


}

int rateUwa() {
    cout << "rate UWA" << endl;
    string dataName = "UWA";

    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/UWA/";
    string configPath = "3D models/Mian/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/Halcon_benchmark/halconResults/uwa/";

    vector<string> uwaModelName{ "parasaurolophus_high", "cheff", "chicken_high", "T-rex_high" };
    //cout << uwaModelName[0] << endl;
    map< string, vector<double>> ADDMap;
    map< string, vector<double>> phiMap;
    map< string, vector<double>> dNormMap;
    map< string, vector<double>> timeMap;
    for (int i = 0; i < uwaModelName.size(); i++) {
        ADDMap[uwaModelName[i]] = vector<double>();
        phiMap[uwaModelName[i]] = vector<double>();
        dNormMap[uwaModelName[i]] = vector<double>();
        timeMap[uwaModelName[i]] = vector<double>();
    }
    string ADDName = "../eval_" + dataName + (string)"/eval_ADD.txt";
    string phiName = "../eval_" + dataName + (string)"/eval_phi.txt";
    string dNormName = "../eval_" + dataName + (string)"/eval_dNorm.txt";
    string timeName = "../eval_" + dataName + (string)"/eval_time.txt";

    readMap(ADDMap, ADDName);
    readMap(phiMap, phiName);
    readMap(dNormMap, dNormName);
    readMap(timeMap, timeName);

    Vector3d v1 = results[modle_id][0].Pose.block(0, 0, 3, 3)
        * detectortest.model_para.model[modle_id].model_center + results[modle_id][0].Pose.block(0, 3, 3, 1);
    Vector3d v2 = tpose.block(0, 0, 3, 3)
        * detectortest.model_para.model[modle_id].model_center + tpose.block(0, 3, 3, 1);

    int yn = 1;
    if ((v1 - v2).norm() > detectortest.model_para.model[modle_id].model_diameter * 0.1 ||
        totald / pct1.rows() > detectortest.model_para.model[modle_id].model_diameter * 0.1)
        yn = 0;

    int yn2 = 1;
    if ((v1 - v2).norm() > detectortest.model_para.model[modle_id].model_diameter * 0.2 ||
        totald / pct1.rows() > detectortest.model_para.model[modle_id].model_diameter * 0.2)
        yn2 = 0;

    int yn3 = 1;
    if ((v1 - v2).norm() > detectortest.model_para.model[modle_id].model_diameter * 0.3 ||
        totald / pct1.rows() > detectortest.model_para.model[modle_id].model_diameter * 0.3)
        yn3 = 0;


    return 0;
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

    //evalUwa();
        rateUwa();

    
        //writeMap();
        //readMap();
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


int writeMap(const map<string, vector<double>> & m, const string & outPath) {
    //map<string, vector<double>> m = { {"1th", vector<double>{0,1,2}} , {"2th", vector<double>{0,1,2}} };
    // 存入文件out.txt
    ofstream of(outPath);
    for (const auto& i : m) {
        of << i.first << ' ';
        for (auto & v: i.second)
            of << v << ' ';
        of << std::endl;

    }
    of.close();
    return 0;
}

int readMap(map<string, vector<double>>& m2, const string& inPath) {
    // 读取文件，存入map m2中
    //map<string, vector<double>> m2;
    ifstream iff (inPath);
    if (!iff.is_open()) { cout << "not open: " << inPath << endl; exit(1); }
    string keyval;
    while (getline(iff, keyval)) {
        std::vector<std::string> mStr;
        boost::split(mStr, keyval, boost::is_any_of(" "));
        for (int i = 1; i < mStr.size()-1; i++)
            m2[mStr[0]].push_back(stod(mStr[i]));
    }
    iff.close();
    return 0;
}