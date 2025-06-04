

#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"
#include "src/c_utils.hpp"
//#include "src/precomp.hpp"
//#include "opencv2/flann/matrix.h" 

#include <Windows.h>
#include <fstream>
#include "Eigen/Dense"
#include <opencv2/core/eigen.hpp>
#include <boost/algorithm/string.hpp>

#include<algorithm>

using namespace std;
using namespace cv;
using namespace ppf_match_3d;
using namespace kdtree;



int evalUwa(string & Method) {
    string method = Method;
    cout << "*****************************  " << method << "  *****************************" << endl;

    string dataName = "UWA";
    cout << "eval " << dataName << endl;

    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/"+ dataName + "/";
    string configPath = "3D models/Mian/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/testResults/" + method + "Results/" + dataName + "/";
    string modelPath = rootPath + configPath;
    string evalFileRootPath = "D:/wenhao.sun/Documents/GitHub/1-project/eval/" + method + "Eval/" + dataName + "/";

    vector<string> uwaModelName{ "parasaurolophus_high", "cheff", "chicken_high", "T-rex_high" };
    //cout << uwaModelName[0] << endl;
    map< string, vector<double>> ADDMap;
    map< string, vector<double>> ADIMap;
    map< string, vector<double>> centerErrorMap;
    map< string, vector<double>> phiMap;
    map< string, vector<double>> dNormMap;
    map< string, vector<double>> timeMap;
    map< string, vector<double>> occulsionMap;
    map< string, vector<string>> checkTableMap;

    for (int i = 0; i < uwaModelName.size(); i++) {
        ADDMap[uwaModelName[i]] = vector<double>();
        ADIMap[uwaModelName[i]] = vector<double>();
        centerErrorMap[uwaModelName[i]] = vector<double>();
        phiMap[uwaModelName[i]] = vector<double>();
        dNormMap[uwaModelName[i]] = vector<double>();
        timeMap[uwaModelName[i]] = vector<double>();
        occulsionMap[uwaModelName[i]] = vector<double>();
        checkTableMap[uwaModelName[i]] = vector<string>();
    }
    string ADDName              = evalFileRootPath + (string)"/eval_ADD.txt";
    string ADIName              = evalFileRootPath + (string)"/eval_ADI.txt";
    string centerErrorName      = evalFileRootPath + (string)"/eval_centerError.txt";
    string phiName              = evalFileRootPath + (string)"/eval_phi.txt";
    string dNormName            = evalFileRootPath + (string)"/eval_dNorm.txt";
    string timeName             = evalFileRootPath + (string)"/eval_time.txt";
    string occulsionName        = evalFileRootPath + (string)"/eval_occulsion.txt";
    string DiamtersName         = evalFileRootPath + (string)"/eval_Diamters.txt";
    string checkTableName       = evalFileRootPath + (string)"/checkTable.txt";

    // 计算直径, center point
    map<string, double> Diamters;
    map<string, Vec3f> Centers;
    map<string, Mat> modelPC;
    string modelFilePath;
    for (int i = 0; i < uwaModelName.size(); i++) {
        modelFilePath = modelPath + uwaModelName[i] + "_0.ply";
        Mat pc = loadPLYSimple(modelFilePath.c_str(), 1);
        modelPC[uwaModelName[i]] = pc;
        Vec2f xRange, yRange, zRange;
        computeBboxStd(pc, xRange, yRange, zRange);
        float dx = xRange[1] - xRange[0];
        float dy = yRange[1] - yRange[0];
        float dz = zRange[1] - zRange[0];
        float diameter = sqrt(dx * dx + dy * dy + dz * dz);
        Diamters[uwaModelName[i]] = diameter;

        Vec3f center;
        center[0] = (xRange[1] + xRange[0]) / 2;
        center[1] = (yRange[1] + yRange[0]) / 2;
        center[2] = (zRange[1] + zRange[0]) / 2;
        Centers[uwaModelName[i]] = center;
    }


    // 所有场景
    string cfgsFile = rootPath + configPath + "configFilesList.txt";
    vector<string> cfgNameAll;
    ifstream cfg_ifs(cfgsFile);
    string cfgName0;
    while (getline(cfg_ifs, cfgName0)) {
        cfgNameAll.push_back(cfgName0);
    }
    cfg_ifs.close();

    for (int icfg = 0; icfg < cfgNameAll.size(); icfg++) {
        //for (int icfg = 0; icfg < 1; icfg++) {
        auto& cfgName = cfgNameAll[icfg];
        // 单个场景
        string cfgPath0 = rootPath + configPath + cfgName;//cfgName
        LPCTSTR cfgPath = cfgPath0.c_str();

        // 多个模型

        LPSTR  modelNumCh = new char[1024];
        GetPrivateProfileString("MODELS", "NUMBER", "NULL", modelNumCh, 512, cfgPath);
        int modelNum = atoi(modelNumCh);
        string modelKey, modelGTKey, modelOccluKey;
        for (int i = 0; i < modelNum; i++) {
            modelKey = "MODEL_" + to_string(i);
            modelGTKey = modelKey + "_GROUNDTRUTH";
            modelOccluKey = modelKey + "_OCCLUSION";

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


            //string mPath = mPath1.substr(0, mPath1.length() - 4) + "_0.ply";
            //Mat pc = loadPLYSimple(mPath.c_str(), 1);
            //Vec3f p1(pc.ptr<float>(0));


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

            // build check table
            // reaf scene path from cfg
            LPSTR  scenePath0 = new char[1024];
            GetPrivateProfileString("SCENE", "PATH", "NULL", scenePath0, 512, cfgPath);
            string sceneName;  //用来索引已经读取的模型点云
            {
                std::vector<std::string> mStr, mn;
                boost::split(mStr, scenePath0, boost::is_any_of("/"));
                boost::split(mn, *(mStr.end() - 1), boost::is_any_of("."));
                sceneName = mn[0];
            }
            string str = gtName + "_" + sceneName; ////////////////////////
            checkTableMap[modelNameInPred].push_back(str);

            // read occulsion
            LPSTR  occlu = new char[1024];
            GetPrivateProfileString("MODELS", modelOccluKey.c_str(), "NULL", occlu, 512, cfgPath);
            string occlus = occlu;
            double occlud = stod(occlus);
            occulsionMap[modelNameInPred].push_back(occlud);

            // ADD
            Mat& pc = modelPC[modelNameInPred];
            Mat pct_gt = transformPCPose(pc, gt_pose); //pc是原始模型
            Mat pct_pred = transformPCPose(pc, pred_pose); //pc是原始模型

            double totalD_ADD = 0;
            for (int ii = 0; ii < pct_gt.rows; ii++)
            {
                Vec3f v1(pct_gt.ptr<float>(ii));
                //const Vec3f n1(pct_gt.ptr<float>(ii) + 3);
                Vec3f v2(pct_pred.ptr<float>(ii));
                v1 = v1 - v2;
                totalD_ADD += cv::norm(v1);
            }
            totalD_ADD /= pct_gt.rows;
            ADDMap[modelNameInCfg].push_back(totalD_ADD);

            // ADI
            Mat features;
            Mat queries;
            pct_gt.colRange(0, 3).copyTo(features);
            pct_pred.colRange(0, 3).copyTo(queries);

            //cout << pc.at<float>(0, 0) << pc.at<float>(0, 1) << pc.at<float>(0, 2) << endl;

            KDTree* model_tree = BuildKDTree(features);
            std::vector<std::vector<int>> indices;
            std::vector<std::vector<float>> dists;
            SearchKDTree(model_tree, queries, indices, dists, 1);
            delete model_tree;
            double totalD_ADI = 0;
            for (int ii = 0; ii < queries.rows; ii++)
            {
                totalD_ADI += sqrt(dists[ii][0]);
            }
            totalD_ADI /= queries.rows;
            ADIMap[modelNameInCfg].push_back(totalD_ADI);

            //centerError
            Vec3f centerV = Centers[modelNameInCfg];
            Mat center = Mat(1, centerV.rows, CV_32F);
            float* pcData = center.ptr<float>(0);
            pcData[0] = (float)centerV[0];
            pcData[1] = (float)centerV[1];
            pcData[2] = (float)centerV[2];

            //cout << center.row(0) << endl;
            Mat ctt_gt = transformPCPose(center, gt_pose); //pc是原始模型
            Mat ctt_pred = transformPCPose(center, pred_pose); //pc是原始模型
            Vec3f cd = ctt_gt.at<Vec3f>(0,0) - ctt_pred.at<Vec3f>(0,0);
            //cout << ctt_gt.at<Vec3f>(0,0) << endl;
            //cout << ctt_pred.at<Vec3f>(0,0) << endl;
            //cout << cd << endl;
            centerErrorMap[modelNameInCfg].push_back(cv::norm(cd));

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
    writeMap(ADIMap, ADIName);
    writeMap(centerErrorMap, centerErrorName);
    writeMap(phiMap, phiName);
    writeMap(dNormMap, dNormName);
    writeMap(timeMap, timeName);
    writeMap(occulsionMap, occulsionName);
    writeMap(Diamters, DiamtersName);
    
    writeMap(checkTableMap, checkTableName);

    int instancesNum(0);
    for (auto it = checkTableMap.begin(); it != checkTableMap.end(); ++it) {
        instancesNum += it->second.size();
        cout << it->first << ": " << it->second.size() << endl;
    }
    cout << "Total: " << instancesNum << endl;  //188


    return 0;
}

void evalKinect() {
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/Kinect/";
    string configPath = "/3D models/CVLab/Kinect/ObjectRecognition/Scenes/2011_06_27/configwithbestview/";
    string rePath = "D:/wenhao.sun/Documents/GitHub/1-project/Halcon_benchmark/halconResults/kinect/";


}

int rateUwa(string& Method) {

    string method = Method;
    cout << "*****************************  " << method << "  *****************************" << endl;
    string dataName = "UWA";
    cout << "rate " << dataName << endl;

    string evalFileRootPath = "D:/wenhao.sun/Documents/GitHub/1-project/eval/" + method + "Eval/"+ dataName + "/";

    // 输出failure case的文件名
    string failureCaseADICenterFilePath = evalFileRootPath + method + "FailureCasesADICenter.txt";
    string failureCaseADICenterOcclusionFilePath = evalFileRootPath + method + "FailureCasesADICenterOcclusion.txt";

    vector<string> uwaModelName{ "parasaurolophus_high", "cheff", "chicken_high", "T-rex_high" };
    //cout << uwaModelName[0] << endl;
    map< string, vector<double>> ADDMap;
    map< string, vector<double>> ADIMap;
    map< string, vector<double>> centerErrorMap;
    map< string, vector<double>> phiMap;
    map< string, vector<double>> dNormMap;
    map< string, vector<double>> timeMap;
    map< string, vector<double>> occulsionMap;
    map< string, vector<string>> checkTableMap;
    map< string, vector<string>> failureCaseADICenterTableMap;
    map< string, vector<double>> failureCaseADICenterOcclusionTableMap;
    map<string, int> tpADD;
    map<string, int> tpADICenter;
    map<string, int> tpManifold;
    map<string, int> tpManifoldLowOccul;
    map<string, int> lowOcculNum;
    map<string, double> rateADD;
    map<string, double> rateADICenter;
    map<string, double> rateManifold;
    map<string, double> Diamters;
    map<string, int> instanceNumMap;
    map<string, double> modelTimeMap;

    for (int i = 0; i < uwaModelName.size(); i++) {
        //ADDMap[uwaModelName[i]] = vector<double>();
        //phiMap[uwaModelName[i]] = vector<double>();
        //dNormMap[uwaModelName[i]] = vector<double>();
        //timeMap[uwaModelName[i]] = vector<double>();
        failureCaseADICenterTableMap[uwaModelName[i]] = vector<string>();
        failureCaseADICenterOcclusionTableMap[uwaModelName[i]] = vector<double>();

        tpADD[uwaModelName[i]] = 0;
        tpADICenter[uwaModelName[i]] = 0;
        tpManifold[uwaModelName[i]] = 0;
        tpManifoldLowOccul[uwaModelName[i]] = 0;
        lowOcculNum[uwaModelName[i]] = 0;
        rateADD[uwaModelName[i]] = 0;
        rateADICenter[uwaModelName[i]] = 0;
        rateManifold[uwaModelName[i]] = 0;
        modelTimeMap[uwaModelName[i]] = 0;
    }
    string ADDName          = evalFileRootPath + (string)"/eval_ADD.txt";
    string ADIName          = evalFileRootPath + (string)"/eval_ADI.txt";
    string centerErrorName  = evalFileRootPath + (string)"/eval_centerError.txt";
    string phiName          = evalFileRootPath + (string)"/eval_phi.txt";
    string dNormName        = evalFileRootPath + (string)"/eval_dNorm.txt";
    string timeName         = evalFileRootPath + (string)"/eval_time.txt";
    string DiamtersName     = evalFileRootPath + (string)"/eval_Diamters.txt";
    string occulsionName    = evalFileRootPath + (string)"/eval_occulsion.txt";
    string checkTableName   = evalFileRootPath + (string)"/checkTable.txt";

    readMap(ADDMap, ADDName);
    readMap(ADIMap, ADIName);
    readMap(phiMap, phiName);
    readMap(dNormMap, dNormName);
    readMap(timeMap, timeName);
    readMap(occulsionMap, occulsionName);
    readMap(checkTableMap, checkTableName);

    readMap(centerErrorMap, centerErrorName);
    readMap(Diamters, DiamtersName);


    //// threshold 
    //vector<double> ADDthres(3);
    //ADDthres[0] = 0.1;
    //ADDthres[1] = 0.2;
    //ADDthres[2] = 0.3;
    double AD_scale = 0.1;
    double manifold_pos_scale = 0.1;


    int allInstNum = 0;
    int allTpMfldNum = 0;
    int allLowOcculNum = 0;
    int allTpMfldLowOcculNum = 0;
    double occluThre = 0.84;

    for (int modelIndex = 0; modelIndex < uwaModelName.size(); modelIndex++) {

        string ModelName = uwaModelName[modelIndex];
        int instanceNumADD = ADDMap[ModelName].size();
        int instanceNumADI = ADIMap[ModelName].size();
        int instanceNumPhi = phiMap[ModelName].size();

        if (instanceNumADD != instanceNumADI || instanceNumADD != instanceNumPhi || instanceNumADI != instanceNumPhi)
        {
            cout << "Wrong instanceNum" << endl;
            return -1;
        }

        int instanceNum = instanceNumADD;
        instanceNumMap[ModelName] = instanceNum;

        // time
        for (int insIndex = 0; insIndex < instanceNum; insIndex++) {
            double time = timeMap[ModelName][insIndex];
            modelTimeMap[ModelName] += time;
        }

        //rate ADD
        double ADDthres = AD_scale * Diamters[ModelName]; ///thres
        for (int insIndex = 0; insIndex < instanceNum; insIndex++) {
            double ADDError = ADDMap[ModelName][insIndex];
            if (ADDError < ADDthres) { ///////////////////////////
                tpADD[ModelName] += 1;
            }
        }
        rateADD[ModelName] = (double)tpADD[ModelName] / (double)instanceNum;
        //cout << " rateADD: " <<  tpADD[ModelName] << " / " << instanceNum << " = " << rateADD[ModelName] << "          " << ModelName << endl;

        //rate ADI
        double ADICenterthres = AD_scale * Diamters[ModelName]; ///thres

        for (int insIndex = 0; insIndex < instanceNum; insIndex++) {
            double ADIError = ADIMap[ModelName][insIndex];
            double centerError = centerErrorMap[ModelName][insIndex];
            if (max(ADIError, centerError) < ADICenterthres) { ///////////////////////////
                tpADICenter[ModelName] += 1;
            }
            else {
                string failureCase = checkTableMap[ModelName][insIndex];
                failureCaseADICenterTableMap[ModelName].push_back(failureCase);
                double failureCaseOcclusion = occulsionMap[ModelName][insIndex];
                failureCaseADICenterOcclusionTableMap[ModelName].push_back(failureCaseOcclusion);
            }
        }
        rateADICenter[ModelName] = (double)tpADICenter[ModelName] / (double)instanceNum;
        //cout  << " rateADICenter: " <<  tpADICenter[ModelName] << " / " << instanceNum << " = " << rateADICenter[ModelName] << "          " << ModelName << endl;

        //rate manifold
        double position_thres = manifold_pos_scale * Diamters[ModelName];
        double radius_thres = 2 * M_PI / 30;
        int lowOccul = 0;
        for (int insIndex = 0; insIndex < instanceNum; insIndex++) {
            double phiError = phiMap[ModelName][insIndex];
            double dNormError = dNormMap[ModelName][insIndex];
            double occlu = occulsionMap[ModelName][insIndex];
            if (phiError < radius_thres && dNormError < position_thres) { ///////////////////////////
                tpManifold[ModelName] += 1;
            }
            if (occlu <= occluThre) {
                lowOcculNum[ModelName] += 1;
                if (phiError < radius_thres && dNormError < position_thres) { ///////////////////////////
                    tpManifoldLowOccul[ModelName] += 1;
                }
            }

        }
        rateManifold[ModelName] = (double)tpManifold[ModelName] / (double)instanceNum;
        //cout << " rateManifold: " << tpManifold[ModelName] << " / " << instanceNum << " = " << rateManifold[ModelName] << "          " << ModelName << endl;

        allInstNum += instanceNum;
        allTpMfldNum += tpManifold[ModelName];

        allLowOcculNum += lowOcculNum[ModelName];
        allTpMfldLowOcculNum += tpManifoldLowOccul[ModelName];

    }

    // recognition rate of all objects; 
    // Manifold
    cout << "manifld" << endl;
    for (int modelIndex = 0; modelIndex < uwaModelName.size(); modelIndex++) {
        string ModelName = uwaModelName[modelIndex];
        cout << " rateManifold: " << tpManifold[ModelName] << " / " << instanceNumMap[ModelName] << " = " << rateManifold[ModelName] << "          " << ModelName << endl;
    }
    cout << "mfld recognition rate: " << allTpMfldNum << " / " << allInstNum << " = " << (double)allTpMfldNum / (double)allInstNum << endl;
    cout << "recognition rate of all objs with less than 84% occlusion: " << allTpMfldLowOcculNum << " / " << allLowOcculNum << " = " << (double)allTpMfldLowOcculNum / (double)allLowOcculNum << endl;
    cout << endl;

    // ADD
    cout << "ADD" << endl;
    double averageRecallADD = 0;
    double recallSumADD = 0;
    for (int modelIndex = 0; modelIndex < uwaModelName.size(); modelIndex++) {
        string ModelName = uwaModelName[modelIndex];
        cout << " rateADD: " <<  tpADD[ModelName] << " / " << instanceNumMap[ModelName] << " = " << rateADD[ModelName] << "          " << ModelName << endl;
        recallSumADD += rateADD[ModelName];
    }
    averageRecallADD = recallSumADD / uwaModelName.size();
    cout << "ADD recognition rate: " << " sum {recall of single obj} / obj num = " << averageRecallADD << endl;
    cout << endl;

    // ADI & center
    cout << "ADI & center" << endl;
    double avergRecallADI = 0;
    double recallSumADI = 0;
    for (int modelIndex = 0; modelIndex < uwaModelName.size(); modelIndex++) {
        string ModelName = uwaModelName[modelIndex];
        cout  << " rateADICenter: " <<  tpADICenter[ModelName] << " / " << instanceNumMap[ModelName] << " = " << rateADICenter[ModelName] << "          " << ModelName << endl;
        recallSumADI += rateADICenter[ModelName];
    }
    avergRecallADI = recallSumADI / uwaModelName.size();
    cout << "ADI&Center recognition rate: " << " sum {recall of single obj} / obj num = " << avergRecallADI << endl;
    cout << endl;

    cout << "Time" << endl;
    double avgAllInferenceTime = 0;
    double sumTime = 0;
    for (int modelIndex = 0; modelIndex < uwaModelName.size(); modelIndex++) {
        string ModelName = uwaModelName[modelIndex];
        double avgTime = modelTimeMap[ModelName] / instanceNumMap[ModelName];
        cout << "rateTime: " << modelTimeMap[ModelName] << " / " << instanceNumMap[ModelName] << " = " << avgTime << "          " << ModelName << endl;
        sumTime += modelTimeMap[ModelName];
    }
    avgAllInferenceTime = sumTime / allInstNum;
    cout << "Average time of all instances of all objects: " << "sum{inference time of one obj in a scene} / inference counts = " << avgAllInferenceTime << endl;
    cout << endl;

    writeMap(failureCaseADICenterTableMap, failureCaseADICenterFilePath); 
    writeMap(failureCaseADICenterOcclusionTableMap, failureCaseADICenterOcclusionFilePath);

    return 0;
}

/*目标是比较 predicted pose 和 gt pose*/
int evalRateIcbin(string& Method) {
    string method = Method;
    cout << "*****************************  " << method << "  *****************************" << endl;

    string dataName = "icbin";
    double ADI_scale = 0.1;
    cout << "eval & rate " << dataName << " scenario2" << ", ADI_scale = " << ADI_scale << endl;

    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/IC-BIN-cvpr16/cvpr16_scenario_2/";
    //string configPath = "3D models/Mian/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/testResults/" + method + "Results/" + dataName + "/";
    string modelPath = rootPath + "meshes/";
    string evalFileRootPath = "D:/wenhao.sun/Documents/GitHub/1-project/eval/" + method + "Eval/" + dataName + "/";

    vector<string> scenariolist = { "coffee_cup", "juice", "mixed"};
    vector<string> uwaModelName{ "coffee_cup", "juice"}; //['coffee_cup', 'juice', 'mixed']
    //cout << uwaModelName[0] << endl;
    map< string, int> modelTPMap;
    map< string, int> modelTotalNumMap;
    //map< string, vector<double>> ADDMap;
    //map< string, vector<double>> ADIMap;
    //map< string, vector<double>> centerErrorMap;
    //map< string, vector<double>> phiMap;
    //map< string, vector<double>> dNormMap;
    vector<double> times;
    //map< string, vector<string>> checkTableMap;
    map<string, double> rateADI;

    for (int i = 0; i < uwaModelName.size(); i++) {
        modelTPMap[uwaModelName[i]] = 0;
        modelTotalNumMap[uwaModelName[i]] = 0;
        //ADDMap[uwaModelName[i]] = vector<double>();
        //ADIMap[uwaModelName[i]] = vector<double>();
        //centerErrorMap[uwaModelName[i]] = vector<double>();
        //phiMap[uwaModelName[i]] = vector<double>();
        //dNormMap[uwaModelName[i]] = vector<double>();
        //checkTableMap[uwaModelName[i]] = vector<string>();
        rateADI[uwaModelName[i]] = 0;
    }
    //string ADDName              = evalFileRootPath + (string)"/eval_ADD.txt";
    //string ADIName              = evalFileRootPath + (string)"/eval_ADI.txt";
    //string centerErrorName      = evalFileRootPath + (string)"/eval_centerError.txt";
    //string phiName              = evalFileRootPath + (string)"/eval_phi.txt";
    //string dNormName            = evalFileRootPath + (string)"/eval_dNorm.txt";
    //string timeName             = evalFileRootPath + (string)"/eval_time.txt";
    //string occulsionName        = evalFileRootPath + (string)"/eval_occulsion.txt";
    //string DiamtersName         = evalFileRootPath + (string)"/eval_Diamters.txt";
    //string checkTableName       = evalFileRootPath + (string)"/checkTable.txt";

    // 计算直径, center point
    map<string, double> Diamters;
    map<string, Vec3f> Centers;
    map<string, Mat> modelPC;
    string modelFilePath;
    for (int i = 0; i < uwaModelName.size(); i++) {
        modelFilePath = modelPath + uwaModelName[i] + "_eval.ply";
        Mat pc = loadPLYSimple(modelFilePath.c_str(), 1);
        modelPC[uwaModelName[i]] = pc;
        Vec2f xRange, yRange, zRange;
        computeBboxStd(pc, xRange, yRange, zRange);
        float dx = xRange[1] - xRange[0];
        float dy = yRange[1] - yRange[0];
        float dz = zRange[1] - zRange[0];
        float diameter = sqrt(dx * dx + dy * dy + dz * dz);
        Diamters[uwaModelName[i]] = diameter;

        Vec3f center;
        center[0] = (xRange[1] + xRange[0]) / 2;
        center[1] = (yRange[1] + yRange[0]) / 2;
        center[2] = (zRange[1] + zRange[0]) / 2;
        Centers[uwaModelName[i]] = center;
    }


    // for all senarios

    for (auto scenarioName : scenariolist) {
        string cfgsFile = rootPath + scenarioName + "_configs.txt";
        vector<string> cfgNameAll;
        ifstream cfg_ifs(cfgsFile);
        string cfgName0;
        while (getline(cfg_ifs, cfgName0)) {
            cfgNameAll.push_back(cfgName0);
        }
        cfg_ifs.close();

        // for all single scene
        for (int icfg = 0; icfg < cfgNameAll.size(); icfg++) {
            //for (int icfg = 0; icfg < 1; icfg++) {
            
            // the test unit is a singe scene, in which there're, let's say,
            // 2 different types objects (1 with 9 instances, and the other 1 with 4 instances).
            string& singleScene = cfgNameAll[icfg];
            std::vector<std::string> mStr, mn;
            boost::split(mStr, singleScene, boost::is_any_of(" "));
            string sid = mStr[0]; // scene id
            map<string, int> modelCountMap; // {"coffe": 9, "juice": 4}
            for (int is = 1; is < mStr.size()-1; is=is+2) {
                modelCountMap[mStr[is]] = atoi(mStr[is+1].c_str());
                bool in = false;
                for (auto modelName : uwaModelName) {
                    if (modelName == mStr[is]) in = true;
                }
                if (!in) { cout << "error here, check it " << endl; exit(1); }
            }

            // for different models
            // iterate over 2 different types objects
            for (auto modelName : uwaModelName) { // model name in config
                int modelCount = modelCountMap[modelName]; // model Count 9
                if (modelCount == 0) continue;
                //1. read predicted 9 poses
                string predFilePath = predPath + scenarioName + "-" + sid + "-" + modelName + ".txt"; //mixed-36-coffee_cup
                ifstream pred_ifs(predFilePath);
                if (!pred_ifs.is_open()) exit(1);

                // file head check
                string modelNameModelCountInPred;
                getline(pred_ifs, modelNameModelCountInPred); //coffee_cup=9
                int eqn1 = modelNameModelCountInPred.find("=");
                string modelNameInPred = modelNameModelCountInPred.substr(0, eqn1);
                int modelCountInPred = atoi(modelNameModelCountInPred.substr(eqn1 + 1, modelNameModelCountInPred.length() - eqn1).c_str());
                if (modelNameInPred != modelName) { cout << "not same: " << "scenario " << scenarioName << "scene " << singleScene << "pred " << predFilePath << endl; exit(1); }
                if (modelCountInPred != modelCount) { cout << "not same: " << "scenario " << scenarioName << "scene " << singleScene << "pred " << predFilePath << endl; exit(1); }
                // time
                string timeStr;
                getline(pred_ifs, timeStr);
                int eqn2 = timeStr.find("=");
                string timeStr2 = timeStr.substr(eqn2 + 1, timeStr.length() - eqn2);
                //cout << timeStr2 << endl;
                double time = stod(timeStr2, 0);
                times.push_back(time);
                // poses
                vector<Matx44d> pred_poses(modelCount);
                vector<float> scores(modelCount);
                string idx_score;
                for (int ip = 0; ip < modelCount; ip++) {
                    pred_ifs >> idx_score; int n = idx_score.find("=");
                    scores[ip] = stof(idx_score.substr(n + 1, idx_score.length() - n));
                    for (int ii = 0; ii < 4; ii++)
                        for (int jj = 0; jj < 4; jj++)
                        {
                            pred_ifs >> pred_poses[ip](ii, jj);
                        }
                }
                pred_ifs.close();
                // check scores, decrease order: 5, 4,3,2,1
                for (int i = 0; i < scores.size() - 1; i++) {
                    if (scores[i] < scores[i + 1]) {
                        cout << "error in score order" << endl; exit(1);
                    }
                }

                //2.  read 9 gt poses 
                string gtPath; //coffee_cup1_0, modelname index sceneid, index from 1
                vector<Matx44d> gt_poses(modelCount);
                for (int igt = 0; igt < modelCount; igt++) {
                    gtPath = rootPath + scenarioName + "/" + modelName+ to_string(igt + 1) + "_" + sid + ".txt";
                    ifstream gt_ifs(gtPath);
                    if (!gt_ifs.is_open()) { cout << "not open: " << gtPath << endl; exit(1); }
                    for (int ii = 0; ii < 4; ii++)
                        for (int jj = 0; jj < 4; jj++)
                        {
                            gt_ifs >> gt_poses[igt](ii, jj);
                        }
                    gt_ifs.close();
                }


                //3. assign tp https://zhuanlan.zhihu.com/p/37910324
                // center tranformed by pose_pred pose_gt
                vector<Mat> cp(modelCount), cg(modelCount);
                Vec3f centerV = Centers[modelName];
                Mat center = Mat(1, centerV.rows, CV_32F);
                float* pcData = center.ptr<float>(0);
                pcData[0] = (float)centerV[0];
                pcData[1] = (float)centerV[1];
                pcData[2] = (float)centerV[2];
                for (int ic = 0; ic < modelCount; ic++) {
                    cg[ic] = transformPCPose(center, gt_poses[ic]); 
                    cp[ic] = transformPCPose(center, pred_poses[ic]); 
                }
                // assign tp for ever pred pose
                vector<int> tp(modelCount, 0);
                vector<int> fp(modelCount, 0);
                vector<bool> gt_detected(modelCount, false);

                double thres = ADI_scale * Diamters[modelName]; ///thres
                Mat& pc = modelPC[modelName];

                for (int ip = 0; ip < modelCount; ip++) {
                    Mat pct_pred = transformPCPose(pc, pred_poses[ip]); //pc是原始模型
                    Mat features;
                    pct_pred.colRange(0, 3).copyTo(features);
                    KDTree* model_tree = BuildKDTree(features);

                    // compare with all gt pose
                    vector<double> ADIpi(modelCount, 1e5);
                    for (int igt = 0; igt < modelCount; igt++) {

                        // prefilter with center distance
                        Vec3f cd = cg[igt].at<Vec3f>(0,0) - cp[ip].at<Vec3f>(0,0); //https://www.cnblogs.com/hsy1941/p/8298314.html
                        if (cv::norm(cd) < 0.5 * Diamters[modelName]) {
                            // ADI
                            Mat pct_gt = transformPCPose(pc, gt_poses[igt]); //pc是原始模型
                            
                            Mat queries;
                            pct_gt.colRange(0, 3).copyTo(queries);
                            //cout << pc.at<float>(0, 0) << pc.at<float>(0, 1) << pc.at<float>(0, 2) << endl;
                            std::vector<std::vector<int>> indices;
                            std::vector<std::vector<float>> dists;
                            SearchKDTree(model_tree, queries, indices, dists, 1);
                            double totalD_ADI = 0;
                            for (int ii = 0; ii < queries.rows; ii++)
                            {
                                totalD_ADI += sqrt(dists[ii][0]);
                            }
                            totalD_ADI /= queries.rows;
                            ADIpi[igt] = totalD_ADI;
                        }

                    }
                    delete model_tree;

                    double minValue = *min_element(ADIpi.begin(), ADIpi.end());
                    int minPosition = min_element(ADIpi.begin(), ADIpi.end()) - ADIpi.begin();

                    // take the minimum ADI
                    if (minValue <= thres) {
                        if (!gt_detected[minPosition]) { // not detected
                            tp[ip] = 1;
                            gt_detected[minPosition] = true; // mark as detected
                        }
                        else {
                            fp[ip] = 1;
                        }
                    }
                    else {
                        fp[ip] = 1;
                    }
                }

                // accumulate tp num, total num
                int tpsum = 0;
                for (auto x : tp)
                    tpsum += x;
                modelTPMap[modelName] += tpsum;
                modelTotalNumMap[modelName] += modelCount;
            }
        }
    }
    
    // ADI 
    cout << "ADI " << endl;
    double avergRecallADI = 0;
    double recallSumADI = 0;
    for (auto modelName : uwaModelName) {
        rateADI[modelName] = (double)modelTPMap[modelName] / (double)modelTotalNumMap[modelName];
        cout << " rateADI: " << modelTPMap[modelName] << " / " << modelTotalNumMap[modelName] << " = " << rateADI[modelName] << "          " << modelName << endl;
        recallSumADI += rateADI[modelName];
    }
    avergRecallADI = recallSumADI / uwaModelName.size();
    cout << "ADI recognition rate: " << " sum {recall of single obj} / obj num = " << avergRecallADI << endl;
    cout << endl;

    cout << "Time" << endl;
    double avgAllInferenceTime = 0;
    double sumTime = 0;
    for (auto x : times) {
        sumTime += x;
    }
    avgAllInferenceTime = sumTime / times.size();
    cout << "Average time of all instances of all objects: " << "sum{inference time of one obj in a scene} / "<< times.size() <<" = " << avgAllInferenceTime << endl;
    cout << endl;

    return 0;
}

int evalRateLmo(string& Method) {
    string method = Method;
    cout << "*****************************  " << method << "  *****************************" << endl;

    string dataName = "lmo";
    double ADDADI_scale = 0.1;
    vector<string> ADD_objs{ "Ape", "Can", "Cat", "Driller", "Duck", "Holepuncher" };
    vector<string> ADI_objs = { "Eggbox", "Glue" };
    cout << "eval & rate " << dataName << ", ADDADI_scale = " << ADDADI_scale << endl;
    cout << "ADD_objs: " << "Ape Can Cat Driller Duck Holepuncher" << endl;
    cout << "ADI_objs: " << "Eggbox Glue" << endl;

    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/lm_lmo_pvnet/OCCLUSION_LINEMOD/OCCLUSION_LINEMOD/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/testResults/" + method + "Results/" + dataName + "/";
    string modelPath = rootPath + "models/";
    string evalFileRootPath = "D:/wenhao.sun/Documents/GitHub/1-project/eval/" + method + "Eval/" + dataName + "/";

    // test
    //int sceneNum = 1214;
    string cfgsFile = rootPath + "configs.txt";
    // val
    //int sceneNum = 20;
    //string cfgsFile = rootPath + "val_configs.txt";

    vector<string> uwaModelName{ "Ape", "Can", "Cat", "Driller", "Duck", "Eggbox", "Glue", "Holepuncher" };
    //cout << uwaModelName[0] << endl;
    map< string, int> modelTPMap;
    map< string, int> modelTotalNumMap;
    //map< string, vector<double>> ADDMap;
    //map< string, vector<double>> ADIMap;
    //map< string, vector<double>> centerErrorMap;
    //map< string, vector<double>> phiMap;
    //map< string, vector<double>> dNormMap;
    vector<double> times;
    //map< string, vector<string>> checkTableMap;
    map<string, double> rateADDADI;

    for (int i = 0; i < uwaModelName.size(); i++) {
        modelTPMap[uwaModelName[i]] = 0;
        modelTotalNumMap[uwaModelName[i]] = 0;
        //ADDMap[uwaModelName[i]] = vector<double>();
        //ADIMap[uwaModelName[i]] = vector<double>();
        //centerErrorMap[uwaModelName[i]] = vector<double>();
        //phiMap[uwaModelName[i]] = vector<double>();
        //dNormMap[uwaModelName[i]] = vector<double>();
        //checkTableMap[uwaModelName[i]] = vector<string>();
        rateADDADI[uwaModelName[i]] = 0;
    }
    //string ADDName              = evalFileRootPath + (string)"/eval_ADD.txt";
    //string ADIName              = evalFileRootPath + (string)"/eval_ADI.txt";
    //string centerErrorName      = evalFileRootPath + (string)"/eval_centerError.txt";
    //string phiName              = evalFileRootPath + (string)"/eval_phi.txt";
    //string dNormName            = evalFileRootPath + (string)"/eval_dNorm.txt";
    //string timeName             = evalFileRootPath + (string)"/eval_time.txt";
    //string occulsionName        = evalFileRootPath + (string)"/eval_occulsion.txt";
    //string DiamtersName         = evalFileRootPath + (string)"/eval_Diamters.txt";
    //string checkTableName       = evalFileRootPath + (string)"/checkTable.txt";

    // 计算直径, center point
    map<string, double> Diamters;
    map<string, Vec3f> Centers;
    map<string, Mat> modelPC;
    string modelFilePath;
    for (int i = 0; i < uwaModelName.size(); i++) {
        modelFilePath = modelPath + uwaModelName[i] + "_eval.ply";
        Mat pc = loadPLYSimple(modelFilePath.c_str(), 1);
        modelPC[uwaModelName[i]] = pc;
        Vec2f xRange, yRange, zRange;
        computeBboxStd(pc, xRange, yRange, zRange);
        float dx = xRange[1] - xRange[0];
        float dy = yRange[1] - yRange[0];
        float dz = zRange[1] - zRange[0];
        float diameter = sqrt(dx * dx + dy * dy + dz * dz);
        Diamters[uwaModelName[i]] = diameter;

        Vec3f center;
        center[0] = (xRange[1] + xRange[0]) / 2;
        center[1] = (yRange[1] + yRange[0]) / 2;
        center[2] = (zRange[1] + zRange[0]) / 2;
        Centers[uwaModelName[i]] = center;
    }



    vector<string> cfgNameAll;
    ifstream cfg_ifs(cfgsFile);
    string cfgName0;
    while (getline(cfg_ifs, cfgName0)) {
        cfgNameAll.push_back(cfgName0);
    }
    cfg_ifs.close();

    // for single model
    for (int icfg = 0; icfg < cfgNameAll.size(); icfg++) {
        //for (int icfg = 6; icfg < 8; icfg++) {

        // the test unit is a singe model
        string& singleModel = cfgNameAll[icfg];
        std::vector<std::string> mStr, mn;
        boost::split(mStr, singleModel, boost::is_any_of(" "));
        string modelName = mStr[0]; // obj id
        vector<string> scenes(mStr.begin() + 1, mStr.end());

        double thres = ADDADI_scale * Diamters[modelName]; ///thres
        bool useADD = (find(ADD_objs.begin(), ADD_objs.end(), modelName) != ADD_objs.end());

        // for every scene
        //for (auto scene : scenes) {
        for (int iscene = 0; iscene < scenes.size(); iscene++) {
            string scene = scenes[iscene];
            // model i in scene j
            //1.  read pred pose
            string predFilePath = predPath + modelName + "-" + scene + ".txt"; //Ape-00010
            ifstream pred_ifs(predFilePath);
            if (!pred_ifs.is_open()) exit(1);

            // file head check
            string modelNameModelCountInPred;
            getline(pred_ifs, modelNameModelCountInPred); //Duck=1
            // time
            string timeStr;
            getline(pred_ifs, timeStr);
            int eqn2 = timeStr.find("=");
            string timeStr2 = timeStr.substr(eqn2 + 1, timeStr.length() - eqn2);
            double time = stod(timeStr2, 0);
            times.push_back(time);
            // score
            string idx_score;
            pred_ifs >> idx_score; 

            // poses
            Matx44d pred_pose;
            for (int ii = 0; ii < 4; ii++)
                for (int jj = 0; jj < 4; jj++)
                {
                    pred_ifs >> pred_pose(ii, jj);
                }
            pred_ifs.close();

            //2.  read gt pose 
            string gtPath = rootPath + "poses/" + modelName + "/info_" + scene + ".txt"; // info_00000
            ifstream gt_ifs(gtPath);
            if (!gt_ifs.is_open()) { cout << "not open: " << gtPath << endl; exit(1); }
            string filehead;
            getline(gt_ifs, filehead);
            getline(gt_ifs, filehead);
            getline(gt_ifs, filehead);
            getline(gt_ifs, filehead);
            if (filehead == "") { 
                cout << "error in load gt: " << gtPath << endl;
                exit(1);
            }
            Matx33d gt_rotation;
            Vec3d gt_translation;
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                {
                    gt_ifs >> gt_rotation(ii, jj);
                }
            gt_ifs >> filehead;
            for (int it = 0; it < 3; it++)
                gt_ifs >> gt_translation(it);
            gt_ifs.close();
            Matx44d gt_pose;
            rtToPose(gt_rotation, gt_translation, gt_pose);

            //3. compute ADD(I)

            Mat& pc = modelPC[modelName];
            Mat pct_gt = transformPCPose(pc, gt_pose); //pc是原始模型
            Mat pct_pred = transformPCPose(pc, pred_pose); //pc是原始模型

            if (useADD) {
                // ADD
                double totalD_ADD = 0;
                for (int ii = 0; ii < pct_gt.rows; ii++)
                {
                    Vec3f v1(pct_gt.ptr<float>(ii));
                    //const Vec3f n1(pct_gt.ptr<float>(ii) + 3);
                    Vec3f v2(pct_pred.ptr<float>(ii));
                    v1 = v1 - v2;
                    totalD_ADD += cv::norm(v1);
                }
                totalD_ADD /= pct_gt.rows;
                if (totalD_ADD < thres) {
                    modelTPMap[modelName] += 1;
                }
            }
            else {
                // ADI
                Mat features;
                Mat queries;
                pct_gt.colRange(0, 3).copyTo(features);
                pct_pred.colRange(0, 3).copyTo(queries);

                //cout << pc.at<float>(0, 0) << pc.at<float>(0, 1) << pc.at<float>(0, 2) << endl;

                KDTree* model_tree = BuildKDTree(features);
                std::vector<std::vector<int>> indices;
                std::vector<std::vector<float>> dists;
                SearchKDTree(model_tree, queries, indices, dists, 1);
                delete model_tree;
                double totalD_ADI = 0;
                for (int ii = 0; ii < queries.rows; ii++)
                {
                    totalD_ADI += sqrt(dists[ii][0]);
                }
                totalD_ADI /= queries.rows;
                if (totalD_ADI < thres) {
                    modelTPMap[modelName] += 1;
                }
            }
            
            modelTotalNumMap[modelName] += 1;


        }
    }

    // ADI 
    cout << "ADD & ADI " << endl;
    double avergRecall = 0;
    double recallSum = 0;
    for (auto modelName : uwaModelName) {
        rateADDADI[modelName] = (double)modelTPMap[modelName] / (double)modelTotalNumMap[modelName];
        cout << " rateADI: " << modelTPMap[modelName] << " / " << modelTotalNumMap[modelName] << " = " << rateADDADI[modelName] << "          " << modelName << endl;
        recallSum += rateADDADI[modelName];
    }
    avergRecall = recallSum / uwaModelName.size();
    cout << "ADI recognition rate: " << " sum {recall of single obj} / obj num = " << avergRecall << endl;
    cout << endl;

    cout << "Time" << endl;
    double avgAllInferenceTime = 0;
    double sumTime = 0;
    for (auto x : times) {
        sumTime += x;
    }
    avgAllInferenceTime = sumTime / times.size();
    cout << "Average time of all instances of all objects: " << "sum{inference time of one obj in a scene} / " << times.size() << " = " << avgAllInferenceTime << endl;
    cout << endl;

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
    string method = "halcon";
    //string method = "ppf_2025";

    // uwa
    //evalUwa(method);
    //rateUwa(method);

    // icbin
    //evalRateIcbin(method);

    // lmo
    evalRateLmo(method);


    //writeMap();
    //readMap();
}

int main2() {
    string path = "D:/wenhao.sun/Documents/GitHub/1-project/ppf_2025/test.txt";
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



