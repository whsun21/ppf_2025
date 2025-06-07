// input output
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>
#include <wrap/ply/plylib.cpp> //https://blog.csdn.net/CUSTESC/article/details/106160295


#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

#include <boost/algorithm/string.hpp>
#define NOMINMAX
#include <Windows.h>

#include <Discregrid/All>


using namespace std;
using namespace cv;
using namespace ppf_match_3d;

int testUwa() {
    string dataName = "UWA";
    string method = "ppf_2025";
    cout << "test " << method << " on dataset " << dataName << endl;

    //para
    string samplingMethod = "cube"; // cube, normal
    double orientationDiffThreshold = 2. / 180. * M_PI;
    double nmsThreshold = 0.5; //0.2 0.5
    bool refineEnabled = true;
    bool nmsEnabled = true;

    // model
    double relativeSamplingStep = 0.025;
    // match
    double keypointStep = 5;
    double relativeSceneDistance = 0.025;
    int N = 1;
    // Create an instance of ICP
    ICP icp(5, 0.005f, 0, 3); //100, 0.005f, 2.5f, 8
    //path
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/" + dataName + "/";
    string configPath = "3D models/Mian/";

    string modelPath = rootPath + configPath;
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/testResults/" + method + "Results/" + dataName + "/";

    // cfg file
    //auto& cfgName = cfgNameAll[icfg];
    //string cfgName = "ConfigScene1.ini";
    //string cfgPath0 = rootPath + configPath + cfgName;//cfgName
    //LPCTSTR cfgPath = cfgPath0.c_str();

    // 提前读取模型
    vector<string> uwaModelName{ "parasaurolophus_high", "cheff", "chicken_high", "T-rex_high" };
    map<string, Mat> modelPC;
    map<string, std::shared_ptr<Discregrid::DiscreteGrid>> modelSDF;
    string modelFilePath;
    string sdfFilePath;
    for (int i = 0; i < uwaModelName.size(); i++) {
        modelFilePath = modelPath + uwaModelName[i] + "_0.ply";
        sdfFilePath   = modelPath + uwaModelName[i] + "_0.cdf";

        MyMesh pcM;
        vcg::tri::io::ImporterPLY<MyMesh>::Open(pcM, modelFilePath.c_str());
        Mat pc;
        vcgMesh2cvMat(pcM, pc);

        //Mat pc = loadPLYSimple(modelFilePath.c_str(), 1);
        modelPC[uwaModelName[i]] = pc;

        // load sdf
        std::cout << "Load SDF...";
        auto sdf = std::shared_ptr<Discregrid::DiscreteGrid>{};
        sdf = std::shared_ptr<Discregrid::CubicLagrangeDiscreteGrid>(
            new Discregrid::CubicLagrangeDiscreteGrid(sdfFilePath));
        std::cout << "DONE" << std::endl;
        modelSDF[uwaModelName[i]] = sdf;
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
        //string cfgName = "ConfigScene2.ini";
        // 单个场景
        string cfgPath0 = rootPath + configPath + cfgName;//cfgName
        LPCTSTR cfgPath = cfgPath0.c_str();

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
        string sceneFileName = rootPath + configPath + sceneName + "_0.ply";

        //string sceneFileName = modelPath + "rs1_0.ply";
        // Read scene
        MyMesh pcS;
        vcg::tri::io::ImporterPLY<MyMesh>::Open(pcS, sceneFileName.c_str());
        if (pcS.face.size() == 0) {
            cout << "Scene has no face. Need faces for normal computing by PerVertexFromCurrentFaceNormal" << endl;
            exit(1);
        }
        tri::UpdateNormal<MyMesh>::PerFace(pcS);
        tri::UpdateNormal<MyMesh>::PerVertexFromCurrentFaceNormal(pcS);
        tri::UpdateNormal<MyMesh>::NormalizePerVertex(pcS);
        Mat pcTest;
        vcgMesh2cvMat(pcS, pcTest);

        // 多个模型在单个场景中
        // read cfg model num
        LPSTR  modelNumCh = new char[1024];
        GetPrivateProfileString("MODELS", "NUMBER", "NULL", modelNumCh, 512, cfgPath);
        int modelNum = atoi(modelNumCh);
        string modelKey, modelGTKey;

        for (int i = 0; i < modelNum; i++) {
            //i = 2;
            modelKey = "MODEL_" + to_string(i);
            modelGTKey = modelKey + "_GROUNDTRUTH";

            // 单个模型
            // read cfg mdoel path
            LPSTR  modelPath0 = new char[1024];
            GetPrivateProfileString("MODELS", modelKey.c_str(), "NULL", modelPath0, 512, cfgPath);
            string modelNameInCfg;  //用来索引已经读取的模型点云
            {
                std::vector<std::string> mStr, mn;
                boost::split(mStr, modelPath0, boost::is_any_of("/"));
                boost::split(mn, *(mStr.end() - 1), boost::is_any_of("."));
                modelNameInCfg = mn[0];
            }

            // read cfg gt 
            LPSTR  gtPath0 = new char[1024];
            GetPrivateProfileString("MODELS", modelGTKey.c_str(), "NULL", gtPath0, 512, cfgPath);
            string gtName; // 真实的位姿结果文件，用来名名预测的位姿结果文件
            {
                std::vector<std::string> mStr, mn;
                boost::split(mStr, gtPath0, boost::is_any_of("/"));
                boost::split(mn, *(mStr.end() - 1), boost::is_any_of("."));
                gtName = mn[0];
            }

            // 索引模型
            Mat& pc = modelPC[modelNameInCfg];
            std::shared_ptr<Discregrid::DiscreteGrid> sdf = modelSDF[modelNameInCfg];

            // train the model
            ppf_match_3d::PPF3DDetector detector(relativeSamplingStep);//0.025
            //detector.enableDebug(true);
            detector.setSamplingMethod(samplingMethod);
            detector.setOrientationDiffThreshold(orientationDiffThreshold);
            detector.setSDF(sdf);
            detector.trainModel(pc);


            //timer
            int64 t1 = cv::getTickCount();


            // Match the model to the scene and get the pose
            vector<Pose3DPtr> results;
            detector.match(pcTest, results, 1.0 / keypointStep, relativeSceneDistance); //1.0/40.0, 0.05；作者建议1.0/5.0，0.025

            //check results size from match call above
            size_t results_size = results.size();
            if (results_size == 0) {
                cout << modelNameInCfg << endl << sceneFileName << endl << "No matching poses found. Exiting." << endl;
                exit(0);
            }

            // 后处理            
            detector.setNMSThreshold(nmsThreshold);
            detector.postProcessing(results, icp, refineEnabled, nmsEnabled); /////////

            // timer 
            int64 t2 = cv::getTickCount();

            // Get only first N results - but adjust to results size if num of results are less than that specified by N
            if (results.size() < N) {
                cout << endl << "Reducing matching poses to be reported (as specified in code): "
                    << N << " to the number of matches found: " << results.size() << endl;
                N = results.size();
            }
            vector<Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

            // 保存位姿，目前支持一个
            Matx44d pose = resultsSub[0]->pose;
            //cout << pose << endl;

            string predName = predPath + gtName + ".txt";
            ofstream of(predName);
            of << modelNameInCfg << endl;
            of << "time=" << (t2 - t1) / cv::getTickFrequency() << endl;
            of << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3) << endl;
            of << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " " << pose(1, 3) << endl;
            of << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << endl;
            of << pose(3, 0) << " " << pose(3, 1) << " " << pose(3, 2) << " " << pose(3, 3) << endl;
            of.close();
        }
    }
}

int testKinect() {
    string dataName = "Kinect";
    string method = "ppf_2025";
    cout << "test " << method << " on dataset " << dataName << endl;

    //para
    string samplingMethod = "cube"; // cube, normal
    double orientationDiffThreshold = 5. / 180. * M_PI;
    double nmsThreshold = 0.5; //0.2 0.5
    bool refineEnabled = true;
    bool nmsEnabled = true;
    bool fsEnabled = false;

    // model
    double relativeSamplingStep = 0.03;
    // match
    double keypointStep = 5;
    double relativeSceneDistance = 0.05;
    int N = 1;
    // Create an instance of ICP
    ICP icp(5, 0.005f, 0, 3); //100, 0.005f, 2.5f, 8
    //path
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/" + dataName + "/";
    string configPath = "/3D models/CVLab/Kinect/ObjectRecognition/Scenes/2011_06_27/configwithbestview/";
    string configPathScene = "/3D models/CVLab/Kinect/ObjectRecognition/Scenes/2011_06_27/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/testResults/" + method + "Results/" + dataName + "/";
    string modelPath = rootPath + configPath;

    // cfg file
    //auto& cfgName = cfgNameAll[icfg];
    //string cfgName = "ConfigScene1.ini";
    //string cfgPath0 = rootPath + configPath + cfgName;//cfgName
    //LPCTSTR cfgPath = cfgPath0.c_str();

    // 提前读取模型
    //vector<string> uwaModelName{ "parasaurolophus_high", "cheff", "chicken_high", "T-rex_high" };
    //map<string, Mat> modelPC;
    //map<string, std::shared_ptr<Discregrid::DiscreteGrid>> modelSDF;
    //string modelFilePath;
    //string sdfFilePath;
    //for (int i = 0; i < uwaModelName.size(); i++) {
    //    modelFilePath = modelPath + uwaModelName[i] + "_0.ply";
    //    sdfFilePath = modelPath + uwaModelName[i] + "_0.cdf";

    //    MyMesh pcM;
    //    vcg::tri::io::ImporterPLY<MyMesh>::Open(pcM, modelFilePath.c_str());
    //    Mat pc;
    //    vcgMesh2cvMat(pcM, pc);

    //    //Mat pc = loadPLYSimple(modelFilePath.c_str(), 1);
    //    modelPC[uwaModelName[i]] = pc;

    //    // load sdf
    //    std::cout << "Load SDF...";
    //    auto sdf = std::shared_ptr<Discregrid::DiscreteGrid>{};
    //    sdf = std::shared_ptr<Discregrid::CubicLagrangeDiscreteGrid>(
    //        new Discregrid::CubicLagrangeDiscreteGrid(sdfFilePath));
    //    std::cout << "DONE" << std::endl;
    //    modelSDF[uwaModelName[i]] = sdf;
    //}


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
        string cfgPath0 = rootPath + configPath + cfgName;//cfgName
        LPCTSTR cfgPath = cfgPath0.c_str();

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
        string sceneFileName = rootPath + configPathScene + sceneName + "_0.ply";

        //string sceneFileName = modelPath + "rs1_0.ply";
        // Read scene
        MyMesh pcS;
        vcg::tri::io::ImporterPLY<MyMesh>::Open(pcS, sceneFileName.c_str());
        if (pcS.face.size() == 0) {
            cout << "Scene has no face. Need faces for normal computing by PerVertexFromCurrentFaceNormal" << endl;
            exit(1);
        }
        tri::UpdateNormal<MyMesh>::PerFace(pcS);
        tri::UpdateNormal<MyMesh>::PerVertexFromCurrentFaceNormal(pcS);
        tri::UpdateNormal<MyMesh>::NormalizePerVertex(pcS);
        Mat pcTest;
        vcgMesh2cvMat(pcS, pcTest);

        // 多个模型在单个场景中
        // read cfg model num
        LPSTR  modelNumCh = new char[1024];
        GetPrivateProfileString("MODELS", "NUMBER", "NULL", modelNumCh, 512, cfgPath);
        int modelNum = atoi(modelNumCh);
        string modelKey, modelGTKey;

        for (int i = 0; i < modelNum; i++) {
            //i = 2;
            modelKey = "MODEL_" + to_string(i);
            modelGTKey = modelKey + "_GROUNDTRUTH";

            // 单个模型
            // read cfg mdoel path
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

            // read cfg gt 
            LPSTR  gtPath0 = new char[1024];
            GetPrivateProfileString("MODELS", modelGTKey.c_str(), "NULL", gtPath0, 512, cfgPath);
            string gtName; // 真实的位姿结果文件，用来名名预测的位姿结果文件
            {
                std::vector<std::string> mStr, mn;
                boost::split(mStr, gtPath0, boost::is_any_of("/"));
                boost::split(mn, *(mStr.end() - 1), boost::is_any_of("."));
                gtName = mn[0];
            }

            // 索引模型
            //Mat& pc = modelPC[modelNameInCfg];
            //std::shared_ptr<Discregrid::DiscreteGrid> sdf = modelSDF[modelNameInCfg];

            // train the model
            ppf_match_3d::PPF3DDetector detector(relativeSamplingStep);//0.025
            detector.enableFreeSpaceConstraint(fsEnabled);
            //detector.enableDebug(true);
            detector.setSamplingMethod(samplingMethod);
            detector.setOrientationDiffThreshold(orientationDiffThreshold);
            //detector.setSDF(sdf);
            detector.trainModel(pc);


            //timer
            int64 t1 = cv::getTickCount();


            // Match the model to the scene and get the pose
            vector<Pose3DPtr> results;
            detector.match(pcTest, results, 1.0 / keypointStep, relativeSceneDistance); //1.0/40.0, 0.05；作者建议1.0/5.0，0.025

            //check results size from match call above
            size_t results_size = results.size();
            if (results_size == 0) {
                cout << modelNameInCfg << endl << sceneFileName << endl << "No matching poses found. Exiting." << endl;
                exit(0);
            }

            // 后处理            
            detector.setNMSThreshold(nmsThreshold);
            detector.postProcessing(results, icp, refineEnabled, nmsEnabled); /////////

            // timer 
            int64 t2 = cv::getTickCount();

            // Get only first N results - but adjust to results size if num of results are less than that specified by N
            if (results.size() < N) {
                cout << endl << "Reducing matching poses to be reported (as specified in code): "
                    << N << " to the number of matches found: " << results.size() << endl;
                N = results.size();
            }
            vector<Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

            // 保存位姿，目前支持一个
            Matx44d pose = resultsSub[0]->pose;
            //cout << pose << endl;

            string predName = predPath + gtName + ".txt";
            ofstream of(predName);
            of << modelNameInCfg << endl;
            of << "time=" << (t2 - t1) / cv::getTickFrequency() << endl;
            of << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3) << endl;
            of << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " " << pose(1, 3) << endl;
            of << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << endl;
            of << pose(3, 0) << " " << pose(3, 1) << " " << pose(3, 2) << " " << pose(3, 3) << endl;
            of.close();
        }
    }
}


int testLmo() {
    string dataName = "lmo";
    string method = "ppf_2025";
    cout << "test " << method << " on dataset " << dataName << endl;

    //para
    string samplingMethod = "cube"; // cube, normal
    double orientationDiffThreshold = 2. / 180. * M_PI;
    double nmsThreshold = 0.5; //0.2 0.5
    bool refineEnabled = true;
    bool nmsEnabled = true;

    // model
    double relativeSamplingStep = 0.025;
    // match
    double keypointStep = 5;
    double relativeSceneDistance = 0.025;
    int N = 1;
    // Create an instance of ICP
    ICP icp(5, 0.005f, 0, 1); //100, 0.005f, 2.5f, 8
    //path
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/lm_lmo_pvnet/OCCLUSION_LINEMOD/OCCLUSION_LINEMOD/";

    string modelPath = rootPath + "models/";
    string cloudPath = rootPath + "/cloud_masked/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/testResults/" + method + "Results/" + dataName + "/";

    // cfg file
    //auto& cfgName = cfgNameAll[icfg];
    //string cfgName = "ConfigScene1.ini";
    //string cfgPath0 = rootPath + configPath + cfgName;//cfgName
    //LPCTSTR cfgPath = cfgPath0.c_str();

    // 提前读取模型
    vector<string> uwaModelName{ "Ape", "Can", "Cat", "Driller", "Duck", "Eggbox", "Glue", "Holepuncher" };
    map<string, Mat> modelPC;
    map<string, std::shared_ptr<Discregrid::DiscreteGrid>> modelSDF;
    string modelFilePath;
    string sdfFilePath;
    for (int i = 0; i < uwaModelName.size(); i++) {
        modelFilePath = modelPath + uwaModelName[i] + "_wt.ply";
        sdfFilePath = modelPath + uwaModelName[i] + "_sdf.cdf";

        MyMesh pcM;
        vcg::tri::io::ImporterPLY<MyMesh>::Open(pcM, modelFilePath.c_str());
        Mat pc;
        vcgMesh2cvMat(pcM, pc);

        //Mat pc = loadPLYSimple(modelFilePath.c_str(), 1);
        modelPC[uwaModelName[i]] = pc;

        // load sdf
        std::cout << "Load SDF...";
        auto sdf = std::shared_ptr<Discregrid::DiscreteGrid>{};
        sdf = std::shared_ptr<Discregrid::CubicLagrangeDiscreteGrid>(
            new Discregrid::CubicLagrangeDiscreteGrid(sdfFilePath));
        std::cout << "DONE" << std::endl;
        modelSDF[uwaModelName[i]] = sdf;
    }


    
    string cfgsFile = rootPath + "configs.txt";
    vector<string> cfgNameAll;
    ifstream cfg_ifs(cfgsFile);
    string cfgName0;
    while (getline(cfg_ifs, cfgName0)) {
        cfgNameAll.push_back(cfgName0);
    }
    cfg_ifs.close();

    // for every single model
    int icfg = 0;
    for (; icfg < cfgNameAll.size(); icfg++) {
        //for (int icfg = 0; icfg < 1; icfg++) {
        
        // the test unit is a singe model
        // 1. train model
        string& singleModel = cfgNameAll[icfg];
        std::vector<std::string> mStr, mn;
        boost::split(mStr, singleModel, boost::is_any_of(" "));
        string modelName = mStr[0]; // obj id
        vector<string> scenes(mStr.begin() + 1, mStr.end());
        // 索引模型
        Mat& pc = modelPC[modelName];
        std::shared_ptr<Discregrid::DiscreteGrid> sdf = modelSDF[modelName];

        // train the model
        ppf_match_3d::PPF3DDetector detector(relativeSamplingStep);//0.025
        //detector.enableDebug(true);
        detector.setSamplingMethod(samplingMethod);
        detector.setOrientationDiffThreshold(orientationDiffThreshold);
        detector.setNMSThreshold(nmsThreshold);
        detector.setSDF(sdf);
        detector.trainModel(pc);

        // 2. for every scene, run match
        int iscene = 0;
        //if (modelName == "Ape") iscene = 936;
        //if (modelName == "Can") iscene = 190;
        //if (modelName == "Cat") iscene = 921;
        for (; iscene < scenes.size(); iscene++) {
            string sid = scenes[iscene];
            //string sid = "00968";
            string sceneFileName = cloudPath + modelName + "/cloud" + "_" + sid + ".ply";
            // Read scene
            Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);
            //timer
            int64 t1 = cv::getTickCount();


            // Match the model to the scene and get the pose
            vector<Pose3DPtr> results;
            detector.match(pcTest, results, 1.0 / keypointStep, relativeSceneDistance); //1.0/40.0, 0.05；作者建议1.0/5.0，0.025

            //check results size from match call above
            size_t results_size = results.size();
            if (results_size == 0) {
                cout << modelName<< endl << sceneFileName << endl << "No matching poses found. Exiting." << endl;
                exit(0);
            }

            // postProcessing           
            detector.postProcessing(results, icp, refineEnabled, nmsEnabled); /////////
            detector.deleteFlannPtr();

            // timer 
            int64 t2 = cv::getTickCount();

            // Get only first N results - but adjust to results size if num of results are less than that specified by N
            if (results.size() < N) {
                cout << endl << "Reducing matching poses to be reported (as specified in code): "
                    << N << " to the number of matches found: " << results.size() << endl;
                N = results.size();
            }
            vector<Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

            // 保存位姿，目前支持一个
            Matx44d pose = resultsSub[0]->pose;
            double score = resultsSub[0]->overlap;
            //cout << pose << endl;

            string predName = predPath + modelName + "-" + sid + ".txt";
            ofstream of(predName);

            of << modelName<< "=" << 1 << endl;
            of << "time=" << (t2 - t1) / cv::getTickFrequency() << endl;
            of << 1 << "=" << score << endl;
            of << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3) << endl;
            of << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " " << pose(1, 3) << endl;
            of << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << endl;
            of << pose(3, 0) << " " << pose(3, 1) << " " << pose(3, 2) << " " << pose(3, 3) << endl;
            of.close();

        }
    }
}
/**/
int testIcbin() {
    string dataName = "icbin";
    string method = "ppf_2025";
    cout << "test " << method << " on dataset " << dataName << endl;

    //para
    string samplingMethod = "cube"; // cube, normal
    double orientationDiffThreshold = 2. / 180. * M_PI;
    double nmsThreshold = 0.5; //0.2 0.5
    bool refineEnabled = true;
    bool nmsEnabled = true;
    double ppf_angle_constraint = 30. / 180. * M_PI; //偏离90度至少20度，点对中1号点的法向与点对连线的夹角，与90度之差的绝对值，大于给定阈值
    double ppf_distance_constraint = 10;  // 1好点和2好点，至少相聚10*sampled resolution

    // model
    double relativeSamplingStep = 0.025;
    // match
    double keypointStep = 5;
    double relativeSceneDistance = 0.05;
    // Create an instance of ICP
    ICP icp(5, 0.005f, 0, 3); //100, 0.005f, 2.5f, 8
    //path
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/IC-BIN-cvpr16/cvpr16_scenario_2/";
    //string configPath = "3D models/Mian/";

    string modelPath = rootPath + "meshes/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/testResults/" + method + "Results/" + dataName + "/";

    // cfg file
    //auto& cfgName = cfgNameAll[icfg];
    //string cfgName = "ConfigScene1.ini";
    //string cfgPath0 = rootPath + configPath + cfgName;//cfgName
    //LPCTSTR cfgPath = cfgPath0.c_str();

    // 提前读取模型
    vector<string> scenariolist = { "coffee_cup", "juice", "mixed" };
    vector<string> uwaModelName{ "coffee_cup", "juice" }; //['coffee_cup', 'juice', 'mixed']
    map<string, Mat> modelPC;
    map<string, std::shared_ptr<Discregrid::DiscreteGrid>> modelSDF;
    string modelFilePath;
    string sdfFilePath;
    for (int i = 0; i < uwaModelName.size(); i++) {
        modelFilePath = modelPath + uwaModelName[i] + "_wt.ply";
        sdfFilePath = modelPath + uwaModelName[i] + "_sdf.cdf";

        MyMesh pcM;
        vcg::tri::io::ImporterPLY<MyMesh>::Open(pcM, modelFilePath.c_str());
        Mat pc;
        vcgMesh2cvMat(pcM, pc);

        //Mat pc = loadPLYSimple(modelFilePath.c_str(), 1);
        modelPC[uwaModelName[i]] = pc;

        // load sdf
        std::cout << "Load SDF...";
        auto sdf = std::shared_ptr<Discregrid::DiscreteGrid>{};
        sdf = std::shared_ptr<Discregrid::CubicLagrangeDiscreteGrid>(
            new Discregrid::CubicLagrangeDiscreteGrid(sdfFilePath));
        std::cout << "DONE" << std::endl;
        modelSDF[uwaModelName[i]] = sdf;
    }


    // for all senarios
    // the special point of this dataset: multi-instances
    for (auto scenarioName : scenariolist) {
        string cfgsFile = rootPath + scenarioName + "_configs.txt";
        string cloud_path = rootPath + scenarioName + "/cloud/";
        vector<string> cfgNameAll;
        ifstream cfg_ifs(cfgsFile);
        string cfgName0;
        while (getline(cfg_ifs, cfgName0)) {
            cfgNameAll.push_back(cfgName0);
        }
        cfg_ifs.close();

        int icfg = 0;
        //if (scenarioName == "coffee_cup") icfg = 47;
        for (; icfg < cfgNameAll.size(); icfg++) {
            //for (int icfg = 0; icfg < 1; icfg++) {
            
            // 1. read config: scene_id obj1 count1  obj2 count2
            // the test unit is a singe scene, in which there're, let's say, 2 different types 
            // objects (1 with 9 instances, and the other 1 with 4 instances).
            string& singleScene = cfgNameAll[icfg];
            //string singleScene = "51 coffee_cup 15 juice 0 ";
            std::vector<std::string> mStr, mn;
            boost::split(mStr, singleScene, boost::is_any_of(" "));
            string sid = mStr[0]; // scene id
            map<string, int> modelCountMap; // {"coffe": 9, "juice": 4}
            for (int is = 1; is < mStr.size() - 1; is = is + 2) {
                modelCountMap[mStr[is]] = atoi(mStr[is + 1].c_str());
                bool in = false;
                for (auto modelName : uwaModelName) {
                    if (modelName == mStr[is]) in = true;
                }
                if (!in) { cout << "error here, check it " << endl; exit(1); }
            }
            // read scene
            string sceneFileName = cloud_path + "cloud" + sid + ".ply";
            Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);


            // 2. for different models, form test combination: scene_id obji counti
            // iterate over 2 different types objects
            for (auto modelName : uwaModelName) { // model name in config
                int modelCount = modelCountMap[modelName]; // model Count 9
                if (modelCount == 0) continue;
                int N = modelCount;

                // read model
                // 索引模型
                Mat& pc = modelPC[modelName];
                std::shared_ptr<Discregrid::DiscreteGrid> sdf = modelSDF[modelName];

                // train the model
                ppf_match_3d::PPF3DDetector detector(relativeSamplingStep);//0.025
                //detector.enableDebug(true);
                detector.setSamplingMethod(samplingMethod);
                detector.setOrientationDiffThreshold(orientationDiffThreshold);
                detector.setNMSThreshold(nmsThreshold);
                detector.setSDF(sdf);
                detector.setPPFAngleConstraint(ppf_angle_constraint); /// 一定在trian之前
                detector.setPPFDistanceConstraint(ppf_distance_constraint); /// 一定在trian之前

                detector.trainModel(pc);

                //timer
                int64 t1 = cv::getTickCount();


                // Match the model to the scene and get the pose
                vector<Pose3DPtr> results;
                detector.match(pcTest, results, 1.0 / keypointStep, relativeSceneDistance); //1.0/40.0, 0.05；作者建议1.0/5.0，0.025

                //check results size from match call above
                size_t results_size = results.size();
                if (results_size == 0) {
                    cout << modelName << endl << sceneFileName << endl << "No matching poses found. Exiting." << endl;
                    exit(0);
                }

                // 后处理            
                detector.postProcessing(results, icp, refineEnabled, nmsEnabled); /////////

                // timer 
                int64 t2 = cv::getTickCount();

                // Get only first N results - but adjust to results size if num of results are less than that specified by N
                if (results.size() < N) {
                    cout << endl << "Reducing matching poses to be reported (as specified in code): "
                        << N << " to the number of matches found: " << results.size() << endl;
                    N = results.size();
                }
                vector<Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

                // 保存位姿，多个
                //cout << pose << endl;

                string predName = predPath + scenarioName + '-' + sid + '-' + modelName +  ".txt";
                ofstream of(predName);

                of << modelName << "=" << N << endl;
                of << "time=" << (t2 - t1) / cv::getTickFrequency() << endl;
                for (int ir = 0; ir < N; ir++) {
                    double score = resultsSub[ir]->overlap;
                    Matx44d pose = resultsSub[ir]->pose;
                    of << ir+1 << "=" << score << endl;
                    of << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3) << endl;
                    of << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " " << pose(1, 3) << endl;
                    of << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << endl;
                    of << pose(3, 0) << " " << pose(3, 1) << " " << pose(3, 2) << " " << pose(3, 3) << endl;
                }
                of.close();
            }

        }
    }
}

int debugUwaFailureCases(string& Method) {
    string method = Method;
    cout << "*****************************  " << method << "  *****************************" << endl;
    string dataName = "UWA";
    cout << "debug Failure Cases: " << dataName << endl;

    string evalFileRootPath = "D:/wenhao.sun/Documents/GitHub/1-project/eval/" + method + "Eval/" + dataName + "/";
    // failure case的文件名
    string failureCaseADICenterFilePath = evalFileRootPath + method + "FailureCasesADICenter.txt";
    string failureCaseADICenterOcclusionFilePath = evalFileRootPath + method + "FailureCasesADICenterOcclusion.txt";
    
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/" + dataName + "/";
    string configPath = "3D models/Mian/";
    string modelPath = rootPath + configPath;
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/testResults/" + method + "Results/" + dataName + "/";

    map< string, vector<string>> failureCaseADICenterTableMap;
    map< string, vector<double>> failureCaseADICenterOcclusionTableMap;
    readMap(failureCaseADICenterTableMap, failureCaseADICenterFilePath);
    readMap(failureCaseADICenterOcclusionTableMap, failureCaseADICenterOcclusionFilePath);
    
    string modelName, modelFilePath, gtName_sceneName, gtName;
    for (auto it = failureCaseADICenterTableMap.begin(); it != failureCaseADICenterTableMap.end(); ++it) {
        modelName = it->first;

        modelFilePath = modelPath + modelName + "_0.ply";
        MyMesh pcM;
        vcg::tri::io::ImporterPLY<MyMesh>::Open(pcM, modelFilePath.c_str());
        Mat pc;
        vcgMesh2cvMat(pcM, pc);

        int fcNum= it->second.size();
        for (int i = 0; i < fcNum; i++) {
            gtName_sceneName = it->second[i]; // "gtName"+ "_" + "sceneName"
            std::vector<std::string> mStr;
            boost::split(mStr, gtName_sceneName, boost::is_any_of("_"));
            gtName = mStr[0];  //也是pred name

            // read pred
            string predFilePath = predPath + gtName + ".txt";
            ifstream pred_ifs(predFilePath);
            if (!pred_ifs.is_open()) exit(1);

            string modelNameInPred;
            getline(pred_ifs, modelNameInPred); ////////
            string timeStr;
            getline(pred_ifs, timeStr);

            Matx44d pred_pose;
            for (int ii = 0; ii < 4; ii++)
                for (int jj = 0; jj < 4; jj++)
                {
                    pred_ifs >> pred_pose(ii, jj);
                }
            pred_ifs.close();

            Mat pct = transformPCPose(pc, pred_pose);
            string debugName = "../samples/data/results/debug_" + method + "_" + dataName + "/" + gtName+".ply";

            writePLY(pct, debugName.c_str());
        }
    }


}

int debugUwa(char** argv)
{
    cout << "******************* debug *******************" << endl;
    cout << argv[1] << " " << argv[2] << endl;

    string dataName = "UWA";
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/" + dataName + "/";
    string configPath = "3D models/Mian/";
    string modelPath = rootPath + configPath;

    //debug
    string gtName = (string)argv[6];
    string debugFolderName = gtName;
    string sceneKeypointFileName = (string)argv[3] + "/" + debugFolderName + "/debug_sampled_scene_ref.ply";
    string debugGTPoseModelPath = (string)argv[3] + "/" + debugFolderName + "/debug_GTPoseModel.ply";
    string gtPath = rootPath + configPath + "/GroundTruth_3Dscenes/" + gtName + ".xf";
    bool debug = true;


    string samplingMethod = "cube"; // cube, normal
    double orientationDiffThreshold = 2. / 180. * M_PI;
    double nmsThreshold = 0.5; //0.2 0.5
    bool refineEnabled = true;
    bool nmsEnabled = true;
    //bool refineEnabled = false;
    //bool nmsEnabled = false;

    string modelFileName = modelPath + (string)argv[1];
    string sceneFileName = modelPath + (string)argv[2];
    std::vector<std::string> mStr, mn;
    boost::split(mStr, modelFileName, boost::is_any_of("/"));
    boost::split(mn, *(mStr.end() - 1), boost::is_any_of("."));
    std::vector<std::string> sStr, sn;
    boost::split(sStr, sceneFileName, boost::is_any_of("/"));
    boost::split(sn, *(sStr.end() - 1), boost::is_any_of("."));
    string resultFileName = (string)argv[3] + "/" + debugFolderName + "/" + mn[0] + "-" + sn[0] + "-" + "PCTrans"; //resultFileName = "../samples/data/results//parasaurolophus_high_0-rs7_0-PCTrans0.ply"
    string sdfFileName = modelPath + mn[0] + ".cdf";
    float keypointStep = stoi((string)argv[4]);  // 5
    size_t N = stoi((string)argv[5]);  // 3

    double relativeSamplingStep = 0.025;
    double relativeSceneDistance = 0.025;

    // Create an instance of ICP
    ICP icp(5, 0.005f, 0, 3); //100, 0.005f, 2.5f, 8; (5, 0.005f, .05f, 3);

    // read gt pose
    ifstream gt_ifs(gtPath);
    if (!gt_ifs.is_open()) { cout << "not open: " << gtPath << endl; exit(1); }
    Matx44d gt_pose;
    for (int ii = 0; ii < 4; ii++)
        for (int jj = 0; jj < 4; jj++)
        {
            gt_ifs >> gt_pose(ii, jj);
        }
    gt_ifs.close();

    // 截取的ROI区域的keypoint
    MyMesh pcK;
    vcg::tri::io::ImporterPLY<MyMesh>::Open(pcK, sceneKeypointFileName.c_str());
    Mat sceneKeypoint;
    vcgMesh2cvMat(pcK, sceneKeypoint);
    //for (int i = 0; i < 6; i++) {
    //    cout << sceneKeypoint.row(i) << "vcg" << endl;
    //}

    // 手动检查法向量！
    //Mat pc = loadPLYSimple(modelFileName.c_str(), 1);

    MyMesh pcM;
    vcg::tri::io::ImporterPLY<MyMesh>::Open(pcM, modelFileName.c_str());
    Mat pc;
    vcgMesh2cvMat(pcM, pc);

    //for (int i = 0; i < pc_vcg.rows; i++) {
    //    cout << pc.row(i) << "loadPLYSimple" <<  endl;
    //    cout << pc_vcg.row(i) << "vcg" << endl;
    //}

    // load sdf
    std::cout << "Load SDF...";
    auto sdf = std::shared_ptr<Discregrid::DiscreteGrid>{};
    sdf = std::shared_ptr<Discregrid::CubicLagrangeDiscreteGrid>(
        new Discregrid::CubicLagrangeDiscreteGrid(sdfFileName));
    std::cout << "DONE" << std::endl;

    // Now train the model
    cout << "Training..." << endl;
    int64 tick1 = cv::getTickCount();
    ppf_match_3d::PPF3DDetector detector(relativeSamplingStep);//0.025, 0.05
    detector.enableDebug(debug);
    detector.setSceneKeypointForDebug(sceneKeypoint);
    detector.setDebugFolderName(debugFolderName);

    detector.setOrientationDiffThreshold(orientationDiffThreshold);
    detector.setSamplingMethod(samplingMethod);
    detector.setNMSThreshold(nmsThreshold);
    detector.setSDF(sdf);
    detector.trainModel(pc); //// 

    detector.setGtPose(gt_pose); // 一定在train之后
    detector.saveGTPoseModel(debugGTPoseModelPath);
    int64 tick2 = cv::getTickCount();
    cout << endl << "Training complete in "
        << (double)(tick2 - tick1) / cv::getTickFrequency()
        << " sec" << endl << "Loading model..." << endl;

    // Read the scene
    tick1 = cv::getTickCount();

    MyMesh pcS; // uwa dataset
    vcg::tri::io::ImporterPLY<MyMesh>::Open(pcS, sceneFileName.c_str());
    if (pcS.face.size() == 0) {
        cout << "Scene has no face. Need faces for normal computing by PerVertexFromCurrentFaceNormal" << endl;
        return -1;
    }
    tri::UpdateNormal<MyMesh>::PerFace(pcS);
    tri::UpdateNormal<MyMesh>::PerVertexFromCurrentFaceNormal(pcS);
    tri::UpdateNormal<MyMesh>::NormalizePerVertex(pcS);

    Mat pcTest;
    vcgMesh2cvMat(pcS, pcTest);
    //for (int i = 0; i < 1; i++) {
    //    vcg::Point3f n = pcS.vert[i].N();
    //    vcg::Point3f p = pcS.vert[i].P();
    //    vcg::Point3f fn = pcS.face[i].N();
    //    //
    //    cout << p[0] << " " << p[1] << " " << p[2] << " " << n[0] << " " << n[1] << " " << n[2] << " " << endl;
    //    cout << fn[0] << " " << fn[1] << " " << fn[2] << " " << endl;
    //}
    //
    // ....
    //Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);
    //Mat pcTest = loadPLYSimple_bin(sceneFileName.c_str(), 1); // uwa dataset

    tick2 = cv::getTickCount();
    cout << endl << "Read Scene Elapsed Time " <<
        (tick2 - tick1) / cv::getTickFrequency() << " sec" << endl;

    // Match the model to the scene and get the pose
    cout << endl << "Starting matching..." << endl;
    vector<Pose3DPtr> results;
    tick1 = cv::getTickCount();
    //debugMatch
    //detector.debugMatch(pcTest, results, 1.0 / keypointStep, relativeSceneDistance); 

    detector.match(pcTest, results, 1.0 / keypointStep, relativeSceneDistance); //1.0/40.0, 0.05；作者建议1.0/5.0，0.025
    
    
    tick2 = cv::getTickCount();
    cout << endl << "PPF Elapsed Time " <<
        (tick2 - tick1) / cv::getTickFrequency() << " sec" << endl;

    //check results size from match call above
    size_t results_size = results.size();
    cout << "Number of matching poses: " << results_size;
    if (results_size == 0) {
        cout << endl << "No matching poses found. Exiting." << endl;
        exit(0);
    }


    cout << endl << "Performing ICP,NMS on " << results.size() << " poses..." << endl;
    int64 t1 = cv::getTickCount();

    // 后处理
    detector.postProcessing(results, icp, refineEnabled, nmsEnabled); /////////

    int64 t2 = cv::getTickCount();
    cout << endl << "ICP,NMS Elapsed Time " <<
        (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

    // Get only first N results - but adjust to results size if num of results are less than that specified by N
    if (results.size() < N) {
        cout << endl << "Reducing matching poses to be reported (as specified in code): "
            << N << " to the number of matches found: " << results.size() << endl;
        N = results.size();
    }
    vector<Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

    //// Create an instance of ICP
    //ICP icp(100, 0.005f, 2.5f, 8);
    //int64 t1 = cv::getTickCount();

    // Register for all selected poses
    //cout << endl << "Performing ICP on " << N << " poses..." << endl;
    // 
    Mat pc_sampled = detector.getSampledModel();

    //icp.registerModelToScene(pc_sampled, pcTest, resultsSub);
    //
    // int64 t2 = cv::getTickCount();
    //
    //cout << endl << "ICP Elapsed Time " <<
    //     (t2-t1)/cv::getTickFrequency() << " sec" << endl;

    cout << "Poses: " << endl;
    // debug first five poses
    //Mat pc_sampled = detector.getSampledModel();
    for (size_t i = 0; i < resultsSub.size(); i++)
    {
        Pose3DPtr result = resultsSub[i];
        cout << "Pose Result " << i << endl;
        result->printPose();
        cout << cv::norm(result->q) << endl;
        //if (i==0)
        {
            Mat pct = transformPCPose(pc_sampled, result->pose); //pc是原始模型
            writePLY(pct, (resultFileName + to_string(i) + ".ply").c_str());
            //writePLY(pct, ("../samples/data/results/obj_000007_" + to_string(i) + ".ply").c_str());
        }
    }

    return 0;

}

int debugIcbin(char** argv) {
    string dataName = "icbin";
    string method = "ppf_2025";
    cout << "debug " << method << " on dataset " << dataName << endl;


    //para
    string samplingMethod = "cube"; // cube, normal
    double orientationDiffThreshold = 5 / 180. * M_PI; // 10
    double nmsThreshold = 0.5; //0.2 0.5
    bool refineEnabled = true;
    bool nmsEnabled = true;
    double ppf_angle_constraint = 5 / 180. * M_PI; //5 ， 偏离90度至少20度，点对中1号点的法向与点对连线的夹角，与90度之差的绝对值，大于给定阈值
    double ppf_distance_constraint = 0;  // 0, 1好点和2好点，至少相聚10*sampled resolution

    // model
    double relativeSamplingStep = 0.03;
    // match
    double keypointStep = 5;
    double relativeSceneDistance = 0.05;
    // Create an instance of ICP
    ICP icp(5, 0.005f, 0, 1); //100, 0.005f, 2.5f, 8
    //path
    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/IC-BIN-cvpr16/cvpr16_scenario_2/";
    //string configPath = "3D models/Mian/";

    string modelPath = rootPath + "meshes/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/testResults/" + method + "Results/" + dataName + "/";

    //debug
    bool debug = true;

    string scenar = (string)argv[1];
    string sceneName = (string)argv[2];
    string modelName = (string)argv[3];
    string sceneFileName = rootPath + scenar + "/cloud/cloud" + sceneName + ".ply"; //cloud0.ply
    //string modelFileName = modelPath + modelName + "_wt.ply";
    string modelFileName = modelPath + modelName + "_plain.ply";
    //string modelFileName = modelPath + modelName + "_pd.ply";
    string debugFolderName = scenar + "-" + sceneName + "-" + modelName; // scenar scenename modelname
    string resultFileName = (string)argv[4] + "/" + debugFolderName + "/" + scenar + "-" + sceneName + "-" + modelName + "-" + "PCTrans"; //resultFileName = "../samples/data/results//coffee_cup 0 coffee_cup-PCTrans0.ply"
    string debugGTPoseModelPath = (string)argv[4] + "/" + debugFolderName + "/";
    string sdfFileName = modelPath + modelName + "_sdf.cdf";
    size_t N = stoi((string)argv[5]);  // 3

    // read gt pose
    string gtPath; //coffee_cup1_0, modelname index sceneid, index from 1
    vector<Matx44d> gt_poses(N);
    for (int igt = 0; igt < N; igt++) {
        gtPath = rootPath + scenar + "/" + modelName + to_string(igt + 1) + "_" + sceneName + ".txt";
        ifstream gt_ifs(gtPath);
        if (!gt_ifs.is_open()) { cout << "not open: " << gtPath << endl; exit(1); }
        for (int ii = 0; ii < 4; ii++)
            for (int jj = 0; jj < 4; jj++)
            {
                gt_ifs >> gt_poses[igt](ii, jj);
            }
        gt_ifs.close();
    }

    //Mat pc = loadPLYSimple(modelFileName.c_str(), 1);

    MyMesh pcM;
    vcg::tri::io::ImporterPLY<MyMesh>::Open(pcM, modelFileName.c_str());
    tri::UpdateNormal<MyMesh>::PerFace(pcM);
    tri::UpdateNormal<MyMesh>::PerVertexFromCurrentFaceNormal(pcM);
    tri::UpdateNormal<MyMesh>::NormalizePerVertex(pcM);
    Mat pc;
    vcgMesh2cvMat(pcM, pc);

    //for (int i = 0; i < pc_vcg.rows; i++) {
    //    cout << pc.row(i) << "loadPLYSimple" <<  endl;
    //    cout << pc_vcg.row(i) << "vcg" << endl;
    //}

    // load sdf
    std::cout << "Load SDF...";
    auto sdf = std::shared_ptr<Discregrid::DiscreteGrid>{};
    sdf = std::shared_ptr<Discregrid::CubicLagrangeDiscreteGrid>(
        new Discregrid::CubicLagrangeDiscreteGrid(sdfFileName));
    std::cout << "DONE" << std::endl;

    // Now train the model
    cout << "Training..." << endl;
    int64 tick1 = cv::getTickCount();
    ppf_match_3d::PPF3DDetector detector(relativeSamplingStep);//0.025, 0.05
    detector.enableDebug(debug);
    //detector.setSceneKeypointForDebug(sceneKeypoint);
    detector.setDebugFolderName(debugFolderName);

    detector.setOrientationDiffThreshold(orientationDiffThreshold);
    detector.setSamplingMethod(samplingMethod);
    detector.setNMSThreshold(nmsThreshold); 
    detector.setSDF(sdf);

    detector.setPPFAngleConstraint(ppf_angle_constraint); /// 一定在trian之前
    detector.setPPFDistanceConstraint(ppf_distance_constraint); /// 一定在trian之前
    
    detector.trainModel(pc); //// 

    detector.setMultiGtPose(gt_poses, debugGTPoseModelPath); // 一定在train之后
    //detector.saveGTPoseModel(debugGTPoseModelPath);
    int64 tick2 = cv::getTickCount();
    cout << endl << "Training complete in "
        << (double)(tick2 - tick1) / cv::getTickFrequency()
        << " sec" << endl << "Loading model..." << endl;

    // Read the scene
    tick1 = cv::getTickCount();

    Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);
    //Mat pcTest = loadPLYSimple_bin(sceneFileName.c_str(), 1); // uwa dataset

    tick2 = cv::getTickCount();
    cout << endl << "Read Scene Elapsed Time " <<
        (tick2 - tick1) / cv::getTickFrequency() << " sec" << endl;

    // Match the model to the scene and get the pose
    cout << endl << "Starting matching..." << endl;
    vector<Pose3DPtr> results;
    tick1 = cv::getTickCount();
    //debugMatch
    //detector.debugMatch(pcTest, results, 1.0 / keypointStep, relativeSceneDistance); 

    detector.match(pcTest, results, 1.0 / keypointStep, relativeSceneDistance); //1.0/40.0, 0.05；作者建议1.0/5.0，0.025


    tick2 = cv::getTickCount();
    cout << endl << "PPF Elapsed Time " <<
        (tick2 - tick1) / cv::getTickFrequency() << " sec" << endl;

    //check results size from match call above
    size_t results_size = results.size();
    cout << "Number of matching poses: " << results_size;
    if (results_size == 0) {
        cout << endl << "No matching poses found. Exiting." << endl;
        exit(0);
    }


    cout << endl << "Performing ICP,NMS on " << results.size() << " poses..." << endl;
    int64 t1 = cv::getTickCount();

    // 后处理
    detector.postProcessing(results, icp, refineEnabled, nmsEnabled); /////////

    int64 t2 = cv::getTickCount();
    cout << endl << "ICP,NMS Elapsed Time " <<
        (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

    // Get only first N results - but adjust to results size if num of results are less than that specified by N
    if (results.size() < N) {
        cout << endl << "Reducing matching poses to be reported (as specified in code): "
            << N << " to the number of matches found: " << results.size() << endl;
        N = results.size();
    }
    vector<Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

    //// Create an instance of ICP
    //ICP icp(100, 0.005f, 2.5f, 8);
    //int64 t1 = cv::getTickCount();

    // Register for all selected poses
    //cout << endl << "Performing ICP on " << N << " poses..." << endl;
    // 
    Mat pc_sampled = detector.getSampledModel();

    //icp.registerModelToScene(pc_sampled, pcTest, resultsSub);
    //
    // int64 t2 = cv::getTickCount();
    //
    //cout << endl << "ICP Elapsed Time " <<
    //     (t2-t1)/cv::getTickFrequency() << " sec" << endl;

    cout << "Poses: " << endl;
    // debug first five poses
    //Mat pc_sampled = detector.getSampledModel();
    for (size_t i = 0; i < resultsSub.size(); i++)
    {
        Pose3DPtr result = resultsSub[i];
        cout << "Pose Result " << i << endl;
        result->printPose();
        //cout << cv::norm(result->q) << endl;
        //if (i==0)
        {
            Mat pct = transformPCPose(pc_sampled, result->pose); //pc是原始模型
            writePLY(pct, (resultFileName + to_string(i) + ".ply").c_str());
            //writePLY(pct, ("../samples/data/results/obj_000007_" + to_string(i) + ".ply").c_str());
        }
    }

    return 0;
}



int main(int argc, char** argv) {
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

    // uwa
    //testUwa();
    //debugUwaFailureCases(method);
    //debugUwa(argv);

    // kinect
    //testKinect();

    // lmo
    //testLmo();

    // icbin
    //testIcbin();
    debugIcbin(argv);
}


