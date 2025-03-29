// input output
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>
#include <wrap/ply/plylib.cpp> //https://blog.csdn.net/CUSTESC/article/details/106160295


#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

#include <boost/algorithm/string.hpp>
#include <Windows.h>


using namespace std;
using namespace cv;
using namespace ppf_match_3d;

int testUwa() {
    string dataName = "UWA";
    string method = "ppf_2025";
    cout << "test " << method << " on dataset " << dataName << endl;

    //para
    // model
    double relativeSamplingStep = 0.025;
    // match
    double keypointStep = 5;
    double relativeSceneDistance = 0.025;
    int N = 1;
    // Create an instance of ICP
    ICP icp(5, 0.005f, .05f, 3); //100, 0.005f, 2.5f, 8
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
    string modelFilePath;
    for (int i = 0; i < uwaModelName.size(); i++) {
        modelFilePath = modelPath + uwaModelName[i] + "_0.ply";

        MyMesh pcM;
        vcg::tri::io::ImporterPLY<MyMesh>::Open(pcM, modelFilePath.c_str());
        Mat pc;
        vcgMesh2cvMat(pcM, pc);

        //Mat pc = loadPLYSimple(modelFilePath.c_str(), 1);
        modelPC[uwaModelName[i]] = pc;
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
            return -1;
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

            // train the model
            ppf_match_3d::PPF3DDetector detector(relativeSamplingStep);//0.025
            //detector.enableDebug(true);
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
            int postPoseNum = 100;
            if (results_size < postPoseNum) postPoseNum = results_size;
            vector<Pose3DPtr> resultsPost(results.begin(), results.begin() + postPoseNum);

            bool refineEnabled = true;
            bool nmsEnabled = true;
            detector.postProcessing(resultsPost, icp, refineEnabled, nmsEnabled); /////////

            // timer 
            int64 t2 = cv::getTickCount();

            // Get only first N results - but adjust to results size if num of results are less than that specified by N
            if (resultsPost.size() < N) {
                cout << endl << "Reducing matching poses to be reported (as specified in code): "
                    << N << " to the number of matches found: " << resultsPost.size() << endl;
                N = resultsPost.size();
            }
            vector<Pose3DPtr> resultsSub(resultsPost.begin(), resultsPost.begin() + N);

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
    //string method = "halcon";
    string method = "ppf_2025";


    //testUwa();
    debugUwaFailureCases(method);

}


static void help(const string& errorMessage)
{
    cout << "Program init error : " << errorMessage << endl;
    cout << "\nUsage : ppf_matching [input model file] [input scene file]" << endl;
    cout << "\nPlease start again with new parameters" << endl;
}

int main1(int argc, char** argv)
{


    // welcome message
    cout << "****************************************************" << endl;
    cout << "* Surface Matching demonstration : demonstrates the use of surface matching"
        " using point pair features." << endl;
    cout << "* The sample loads a model and a scene, where the model lies in a different"
        " pose than the training.\n* It then trains the model and searches for it in the"
        " input scene. The detected poses are further refined by ICP\n* and printed to the "
        " standard output." << endl;
    cout << "****************************************************" << endl;

    if (argc < 3)
    {
        help("Not enough input arguments");
        exit(1);
    }

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

    string modelFileName = (string)argv[1];
    string sceneFileName = (string)argv[2];
    std::vector<std::string> mStr, mn;
    boost::split(mStr, modelFileName, boost::is_any_of("/"));
    boost::split(mn, *(mStr.end() - 1), boost::is_any_of("."));
    std::vector<std::string> sStr, sn;
    boost::split(sStr, sceneFileName, boost::is_any_of("/"));
    boost::split(sn, *(sStr.end() - 1), boost::is_any_of("."));
    string resultFileName = (string)argv[3] + "/" + mn[0] + "-" + sn[0] + "-" + "PCTrans"; //resultFileName = "../samples/data/results//chicken_small2-rs1_normals-PCTrans.ply"
    float keypointStep = stoi((string)argv[4]);
    size_t N = stoi((string)argv[5]);




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


    // Now train the model
    cout << "Training..." << endl;
    int64 tick1 = cv::getTickCount();
    ppf_match_3d::PPF3DDetector detector(0.025, 0.025);//0.025, 0.05
    detector.enableDebug(true);
    detector.trainModel(pc); //// pc_vcg
    int64 tick2 = cv::getTickCount();
    cout << endl << "Training complete in "
        << (double)(tick2 - tick1) / cv::getTickFrequency()
        << " sec" << endl << "Loading model..." << endl;

    // Read the scene
    tick1 = cv::getTickCount();

    MyMesh pcS;
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
    detector.match(pcTest, results, 1.0 / keypointStep, 0.025); //1.0/40.0, 0.05；作者建议1.0/5.0，0.025
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

    // Create an instance of ICP
    ICP icp(5, 0.005f, 2.5f, 3);

    int64 t1 = cv::getTickCount();
    int postPoseNum = 100;
    if (results_size < postPoseNum) postPoseNum = results_size;
    vector<Pose3DPtr> resultsPost(results.begin(), results.begin() + postPoseNum);

    cout << endl << "Performing ICP,NMS on " << resultsPost.size() << " poses..." << endl;

    // 后处理
    bool refineEnabled = true;
    bool nmsEnabled = true;
    detector.postProcessing(resultsPost, icp, refineEnabled, nmsEnabled); /////////

    int64 t2 = cv::getTickCount();
    cout << endl << "ICP,NMS Elapsed Time " <<
        (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

    // Get only first N results - but adjust to results size if num of results are less than that specified by N
    if (resultsPost.size() < N) {
        cout << endl << "Reducing matching poses to be reported (as specified in code): "
            << N << " to the number of matches found: " << resultsPost.size() << endl;
        N = resultsPost.size();
    }
    vector<Pose3DPtr> resultsSub(resultsPost.begin(), resultsPost.begin() + N);

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
