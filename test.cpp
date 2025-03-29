// input output
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>
#include <wrap/ply/plylib.cpp> //https://blog.csdn.net/CUSTESC/article/details/106160295


#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

#include <boost/algorithm/string.hpp>


using namespace std;
using namespace cv;
using namespace ppf_match_3d;

int testUwa() {

    string dataName = "UWA";
    cout << "test " << dataName << endl;

    string rootPath = "D:/wenhao.sun/Documents/datasets/object_recognition/OpenCV_datasets/UWA/";
    string configPath = "3D models/Mian/";
    string predPath = "D:/wenhao.sun/Documents/GitHub/1-project/Halcon_benchmark/halconResults/uwa/";
    string modelPath = rootPath + configPath;

    vector<string> uwaModelName{ "parasaurolophus_high", "cheff", "chicken_high", "T-rex_high" };
    //cout << uwaModelName[0] << endl;
    map< string, vector<double>> ADDMap;
    map< string, vector<double>> ADIMap;
    map< string, vector<double>> centerErrorMap;
    map< string, vector<double>> phiMap;
    map< string, vector<double>> dNormMap;
    map< string, vector<double>> timeMap;
    map< string, vector<double>> occulsionMap;

    for (int i = 0; i < uwaModelName.size(); i++) {
        ADDMap[uwaModelName[i]] = vector<double>();
        ADIMap[uwaModelName[i]] = vector<double>();
        centerErrorMap[uwaModelName[i]] = vector<double>();
        phiMap[uwaModelName[i]] = vector<double>();
        dNormMap[uwaModelName[i]] = vector<double>();
        timeMap[uwaModelName[i]] = vector<double>();
        occulsionMap[uwaModelName[i]] = vector<double>();
    }
    string ADDName = "../eval_" + dataName + (string)"/eval_ADD.txt";
    string ADIName = "../eval_" + dataName + (string)"/eval_ADI.txt";
    string centerErrorName = "../eval_" + dataName + (string)"/eval_centerError.txt";
    string phiName = "../eval_" + dataName + (string)"/eval_phi.txt";
    string dNormName = "../eval_" + dataName + (string)"/eval_dNorm.txt";
    string timeName = "../eval_" + dataName + (string)"/eval_time.txt";
    string occulsionName = "../eval_" + dataName + (string)"/eval_occulsion.txt";

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
    string DiamtersName = "../eval_" + dataName + (string)"/eval_Diamters.txt";


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
            Vec3f cd = ctt_gt.at<Vec3f>(0) - ctt_pred.at<Vec3f>(0);
            //cout << ctt_gt.at<Vec3f>(0) << endl;
            //cout << ctt_pred.at<Vec3f>(0) << endl;
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

    int instancesNum(0);
    for (auto it = ADDMap.begin(); it != ADDMap.end(); ++it) {
        instancesNum += it->second.size();
        cout << it->first << ": " << it->second.size() << endl;
    }
    cout << "Total: " << instancesNum << endl;  //188


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
