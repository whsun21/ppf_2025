

#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

#include <Windows.h>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;
using namespace ppf_match_3d;

static void help(const string& errorMessage)
{
    cout << "Program init error : " << errorMessage << endl;
    cout << "\nUsage : ppf_matching [input model file] [input scene file]" << endl;
    cout << "\nPlease start again with new parameters" << endl;
}

int main(int argc, char** argv)
{
    char currentPath[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, currentPath);
    string currentPath_s = (string)currentPath;

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
    Mat pc = loadPLYSimple(modelFileName.c_str(), 1);

    // Now train the model
    cout << "Training..." << endl;
    int64 tick1 = cv::getTickCount();
    ppf_match_3d::PPF3DDetector detector(0.025, 0.05);//0.025, 0.05
    detector.trainModel(pc);
    int64 tick2 = cv::getTickCount();
    cout << endl << "Training complete in "
        << (double)(tick2 - tick1) / cv::getTickFrequency()
        << " sec" << endl << "Loading model..." << endl;

    // Read the scene
    Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);

    // Match the model to the scene and get the pose
    cout << endl << "Starting matching..." << endl;
    vector<Pose3DPtr> results;
    tick1 = cv::getTickCount();
    detector.match(pcTest, results, 1.0 / keypointStep, 0.025); //1.0/40.0, 0.05；作者建议1.0/5.0
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
    ICP icp(5, 0.005f, 2.5f, 8);

    int64 t1 = cv::getTickCount();
    int postPoseNum = 50;
    if (results_size < postPoseNum) postPoseNum = results_size;
    vector<Pose3DPtr> resultsPost(results.begin(), results.begin() + postPoseNum);

    cout << endl << "Performing ICP on " << resultsPost.size() << " poses..." << endl;

    // 后处理
    detector.postProcessing(resultsPost, icp, pcTest, pc); /////////

    int64 t2 = cv::getTickCount();
    cout << endl << "ICP Elapsed Time " <<
        (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

    // Get only first N results - but adjust to results size if num of results are less than that specified by N
    if (postPoseNum < N) {
        cout << endl << "Reducing matching poses to be reported (as specified in code): "
            << N << " to the number of matches found: " << postPoseNum << endl;
        N = postPoseNum;
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
        }
    }

    return 0;

}
