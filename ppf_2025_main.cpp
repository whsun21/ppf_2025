// input output
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>
#include <wrap/ply/plylib.cpp> //https://blog.csdn.net/CUSTESC/article/details/106160295

//class MyFace;
//class MyVertex;
//
//struct MyUsedTypes : public vcg::UsedTypes<	vcg::Use<MyVertex>::AsVertexType, vcg::Use<MyFace>::AsFaceType> {};
//
//class MyVertex : public vcg::Vertex< MyUsedTypes, vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::Color4b, vcg::vertex::BitFlags  > {};
//class MyFace : public vcg::Face < MyUsedTypes, vcg::face::VertexRef, vcg::face::Normal3f, vcg::face::Color4b, vcg::face::BitFlags > {};
//class MyMesh : public vcg::tri::TriMesh< std::vector<MyVertex>, std::vector<MyFace> > {};


#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

#include <boost/algorithm/string.hpp>




using namespace std;
using namespace cv;
using namespace ppf_match_3d;

static void help(const string& errorMessage)
{
    cout << "Program init error : "<< errorMessage << endl;
    cout << "\nUsage : ppf_matching [input model file] [input scene file]"<< endl;
    cout << "\nPlease start again with new parameters"<< endl;
}

int main(int argc, char** argv)
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
    boost::split(mn, *(mStr.end()-1), boost::is_any_of("."));
    std::vector<std::string> sStr, sn;
    boost::split(sStr, sceneFileName, boost::is_any_of("/"));
    boost::split(sn, *(sStr.end() - 1), boost::is_any_of("."));
    string resultFileName = (string)argv[3]+"/"+ mn[0]+"-"+sn[0] +"-" + "PCTrans"; //resultFileName = "../samples/data/results//chicken_small2-rs1_normals-PCTrans.ply"
    float keypointStep = stoi((string)argv[4]);
    size_t N = stoi((string)argv[5]);

    double relativeSamplingStep = 0.025;
    double relativeSceneDistance = 0.025;



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
    ppf_match_3d::PPF3DDetector detector(relativeSamplingStep);//0.025, 0.05
    detector.enableDebug(true);
    detector.trainModel(pc); //// pc_vcg
    int64 tick2 = cv::getTickCount();
    cout << endl << "Training complete in "
         << (double)(tick2-tick1)/ cv::getTickFrequency()
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
    detector.match(pcTest, results, 1.0/ keypointStep, relativeSceneDistance); //1.0/40.0, 0.05；作者建议1.0/5.0，0.025
    tick2 = cv::getTickCount();
    cout << endl << "PPF Elapsed Time " <<
         (tick2-tick1)/cv::getTickFrequency() << " sec" << endl;

    //check results size from match call above
    size_t results_size = results.size();
    cout << "Number of matching poses: " << results_size;
    if (results_size == 0) {
        cout << endl << "No matching poses found. Exiting." << endl;
        exit(0);
    }

    // Create an instance of ICP
    ICP icp(5, 0.005f, .05f, 3); //100, 0.005f, 2.5f, 8

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
         (t2-t1)/cv::getTickFrequency() << " sec" << endl;

    // Get only first N results - but adjust to results size if num of results are less than that specified by N
    if (resultsPost.size() < N) {
        cout << endl << "Reducing matching poses to be reported (as specified in code): "
             << N << " to the number of matches found: " << resultsPost.size() << endl;
        N = resultsPost.size();
    }
    vector<Pose3DPtr> resultsSub(resultsPost.begin(), resultsPost.begin()+N);
    
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
    for (size_t i=0; i<resultsSub.size(); i++)
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
