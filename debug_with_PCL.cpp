// pcl
#include <iostream>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/io.h>
#include <pcl/io/ply_io.h>


using namespace std;
using namespace pcl;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;


int showVotes() {
    //read votes
    string rootPath = "D:/wenhao.sun/Documents/GitHub/1-project/ppf_2025/samples/data/results/parasaurolophus-rs7/";
    string poseInstance = "debug_afterHoughVot_1_numVotes_77.000000";

    cout << "****************** show votes ******************" << endl;
    cout << "path: " << rootPath << endl;
    cout << "case: " << poseInstance << endl;

    string miPath = rootPath + poseInstance + "_mi.ply";
    string mjPath = rootPath + poseInstance + "_mj.ply";
    string siPath = rootPath + poseInstance + "_si.ply";
    string sjPath = rootPath + poseInstance + "_sj.ply";
    pcl::PointCloud<PointXYZRGBNormal>::Ptr mi(new pcl::PointCloud<PointXYZRGBNormal >);
    pcl::io::loadPLYFile(miPath, *mi);
    pcl::PointCloud<PointXYZRGBNormal>::Ptr mj(new pcl::PointCloud<PointXYZRGBNormal >);
    pcl::io::loadPLYFile(mjPath, *mj);
    pcl::PointCloud<PointXYZRGBNormal>::Ptr si(new pcl::PointCloud<PointXYZRGBNormal >);
    pcl::io::loadPLYFile(siPath, *si);
    pcl::PointCloud<PointXYZRGBNormal>::Ptr sj(new pcl::PointCloud<PointXYZRGBNormal >);
    pcl::io::loadPLYFile(sjPath, *sj);

    pcl::PointCloud<PointType>::Ptr mi_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<NormalType>::Ptr mi_normals(new pcl::PointCloud<NormalType>);
    pcl::copyPointCloud(*mi, *mi_cloud);
    pcl::copyPointCloud(*mi, *mi_normals);

    pcl::PointCloud<PointType>::Ptr mj_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<NormalType>::Ptr mj_normals(new pcl::PointCloud<NormalType>);
    pcl::copyPointCloud(*mj, *mj_cloud);
    pcl::copyPointCloud(*mj, *mj_normals);

    pcl::PointCloud<PointType>::Ptr si_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<NormalType>::Ptr si_normals(new pcl::PointCloud<NormalType>);
    pcl::copyPointCloud(*si, *si_cloud);
    pcl::copyPointCloud(*si, *si_normals);

    pcl::PointCloud<PointType>::Ptr sj_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<NormalType>::Ptr sj_normals(new pcl::PointCloud<NormalType>);
    pcl::copyPointCloud(*sj, *sj_cloud);
    pcl::copyPointCloud(*sj, *sj_normals);


    //set PCL viewer
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);

    viewer.addPointCloud<PointType>(mi_cloud, "mi");  // white
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "mi");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "mi");
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(mi_cloud, mi_normals, 1, 6, "mi cloud normals");

    viewer.addPointCloud<PointType>(mj_cloud, "mj");  // white
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "mj");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 1, "mj");
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(mj_cloud, mj_normals, 1, 6, "mj cloud normals");

    viewer.addPointCloud<PointType>(si_cloud, "si");  // white
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "si");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.5, 0.5, 1, "si");
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(si_cloud, si_normals, 1, 6, "si cloud normals");

    viewer.addPointCloud<PointType>(sj_cloud, "sj");  // white
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sj");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 1, "sj");
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(sj_cloud, sj_normals, 1, 6, "sj cloud normals");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 1, "sj cloud normals");

    // line
    for (int i = 0; i < mj->size(); i++) {
        std::stringstream ss_line;
        ss_line << "correspondence_line" << i;
        PointType& model_point = mj_cloud->at(i);
        PointType& scene_point = sj_cloud->at(i);

        //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
        viewer.addLine<PointType, PointType>(model_point, scene_point, 0, 255, 0, ss_line.str());

    }

    // show
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
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
    //string method = "halcon";
    string method = "ppf_2025";


    //testUwa();
    //debugUwaFailureCases(method);
    //debug(argv);
    showVotes();

}


