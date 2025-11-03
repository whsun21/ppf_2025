// pcl
#include <iostream>

//#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/io.h>
//#include <pcl/io/ply_io.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/ply_io.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h> //下一小节使用

#include "opencv2/core/utility.hpp"
#include "surface_matching/ppf_helpers.hpp"


using namespace std;
using namespace pcl;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;


int showVotes() {
    //read votes
    string rootPath = "D:/wenhao.sun/Documents/GitHub/1-project/ppf_2025/samples/data/results/parasaurolophus-rs7/";
    string poseInstance = "debug_afterHoughVot_1_numVotes_19.000000";

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


    //set PCL viewer0
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    //设置点云颜色

    viewer.setBackgroundColor(0.0, 0.0, 0.0);// rgb(41, 133, 122)
    //viewer.setBackgroundColor(168 / 255., 160 / 255., 138 / 255.); //142/255.,169/255.,127/255.

    viewer.addPointCloud<PointType>(mi_cloud, "mi");  // white
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "mi");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "mi");
    //viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(mi_cloud, mi_normals, 1, 6, "mi cloud normals");

    viewer.addPointCloud<PointType>(mj_cloud, "mj");  // white
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "mj");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 1, "mj");
    //viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(mj_cloud, mj_normals, 1, 6, "mj cloud normals");

    viewer.addPointCloud<PointType>(si_cloud, "si");  // white
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "si");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.5, 0.5, 1, "si");
    //viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(si_cloud, si_normals, 1, 6, "si cloud normals");

    viewer.addPointCloud<PointType>(sj_cloud, "sj");  // white
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sj");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 1, "sj");
    //viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(sj_cloud, sj_normals, 1, 6, "sj cloud normals");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 1, "sj cloud normals");

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

int getPlane() {
    string scene_p = "D:/wenhao.sun/Documents/datasets/object_recognition/IC-BIN-cvpr16/cvpr16_scenario_2/coffee_cup/cloud/cloud26.ply";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    //pcl::io::loadPLYFile(scene_p, *cloud_in);
    PointCloud<PointNormal>::Ptr cloud_pn(new PointCloud<PointNormal>());
    if (pcl::io::loadPLYFile(scene_p, *cloud_pn) == -1)
    {
        PCL_ERROR("read false");
        return 0;
    }
    pcl::copyPointCloud(*cloud_pn, *cloud_in);


    pcl::SACSegmentation<pcl::PointXYZ> seg(true);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_PROSAC); //SAC_RANSAC
    seg.setMaxIterations(10);
    seg.setDistanceThreshold(0.08);

    //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);//下一小节使用

    //pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::ExtractIndices<pcl::PointNormal> extract;

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    //std::vector<pcl::ModelCoefficients> coeffs(3);
    //std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> planes(3);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::ModelCoefficients coeff;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr plane;

    //*cloud_copy = *cloud_in;

    pcl::PointCloud<pcl::PointNormal>::Ptr plane(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr keeped(new pcl::PointCloud<pcl::PointNormal>);
    seg.setInputCloud(cloud_in);
    //tree->setInputCloud(cloud_copy);//下一小节使用
    //seg.setSamplesMaxDist(3, tree);//下一小节使用
    seg.segment(*inliers, coeff);

    extract.setInputCloud(cloud_pn);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plane);
    extract.setNegative(true);
    extract.filter(*keeped);

    cv::Mat pcTest = cv::Mat(keeped->size(), 6, CV_32FC1);
    for (int i = 0; i < keeped->size(); i++)
    {
        float* data = pcTest.ptr<float>(i);
        // scene_normal->at(i).x *= 1000.0;
        // scene_normal->at(i).y *= 1000.0;
        // scene_normal->at(i).z *= 1000.0;
        data[0] = keeped->at(i).x;
        data[1] = keeped->at(i).y;
        data[2] = keeped->at(i).z;
        data[3] = keeped->at(i).normal_x;
        data[4] = keeped->at(i).normal_y;
        data[5] = keeped->at(i).normal_z;
    }
    cv::ppf_match_3d::writePLY(pcTest, "seg.ply");
    

    //plane = cloud_temp;

    //for (int i = 0; i < 3; i++) {
    //    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
    //    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remaining(new pcl::PointCloud<pcl::PointXYZ>);
    //    seg.setInputCloud(cloud_copy);
    //    //tree->setInputCloud(cloud_copy);//下一小节使用
    //    //seg.setSamplesMaxDist(3, tree);//下一小节使用
    //    seg.segment(*inliers, coeffs[i]);

    //    extract.setInputCloud(cloud_copy);
    //    extract.setIndices(inliers);
    //    extract.setNegative(false);
    //    extract.filter(*cloud_temp);
    //    extract.setNegative(true);
    //    extract.filter(*cloud_remaining);

    //    planes[i] = cloud_temp;
    //    cloud_copy = cloud_remaining;
    //}

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer"));
    int v1 = 1;
    viewer->createViewPort(0, 0, 1, 1, v1);
    viewer->setBackgroundColor(1, 1, 1, v1);
    //viewer->addPointCloud(cloud_in, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_in, 255, 255, 0), "cloud_in");
    viewer->addPointCloud(keeped, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal>(keeped, 255, 0, 0), "keeped");
    viewer->addPointCloud(plane, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal>(plane, 0, 255, 0), "plane");
    //viewer->addPointCloud(planes[0], pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(planes[0], 255, 0, 0), "plane1");
    //viewer->addPointCloud(planes[1], pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(planes[1], 0, 255, 0), "plane2");
    //viewer->addPointCloud(planes[2], pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(planes[2], 0, 0, 255), "plane3");

    while (!viewer->wasStopped()) {
        viewer->spinOnce(10);
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

    //getPlane();

}


