
#include "surface_matching/ppf_helpers.hpp"

#include<vcg/complex/complex.h>
#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>
#include <wrap/ply/plylib.cpp> //https://blog.csdn.net/CUSTESC/article/details/106160295

using namespace vcg;
using namespace std;
using namespace cv;
using namespace ppf_match_3d;


//class MyVertex; class MyEdge; class MyFace;
//struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>   ::AsVertexType,
//    vcg::Use<MyEdge>     ::AsEdgeType,
//    vcg::Use<MyFace>     ::AsFaceType> {};
//
//class MyVertex : public vcg::Vertex< MyUsedTypes, vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::BitFlags  > {};
//class MyFace : public vcg::Face<   MyUsedTypes, vcg::face::FFAdj, vcg::face::Normal3f, vcg::face::VertexRef, vcg::face::BitFlags > {};
//class MyEdge : public vcg::Edge<   MyUsedTypes> {};
//
//class MyMesh : public vcg::tri::TriMesh< std::vector<MyVertex>, std::vector<MyFace>, std::vector<MyEdge>  > {};
int main(int argc, char** argv)
{
    //if (argc < 2)
    //{
    //    printf("Usage trimesh_base <meshfilename.obj>\n");
    //    return -1;
    //}

    MyMesh m;
    string plyname = "../samples/data/rs22_0.ply";
    //string plyname = "../samples/data/rs22_0_cv.ply";

    int mask;
    vcg::tri::io::Importer<MyMesh>::LoadMask(plyname.c_str(), mask);//

    if (vcg::tri::io::ImporterPLY<MyMesh>::Open(m, plyname.c_str()) != vcg::tri::io::ImporterOFF<MyMesh>::NoError)
    {
        printf("Error reading file  %s\n", argv[1]);
        exit(0);
    }

    vcg::tri::RequirePerVertexNormal(m);
    //vcg::tri::UpdateNormal<MyMesh>::PerVertexNormalized(m);
    printf("Input mesh  vn:%i fn:%i\n", m.VN(), m.FN());
    printf("Mesh has %i vert and %i faces\n", m.VN(), m.FN());

    for (int i = 0; i < 1; i++) {
        
        vcg::Point3f n = m.vert[i].N();
        vcg::Point3f p = m.vert[i].P();
        //
        cout << "origin: " << endl;
        cout << p[0] << " " << p[1] << " " << p[2] << " " << n[0] << " " << n[1] << " " << n[2] << " " << endl;
        if (m.face.size() > 0) {
            vcg::Point3f fn = m.face[i].N();
            cout << fn[0] << " " << fn[1] << " " << fn[2] << " " << endl;
       }
    }

    tri::UpdateNormal<MyMesh>::PerFace(m);
    for (int i = 0; i < 1; i++) {

        vcg::Point3f n = m.vert[i].N();
        vcg::Point3f p = m.vert[i].P();
        //
        cout << "PerFace: " << endl;
        cout << p[0] << " " << p[1] << " " << p[2] << " " << n[0] << " " << n[1] << " " << n[2] << " " << endl;
        if (m.face.size() > 0) {
            vcg::Point3f fn = m.face[i].N();
            cout << fn[0] << " " << fn[1] << " " << fn[2] << " " << endl;
        }

    }


    tri::UpdateNormal<MyMesh>::PerVertexFromCurrentFaceNormal(m);
    for (int i = 0; i < 1; i++) {

        vcg::Point3f n = m.vert[i].N();
        vcg::Point3f p = m.vert[i].P();
        //
        cout << "PerVertexFromCurrentFaceNormal: " << endl;
        cout << p[0] << " " << p[1] << " " << p[2] << " " << n[0] << " " << n[1] << " " << n[2] << " " << endl;
        if (m.face.size() > 0) {
            vcg::Point3f fn = m.face[i].N();
            cout << fn[0] << " " << fn[1] << " " << fn[2] << " " << endl;
        }
    }

    //tri::UpdateNormal<MyMesh>::NormalizePerFace(m);
    //for (int i = 0; i < 1; i++) {

    //    vcg::Point3f n = m.vert[i].N();
    //    vcg::Point3f p = m.vert[i].P();
    //    //
    //    cout << "NormalizePerFace: " << endl;
    //    cout << p[0] << " " << p[1] << " " << p[2] << " " << n[0] << " " << n[1] << " " << n[2] << " " << endl;
    //    cout << fn[0] << " " << fn[1] << " " << fn[2] << " " << endl;
    //}

    tri::UpdateNormal<MyMesh>::NormalizePerVertex(m);
    for (int i = 0; i < 1; i++) {

        vcg::Point3f n = m.vert[i].N();
        vcg::Point3f p = m.vert[i].P();
        //
        cout << "NormalizePerVertex: " << endl;
        cout << p[0] << " " << p[1] << " " << p[2] << " " << n[0] << " " << n[1] << " " << n[2] << " " << endl;
        if (m.face.size() > 0) {
            vcg::Point3f fn = m.face[i].N();
            cout << fn[0] << " " << fn[1] << " " << fn[2] << " " << endl;
        }
    }


    vcg::tri::io::ExporterPLY<MyMesh>::Save(m, "../samples/data/rs22_0_vcg.ply", tri::io::Mask::IOM_VERTNORMAL, false);

    Mat pc;
    vcgMesh2cvMat(m, pc);

    writePLY(pc, "../samples/data/rs22_0_cv.ply");


    return 0;
}