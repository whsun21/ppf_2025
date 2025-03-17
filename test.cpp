

#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

#include <Windows.h>

using namespace std;
using namespace cv;
using namespace ppf_match_3d;


int main(int argc, char** argv)
{
    char currentPath[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, currentPath);

    //cout << currentPath;
    string currentPath_s = (string)currentPath;

    string modelFileName =  "../samples/data/parasaurolophus_6700.ply";
    Mat pc = loadPLYSimple(modelFileName.c_str(), 1);

    string resultFileName = "../samples/data/results/para6700-rs1_normals-PCTrans.ply";
    //writePLY(pc, (currentPath_s + "/" + resultFileName).c_str());
    writePLY(pc, (resultFileName).c_str());

    return 0;
    
}
