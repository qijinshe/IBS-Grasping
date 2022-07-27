#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>
#include <sensor_msgs/PointCloud2.h>

#include <iostream>
#include <Eigen/Dense>
#include <functional>
#include <math.h>
#include <queue>
#include <unordered_set>

#include <omp.h>

//#define RESO 32
//#define RESO 24
#define RESO 20
using namespace std;
#define BOUND (2*RESO+1)
#define PCL_NO_PRECOMPILE
const int pidnum = 10;

struct PointIBS              
{
    float x;
    float y;
    float z;
    float nor1x;
    float nor1y;
    float nor1z;
    float nor2x;
    float nor2y;
    float nor2z;
    float dis1;
    float dis2;
    float dis3;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
 
POINT_CLOUD_REGISTER_POINT_STRUCT(PointIBS,
    (float,x,x)
    (float,y,y)
    (float,z,z)
    (float,dis1,dis1)
    (float,dis2,dis2)
    (float,dis3,dis3) // for other use(Current: Hand Idx)
    (float, nor1x, nor1x)
    (float, nor1y, nor1y)
    (float, nor1z, nor1z)
    (float, nor2x, nor2x)
    (float, nor2y, nor2y)
    (float, nor2z, nor2z)
);

struct PointIdx
{
    float x;
    float y;
    float z;
    float normal_x;
    float normal_y;
    float normal_z;
    float idx_h;
    float idx_i;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointIdx,
    (float,x,x)
    (float,y,y)
    (float,z,z)
    (float, normal_x, normal_x)
    (float, normal_y, normal_y)
    (float, normal_z, normal_z)
    (float, idx_h, idx_h)
    (float, idx_i, idx_i)
);

struct Idxpoint
{
    int x, y, z;
    Idxpoint(int a, int b, int c): x(a), y(b), z(c){};
    bool operator==(const Idxpoint &idx) const
    {
        return x==idx.x && y==idx.y && z==idx.z;
    };
};

struct hash_idx
{
    size_t operator()(const Idxpoint &idx) const
    {
        return hash<int>()(idx.x * BOUND * BOUND + idx.y * BOUND + idx.z);
    }
};

struct Feature
{
    float x,y,z, n1x, n1y, n1z, n2x, n2y, n2z, d1, d2;
};

class Surface
{

    //Function
    public:
        // Surface(float radius);
        Surface(float radius, pcl::PointXYZ* obj_center, pcl::PointXYZ* hand_center, void (*pf)(const Eigen::Vector3f&, float&, float&, float&, Eigen::Vector3f&, Eigen::Vector3f&));
        Eigen::Vector3f transfer(Idxpoint idx);
        void initialize();
        void find_ibs();
        float query(Idxpoint idx);
        bool exquery(Idxpoint idx);
        void expand(Idxpoint idx);
        void surrending_check(Idxpoint idx, int sur);
        bool surrending_check2(Idxpoint idx, int sur);
        void test(Idxpoint idx);
        // void createIBSPoint(int x, int y, int z);
        void createIBSPoint(Eigen::Vector3f& vec, PointIBS& pointibs);
    private:
        inline void iteration(PointIBS* pt, int idx);
        void (*pf)(const Eigen::Vector3f&, float&, float&, float&, Eigen::Vector3f&, Eigen::Vector3f&);
    // Attribute
    public:
        int counter;
        int qcounter;
        float scale;
        float xf, yf, zf;
        pcl::PointCloud<PointIBS> cloud;
        // pcl::PointCloud<pcl::PointXYZ> cloud;
        // pcl::PointCloud<pcl::PointXYZ> cloudt;
    private:
        bool ***vis;
        bool ***exp;
        bool ***incircle;
        float ***val;
        PointIBS ***ibs;
        pcl::PointXYZ *obj_center;
        pcl::PointXYZ *hand_center;
        queue<Idxpoint> expandin;
        unordered_set<Idxpoint, hash_idx> candidates1;
        unordered_set<Idxpoint, hash_idx> candidates2;
};