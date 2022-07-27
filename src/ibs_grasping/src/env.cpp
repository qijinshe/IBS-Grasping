#include "env.h"
#include "ibs.h"

pcl::PointCloud<pcl::PointXYZ> cloud1;
pcl::PointCloud<pcl::PointXYZ> cloud2;
pcl::PointCloud<pcl::PointNormal> cloud2_normal;
pcl::PointCloud<pcl::PointNormal> cloud2_backup;
pcl::PointCloud<pcl::PointNormal> plane;
pcl::PointCloud<pcl::PointXYZI> cloud2_labeled;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_at1(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<PointIdx>::Ptr cloud_n2(new pcl::PointCloud<PointIdx>());
//
pcl::PointCloud<pcl::PointNormal>::Ptr cloud_f1(new pcl::PointCloud<pcl::PointNormal>());
pcl::PointCloud<pcl::PointNormal>::Ptr cloud_f2(new pcl::PointCloud<pcl::PointNormal>());
pcl::PointCloud<PointIdx>::Ptr cloud_t1(new pcl::PointCloud<PointIdx>());
pcl::PointCloud<PointIdx>::Ptr cloud_t2(new pcl::PointCloud<PointIdx>());
//
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree1;
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2;
//
ros::Publisher pcl_pub;
ros::Publisher pcl_pub_o;
ros::Publisher pcl_pub_h;
ros::Publisher pcl_pub_o_at;
ros::Publisher pcl_pub_h_at;
//
ros::Publisher pcl_hinfo;
ros::Publisher pcl_oinfo;
//
sensor_msgs::PointCloud2 output;
sensor_msgs::PointCloud2 output_o;
sensor_msgs::PointCloud2 output_h;
sensor_msgs::PointCloud2 output_o_at;
sensor_msgs::PointCloud2 output_h_at;
//
sensor_msgs::PointCloud2 hinfo;
sensor_msgs::PointCloud2 oinfo;
//
Surface* sur;
bool *label; 
pcl::PointXYZ* center1;
pcl::PointXYZ* center2;

Eigen::Matrix4f getTransform(float w, float x, float y, float z, float dx, float dy, float dz)
{
    Eigen::Quaterniond q(w, x, y, z);
    Eigen::Matrix3d rx = q.toRotationMatrix();
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0,0) = (double)rx(0, 0);
    transform(0,1) = (double)rx(0, 1);
    transform(0,2) = (double)rx(0, 2);
    transform(1,0) = (double)rx(1, 0);
    transform(1,1) = (double)rx(1, 1);
    transform(1,2) = (double)rx(1, 2);
    transform(2,0) = (double)rx(2, 0);
    transform(2,1) = (double)rx(2, 1);
    transform(2,2) = (double)rx(2, 2);
    transform(0,3) = dx;
    transform(1,3) = dy;
    transform(2,3) = dz;
    return transform;
}


bool publishNewPC()
{
    pcl::toROSMsg(sur->cloud, output);
    pcl::toROSMsg(cloud1, output_h);
    pcl::toROSMsg(cloud2, output_o);
    output.header.frame_id = "object";
    output_o.header.frame_id = "object";
    output_h.header.frame_id = "object";
    pcl_pub.publish(output);
    pcl_pub_o.publish(output_o);
    pcl_pub_h.publish(output_h);
    pcl::toROSMsg(*cloud_at1, output_h_at);
    pcl::toROSMsg(*cloud_n2, output_o_at);
    output_o_at.header.frame_id = "object";
    output_h_at.header.frame_id = "object";
    pcl_pub_o_at.publish(output_o_at);
    pcl_pub_h_at.publish(output_h_at);
    return true;
}


void computeInfoWithKD(const Eigen::Vector3f& pot, 
                const pcl::PointCloud<pcl::PointXYZ>& hcloud, const pcl::PointCloud<pcl::PointXYZ>& ocloud,
                const pcl::KdTreeFLANN<pcl::PointXYZ>& hkd, const pcl::KdTreeFLANN<pcl::PointXYZ>& okd,
                float& hand_dis, float& obj_dis, float& ho_dis,
                Eigen::Vector3f& nor1, Eigen::Vector3f& nor2)
{
    int flag1 = 0;
    int flag2 = 0;
    int K = 1;
    bool tmp = (ho_dis > -1);
    float idx = ho_dis;
    pcl::PointXYZ searchPoint(pot(0), pot(1), pot(2));
    std::vector<int> pointIdxNKNSearch1(K);
    std::vector<int> pointIdxNKNSearch2(K);
    std::vector<float> pointNKNSquaredDistance1(K);
    std::vector<float> pointNKNSquaredDistance2(K);
    flag1 = hkd.nearestKSearch(searchPoint, K, pointIdxNKNSearch1, pointNKNSquaredDistance1); //There may be noise
    flag2 = okd.nearestKSearch(searchPoint, K, pointIdxNKNSearch2, pointNKNSquaredDistance2); //There may be noise
    hand_dis = sqrt(pointNKNSquaredDistance1[0]);
    obj_dis  = sqrt(pointNKNSquaredDistance2[0]);
    // ToHand Normal and ToObj Normal
    nor1(0) = (hcloud[pointIdxNKNSearch1[0]].x - pot(0));
    nor1(1) = (hcloud[pointIdxNKNSearch1[0]].y - pot(1));
    nor1(2) = (hcloud[pointIdxNKNSearch1[0]].z - pot(2));
    nor2(0) = (cloud2_normal[pointIdxNKNSearch2[0]].x - pot(0));
    nor2(1) = (cloud2_normal[pointIdxNKNSearch2[0]].y - pot(1));
    nor2(2) = (cloud2_normal[pointIdxNKNSearch2[0]].z - pot(2));
    // ho_dis = (nor1 - nor2).norm();
    ho_dis = pointIdxNKNSearch1[0]; 
    // Change2Label 4.02
    float plabel = (2 * float(label[pointIdxNKNSearch2[0]]) - 1);
    ho_dis = (ho_dis+100) * plabel;
    nor1.normalize();
    nor2.normalize();
    
    #pragma omp critical
    {
        // if(tmp && cos>0.5 && hand_dis <0.1)
        if(tmp && obj_dis < 0.05)
        {
            PointIdx* point = new PointIdx();
            point->x = cloud2_normal[pointIdxNKNSearch2[0]].x;
            point->y = cloud2_normal[pointIdxNKNSearch2[0]].y;
            point->z = cloud2_normal[pointIdxNKNSearch2[0]].z;
            point->normal_x = cloud2_normal[pointIdxNKNSearch2[0]].normal_x;
            point->normal_y = cloud2_normal[pointIdxNKNSearch2[0]].normal_y;
            point->normal_z = cloud2_normal[pointIdxNKNSearch2[0]].normal_z;
            point->idx_h = pointIdxNKNSearch1[0];
            point->idx_i = idx;
            cloud_n2->push_back(*point);
        }
    }
}


void get_dis(const Eigen::Vector3f& qpoint, float& hand_dis, float& obj_dis, float& ho_dis, \
Eigen::Vector3f& nor1, Eigen::Vector3f& nor2)
{
    // std::cout << qpoint << std::endl;
    computeInfoWithKD(qpoint, cloud1, cloud2, kdtree1, kdtree2, hand_dis, obj_dis, ho_dis, nor1, nor2);
    return;
}
    

bool get_ibs_hand2(ibs_grasping::GetIBSHand2::Request &req, ibs_grasping::GetIBSHand2::Response &res)
{
    pcl::PCLPointCloud2 pc1;
    pcl_conversions::toPCL(req.hand, pc1);
    pcl::fromPCLPointCloud2(pc1, cloud1);
    center1->x = req.hx;
    center1->y = req.hy;
    center1->z = req.hz;
    kdtree1.setInputCloud(cloud1.makeShared());
    cloud_at1->clear();
    cloud_n2->clear();
    sur->find_ibs();
    publishNewPC();
    res.ibs = output;
    res.hcontact = output_h_at;
    res.ocontact = output_o_at;
    return true;
}

bool load_obj2(ibs_grasping::LoadObj2::Request &req, ibs_grasping::LoadObj2::Response &res)
{
    string str = req.name;
    if(!str.empty())
    {
        pcl::io::loadPCDFile<pcl::PointNormal> (req.name, cloud2_backup);
    }
    Eigen::Matrix4f transform_1 = getTransform(req.w, req.x, req.y, req.z, req.dx, req.dy, req.dz);
    pcl::transformPointCloudWithNormals(cloud2_backup, cloud2_normal, transform_1);
    //
    delete[] label;
    int edge = 100;
    int size1 = cloud2_normal.points.size();
    int size2 = (edge * edge);
    float min_z = 100;
    pcl::PointNormal p;
    for(int i = 0; i<size1; i++)
    {
        if(min_z>cloud2_normal.points[i].z)
        {
            min_z = cloud2_normal.points[i].z;
            p = cloud2_normal.points[i];
        }
    }      
    // Construct Plane
    if(!str.empty())
    {
        plane.clear();
        for(int i=0;i<edge;i++)
        for(int j=0;j<edge;j++)
        {
            pcl::PointNormal pt;
            pt.x = 0.08*(i-edge/2);
            pt.y = 0.08*(j-edge/2);
            pt.z = p.z;
            pt.normal[0] = 0;
            pt.normal[1] = 0;
            pt.normal[2] = 1;
            plane.push_back(pt);
        }
    }
    label = new bool[size1+size2];
    for(int i=0;i<(size1+size2);i++)
        if(i<size1)
            label[i] = true;
        else
            label[i] = false;
    cloud2_normal += plane;
    pcl::copyPointCloud(cloud2_normal, cloud2);
    kdtree2.setInputCloud(cloud2.makeShared());
    center2->x = req.dx;
    center2->y = req.dy;
    center2->z = req.dz;
    res.ok = true;
    return true;
}

bool load_obj_FP(ibs_grasping::LoadObjFP::Request &req, ibs_grasping::LoadObjFP::Response &res)
{
    pcl::PCLPointCloud2 pc2;
    pcl_conversions::toPCL(req.hand, pc2);
    pcl::fromPCLPointCloud2(pc2, cloud2_labeled);
    label = new bool[cloud2_labeled.points.size()];
    for(int i=0; i<cloud2_labeled.points.size(); i++)
    {
        label[i] = (cloud2_labeled[i].intensity > 0);
    }
    pcl::copyPointCloud(cloud2_labeled, cloud2_normal);
    pcl::copyPointCloud(cloud2_labeled, cloud2);
    kdtree2.setInputCloud(cloud2.makeShared());
    center2->x = req.hx;
    center2->y = req.hy;
    center2->z = req.hz;
    res.ok = true;
    return true;
}


int main(int argc, char **argv)
{
    center1 = new pcl::PointXYZ(0,0,1);
    center2 = new pcl::PointXYZ(0,0,0);
    // sur = new Surface(2.0, center2, center1, get_dis);
    sur = new Surface(1.6, center2, center1, get_dis);
    ros::init(argc, argv, "ibs_calculator");
    ros::NodeHandle nh1; //publish
    ros::NodeHandle nh2; //service
    //
    pcl_pub = nh1.advertise<sensor_msgs::PointCloud2> ("ibs_points", 10);
    // Objects Points
    pcl_pub_o = nh1.advertise<sensor_msgs::PointCloud2> ("obj_points", 10);
    pcl_pub_h = nh1.advertise<sensor_msgs::PointCloud2> ("hand_points", 10);
    // Supporting Points
    pcl_pub_o_at = nh1.advertise<sensor_msgs::PointCloud2> ("obj_at", 10);
    pcl_pub_h_at = nh1.advertise<sensor_msgs::PointCloud2> ("hand_at", 10);
    pcl_hinfo = nh1.advertise<sensor_msgs::PointCloud2> ("hsample", 10);
    pcl_oinfo = nh1.advertise<sensor_msgs::PointCloud2> ("osample", 10);
    ros::Duration(1.6).sleep();
    ros::ServiceServer service1 = nh2.advertiseService("get_ibs6", get_ibs_hand2);
    ros::ServiceServer service2 = nh2.advertiseService("load_obj6", load_obj2);
    ros::ServiceServer service3 = nh2.advertiseService("load_obj_fp", load_obj_FP);
    ROS_INFO("Ready to handle query");
    ros::spin();
    return 0;
}