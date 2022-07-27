#include "ibs.h"

Surface::Surface(float radius, pcl::PointXYZ* obj_center, pcl::PointXYZ* hand_center, \
void (*pf)(const Eigen::Vector3f&, float&, float&, float&, Eigen::Vector3f&, Eigen::Vector3f&))
{
    //Boundary
    this->xf = -radius;
    this->yf = -radius;
    this->zf = -radius;
    this->scale = radius/(RESO);
    this->pf = pf;
    this->obj_center = obj_center;
    this->hand_center = hand_center;
    this->vis = new bool**[BOUND];
    this->exp = new bool**[BOUND];
    this->val = new float**[BOUND];
    this->ibs = new PointIBS**[BOUND];
    for(int i=0;i<BOUND;i++)
    {
        vis[i] = new bool*[BOUND];
        exp[i] = new bool*[BOUND];
        val[i] = new float*[BOUND];
        ibs[i] = new PointIBS*[BOUND];
        for(int j=0;j<BOUND;j++)
        {
            vis[i][j] = new bool[BOUND];
            exp[i][j] = new bool[BOUND];
            val[i][j] = new float[BOUND];
            ibs[i][j] = new PointIBS[BOUND];
        }
    }
    this->incircle = new bool**[BOUND+2];
    for(int i=0;i<BOUND+2;i++)
    {
        incircle[i] = new bool*[BOUND+2];
        for(int j=0;j<BOUND+2;j++){
            incircle[i][j] = new bool[BOUND+2];
            for(int k=0;k<BOUND+2;k++)
            {
                Idxpoint nxt(i,j,k);
                int rad = (RESO + 1);
                int xx = (nxt.x - rad)*(nxt.x - rad);
                int yy = (nxt.y - rad)*(nxt.y - rad);
                int zz = (nxt.z - rad)*(nxt.z - rad);
                int dis = xx+yy+zz;
                if(dis < rad*rad)
                    incircle[i][j][k] = true;
                else
                    incircle[i][j][k] = false;
            }
        }    
    }
}

void Surface::initialize()
{
    this->cloud.clear();
    this->candidates1.clear();
    this->candidates2.clear();
    this->counter = 0;
    this->qcounter = 0;
    for(int i=0;i<BOUND;i++)
        for(int j=0;j<BOUND;j++)
        {
            memset(this->vis[i][j], 0, sizeof(bool)*BOUND);
            memset(this->exp[i][j], 0, sizeof(bool)*BOUND);
            memset(this->val[i][j], 0, sizeof(float)*BOUND);
            memset(this->ibs[i][j], 0, sizeof(PointIBS)*BOUND);
        }        
    Eigen::Vector3f hp(0, 0, 0);
    Eigen::Vector3f op, mid;
    int xx,yy,zz;
    op(0) = 1.5*obj_center->x-1.5*hand_center->x;
    op(1) = 1.5*obj_center->y-1.5*hand_center->y;
    op(2) = 1.5*obj_center->z-1.5*hand_center->z;
    op = op/scale;
    while(true)
    {
        mid = (op+hp)/2;
        if(mid.squaredNorm() > RESO*RESO)
        {
            op = mid;
            continue;
        }
        xx = (int)floor(mid(0)) + RESO;
        yy = (int)floor(mid(1)) + RESO;
        zz = (int)floor(mid(2)) + RESO;
        // std::cout << xx << " " << yy << " " << zz << std::endl;
        float mv = query(Idxpoint(xx, yy, zz));
        if(mv <= 0)
            hp = mid;
        else
            op = mid;
        if((hp-op).squaredNorm()<1)
        {
            break;
        }
    }
    // if(abs(query(Idxpoint(xx, yy, zz))) < 1.73 * scale)
    cloud.push_back(ibs[xx][yy][zz]);
    exp[xx][yy][zz] = true;
    this->expandin.push(Idxpoint(xx, yy, zz));
    this->counter += 1;
}

inline void Surface::iteration(PointIBS* pt, int idx)
{
    // float weight = 0.75;
    float weight;
    float cos = (pt->nor1x * pt->nor2x) + (pt->nor1y * pt->nor2y) + (pt->nor1z * pt->nor2z);
    float adis1 = abs(pt->dis1);
    float adis2 = abs(pt->dis2);
    float length = (adis1 - adis2);
    bool flag = (length >=0);
    if(flag)
    {
        weight = 0.5 * (adis1 + adis2)/(adis1 - adis2*cos + 1e-10);
    }
    else
    {
        length = -length;
        weight = 0.5 * (adis1 + adis2)/(adis2 - adis1*cos + 1e-10);
    }
    length = length * weight;
    Eigen::Vector3f vec(pt->x,pt->y,pt->z);
    if(flag)
    {
        vec(0) += length*pt->nor1x;
        vec(1) += length*pt->nor1y;
        vec(2) += length*pt->nor1z;
    }
    else
    {
        vec(0) += length*pt->nor2x;
        vec(1) += length*pt->nor2y;
        vec(2) += length*pt->nor2z;
    }
    Eigen::Vector3f nor1, nor2;
    float dis1, dis2, dis3;
    dis3 = (float)idx;
    (*pf)(vec, dis1, dis2, dis3, nor1, nor2);
    pt->x = vec(0);
    pt->y = vec(1);
    pt->z = vec(2);
    pt->dis1 = dis1;
    pt->dis2 = dis2;
    pt->dis3 = dis3;
    pt->nor1x = nor1(0);
    pt->nor1y = nor1(1);
    pt->nor1z = nor1(2);
    pt->nor2x = nor2(0);
    pt->nor2y = nor2(1);
    pt->nor2z = nor2(2);
}

void Surface::find_ibs()
{
    double start = omp_get_wtime();
    initialize();
    bool last = false;
    int change = 0;
    double mid = omp_get_wtime();
    double tot1 = 0;
    double tot2 = 0;
    int ct = 0;
    double a=0;
    double b=0;
    double c=0;
    double aa = 0;
    double bb = 0;
    double cc = 0;
    double aa1 = 0;
    double aa2 = 0;
    float center = 0;
    while(true)
    {
        std::vector<Idxpoint> v;
        aa = omp_get_wtime();
        while(!this->expandin.empty())
        {
            v.emplace_back(std::move(this->expandin.front()));
            this->expandin.pop();
        }
        // #pragma omp parallel for num_threads(pidnum) schedule(dynamic)
        // for(int i=0;i<v.size();i++)
        // {
        //     Idxpoint idx = v[i];
        //     center = val[idx.x][idx.y][idx.z];
        //     this->expand(idx);
        // }
        bool flag[v.size()][27];
        memset(flag, false, sizeof(bool)*v.size()*27);
        #pragma omp parallel for num_threads(pidnum) collapse(2)
        for(int i=0;i<v.size();i++)
            for(int aaa=0;aaa<27;aaa++)
            {
                Idxpoint idx = v[i];
                center = val[idx.x][idx.y][idx.z];
                flag[i][aaa] = this->surrending_check2(idx, aaa);
            }
        #pragma omp parallel num_threads(pidnum)
        {
            #pragma omp single
            {
                double block2 = omp_get_wtime();
                aa1 += (block2 - aa);
                for(int iii=0;iii<v.size();iii++)
                for(int aaa=0;aaa<27;aaa++)
                {
                    if(flag[iii][aaa])
                    {
                        int i = aaa/9;
                        int j = aaa/3-3*i;
                        int k = aaa-9*i-3*j;
                        i -= 1;
                        j -= 1;
                        k -= 1;
                        Idxpoint nxt(v[iii].x+i,v[iii].y+j,v[iii].z+k);
                        if(vis[nxt.x][nxt.y][nxt.z])
                        {
                            candidates2.insert(nxt);
                        } 
                        else
                        {
                            candidates1.insert(nxt);
                        }
                    }
                }
                double block3 = omp_get_wtime();
                aa2 += (block3 - block2);
            }
        }
        #pragma omp parallel num_threads(pidnum)
        {
            #pragma omp single
            {
                ///////////////////////////
                bb = omp_get_wtime();
                a += (bb-aa);
                ///////////////////////////
                for(auto it = candidates1.begin();it!=candidates1.end();it++)
                {
                    #pragma omp task firstprivate(it)
                    { 
                        query(*it);
                    }
                    candidates2.insert(*it);
                }
                #pragma omp taskwait
                ///////////////////////////
                cc = omp_get_wtime();
                b += (cc-bb);
                ///////////////////////////
                // std::cout << candidates2.size() << std::endl;
                for(auto it=candidates2.begin();it!=candidates2.end();it++)
                {
                    float val = query(*it);
                    //std::cout << val << std::endl;
                    if((!exp[it->x][it->y][it->z])&&(val*center<=0)&&(abs(val)<2*scale))
                    // if((!exp[it->x][it->y][it->z])&&(val*center<=0))
                    {   
                        this->expandin.push(*it);
                        exp[it->x][it->y][it->z] = true;
                        this->counter +=1;
                        if(val < 0)
                        // if(val>=0)
                        {
                            int idx = -1;
                            cloud.push_back(ibs[it->x][it->y][it->z]);
                            idx = cloud.size()-1;
                            #pragma omp task firstprivate(idx)
                            { 
                                iteration(&cloud.points[idx], idx);
                            }
                        }
                    }
                }
                #pragma omp taskwait
                ///////////////////////////
                double dd = omp_get_wtime();
                c += (dd-cc);
            }
        }
        candidates1.clear();
        candidates2.clear();
        if(this->expandin.empty())
            break;
    }
    double end = omp_get_wtime();
    // std::cout << "a: " <<a << std::endl;
    // std::cout << "b: " <<b << std::endl;
    // std::cout << "c: " <<c << std::endl;
    // std::cout << "a1: " << aa1 << std::endl;
    // std::cout << "a2: " << aa2 << std::endl;
    // std::cout << end - start << std::endl;
    // std::cout << end - mid << std::endl;
    // std::cout << tot2/(tot1+1e-10) << std::endl;
    // std::cout << qcounter << std::endl;
    // std::cout << counter << std::endl;
    // std::cout << cloud.size() << std::endl;
}

Eigen::Vector3f Surface::transfer(Idxpoint idx)
{
    float corx = hand_center->x + (idx.x - RESO) * (this->scale);
    float cory = hand_center->y + (idx.y - RESO) * (this->scale);
    float corz = hand_center->z + (idx.z - RESO) * (this->scale);
    // std::cout << corx << std::endl;
    // std::cout << cory << std::endl;
    // std::cout << corz << std::endl;
    // std::cout << "------" << std::endl;
    Eigen::Vector3f vec(corx, cory, corz);
    return vec;
}

float Surface::query(Idxpoint idx)
{
    int x = idx.x;
    int y = idx.y;
    int z = idx.z;
    if(!this->vis[x][y][z])
    {
        Eigen::Vector3f vec = transfer(Idxpoint(x, y, z));
        createIBSPoint(vec, ibs[x][y][z]);
        vis[x][y][z] = true;
        val[x][y][z] = (ibs[x][y][z].dis1-ibs[x][y][z].dis2);
    }
    return this->val[x][y][z];
}

void Surface::createIBSPoint(Eigen::Vector3f& vec, PointIBS& pointibs)
{
    Eigen::Vector3f nor1, nor2;
    float dis1, dis2, dis3;
    dis3 = -1;
    (*pf)(vec, dis1, dis2, dis3, nor1, nor2);
    pointibs.x = vec(0);
    pointibs.y = vec(1);
    pointibs.z = vec(2);
    //
    pointibs.dis1 = dis1;
    pointibs.dis2 = dis2;
    pointibs.dis3 = dis3;
    //
    pointibs.nor1x = nor1(0);
    pointibs.nor1y = nor1(1);
    pointibs.nor1z = nor1(2);
    //
    pointibs.nor2x = nor2(0);
    pointibs.nor2y = nor2(1);
    pointibs.nor2z = nor2(2);

    #pragma omp critical
    {
        qcounter += 1;
    }
}

void Surface::expand(Idxpoint idx)
{
    // this->sur_points.add(idx);
    int x = idx.x;
    int y = idx.y;
    int z = idx.z;
    float center = this->val[x][y][z];
    int check = 0;
    for(int i=-1;i<2;i++)
    {
        for(int j=-1;j<2;j++)
        {
            for(int k=-1;k<2;k++)
            {
                if(!incircle[x+i+1][y+j+1][z+k+1])
                    continue;
                if(i==0&&j==0&&k==0)
                    continue;
                // if((abs(i)+abs(j)+abs(k))==3)
                //     continue;
                Idxpoint nxt(x+i,y+j,z+k);
                if(exp[nxt.x][nxt.y][nxt.z])
                    continue;
                if(!vis[nxt.x][nxt.y][nxt.z])
                {
                    bool flag = false;
                    for(int ii=-1;ii<2;ii++)
                    for(int jj=-1;jj<2;jj++)
                    for(int kk=-1;kk<2;kk++)
                    {   
                        int cos = (-i)*ii + (-j)*jj + (-k)*kk;
                        if(cos<=0)
                            continue;
                        int nx = x+i+ii;
                        int ny = y+j+jj;
                        int nz = z+k+kk;
                        if(!incircle[nx+1][ny+1][nz+1]||!incircle[nx+ii+1][ny+jj+1][nz+kk+1])
                            continue;
                        if(!vis[nx][ny][nz]||val[nx][ny][nz]*center<0)
                            continue;
                        if(vis[nx+ii][ny+jj][nz+kk]&&val[nx+ii][ny+jj][nz+kk]*center<0)
                        {
                            flag = true;
                            break;
                        }
                    }
                    if(flag)
                        continue;
                }
                if(vis[nxt.x][nxt.y][nxt.z])
                {
                    #pragma omp critical
                    {
                        candidates2.insert(nxt);
                    }
                } 
                else
                {
                    #pragma omp critical
                    {
                        candidates1.insert(nxt);
                    }
                }
            }
        }  
    }
}


// void Surface::expand(Idxpoint idx)
// {
//     // this->sur_points.add(idx);
//     int x = idx.x;
//     int y = idx.y;
//     int z = idx.z;
//     float center = this->val[x][y][z];
//     int check = 0;
//     // #pragma omp task
//     for(int aaa=0;aaa<27;aaa++)
//     {
//         int i = aaa/9;
//         int j = aaa/3-3*i;
//         int k = aaa-9*i-3*j;
//         i -= 1;
//         j -= 1;
//         k -= 1;
//         if(!incircle[x+i+1][y+j+1][z+k+1])
//             continue;
//         if(i==0&&j==0&&k==0)
//             continue;
//         Idxpoint nxt(x+i,y+j,z+k);
//         if(exp[nxt.x][nxt.y][nxt.z])
//             continue;
//         if(!vis[nxt.x][nxt.y][nxt.z])
//         {
//             bool flag = false;
//             for(int bbb=0;bbb<27;bbb++)
//             {  
//                 int ii = bbb/9;
//                 int jj = bbb/3-3*ii;
//                 int kk = bbb-9*ii-3*jj;
//                 ii -= 1;
//                 jj -= 1;
//                 kk -= 1;
//                 int cos = (-i)*ii + (-j)*jj + (-k)*kk;
//                 if(cos<=0)
//                     continue;
//                 int nx = x+i+ii;
//                 int ny = y+j+jj;
//                 int nz = z+k+kk;
//                 if(!incircle[nx+1][ny+1][nz+1])
//                     continue;
//                 if(!incircle[nx+ii+1][ny+jj+1][nz+kk+1])
//                     continue;
//                 if(!vis[nx][ny][nz]||val[nx][ny][nz]*center<0)
//                     continue;
//                 if(vis[nx+ii][ny+jj][nz+kk]&&val[nx+ii][ny+jj][nz+kk]*center<0)
//                 {
//                     flag = true;
//                     break;
//                 }
//             }
//             if(flag)
//                 continue;
//         }
//         if(vis[nxt.x][nxt.y][nxt.z])
//         {
//             #pragma omp critical
//             {
//                 candidates2.insert(nxt);
//             }
//         } 
//         else
//         {
//             #pragma omp critical
//             {
//                 candidates1.insert(nxt);
//             }
//         }
//     }
// }

// void Surface::surrending_check(Idxpoint idx, int sur)
// {
//     int x = idx.x;
//     int y = idx.y;
//     int z = idx.z;
//     float center = this->val[x][y][z];
//     int i = sur/9;
//     int j = sur/3-3*i;
//     int k = sur-9*i-3*j;
//     i -= 1;
//     j -= 1;
//     k -= 1;
//     if(!incircle[x+i+1][y+j+1][z+k+1])
//         return;
//     if(i==0&&j==0&&k==0)
//         return;
//     Idxpoint nxt(x+i,y+j,z+k);
//     if(exp[nxt.x][nxt.y][nxt.z])
//         return;
//     if(!vis[nxt.x][nxt.y][nxt.z])
//     {
//         bool flag = false;
//         for(int ii=-1;ii<2;ii++)
//         for(int jj=-1;jj<2;jj++)
//         for(int kk=-1;kk<2;kk++)
//         {   
//             int cos = (-i)*ii + (-j)*jj + (-k)*kk;
//             if(cos<=0)
//                 continue;
//             int nx = x+i+ii;
//             int ny = y+j+jj;
//             int nz = z+k+kk;
//             if(!incircle[nx+1][ny+1][nz+1])
//                 continue;
//             if(!incircle[nx+ii+1][ny+jj+1][nz+kk+1])
//                 continue;
//             if(!vis[nx][ny][nz]||val[nx][ny][nz]*center<0)
//                 continue;
//             if(vis[nx+ii][ny+jj][nz+kk]&&val[nx+ii][ny+jj][nz+kk]*center<0)
//             {
//                 flag = true;
//                 break;
//             }
//         }
//         if(flag)
//             return;
//     }
//     if(vis[nxt.x][nxt.y][nxt.z])
//     {
//         #pragma omp critical
//         {
//             candidates2.insert(nxt);
//         }
//     } 
//     else
//     {
//         #pragma omp critical
//         {
//             candidates1.insert(nxt);
//         }
//     }
//     return;
// }

bool Surface::surrending_check2(Idxpoint idx, int sur)
{
    int x = idx.x;
    int y = idx.y;
    int z = idx.z;
    float center = this->val[x][y][z];
    int i = sur/9;
    int j = sur/3-3*i;
    int k = sur-9*i-3*j;
    i -= 1;
    j -= 1;
    k -= 1;
    if(!incircle[x+i+1][y+j+1][z+k+1])
        return false;
    if(i==0&&j==0&&k==0)
        return false;
    Idxpoint nxt(x+i,y+j,z+k);
    if(exp[nxt.x][nxt.y][nxt.z])
        return false;
    if(!vis[nxt.x][nxt.y][nxt.z])
    {
        bool flag = false;
        for(int ii=-1;ii<2;ii++)
        for(int jj=-1;jj<2;jj++)
        for(int kk=-1;kk<2;kk++)
        {   
            int cos = (-i)*ii + (-j)*jj + (-k)*kk;
            if(cos<=0)
                continue;
            int nx = x+i+ii;
            int ny = y+j+jj;
            int nz = z+k+kk;
            if(!incircle[nx+1][ny+1][nz+1])
                continue;
            if(!incircle[nx+ii+1][ny+jj+1][nz+kk+1])
                continue;
            if(!vis[nx][ny][nz]||val[nx][ny][nz]*center<0)
                continue;
            if(vis[nx+ii][ny+jj][nz+kk]&&val[nx+ii][ny+jj][nz+kk]*center<0)
            {
                flag = true;
                break;
            }
        }
        if(flag)
            return false;
    }
    return true;
}