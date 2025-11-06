#include "kernel_cuda.hpp"

__global__ void kernel_insert_source(const float val, float *qp, float *qc, int offset)
{

    qp[offset]+=val;
    qc[offset]+=val;
}

void insert_source(const float val, float *qp, float *qc, int offset) {
    dim3 threadsPerBlock(1, 1);
    dim3 numBlocks(1,1);
    kernel_insert_source<<<numBlocks, threadsPerBlock>>>(val, qp, qc, offset);
}

__global__ void kernel_propagate(const int sx, const int sy, const int sz, const int bord,
				 const float dx, const float dy, const float dz, const float dt,
				 const int it,
				 const float * const restrict dev_ch1dxx,
				 const float * const restrict dev_ch1dyy,
				 float * restrict dev_ch1dzz, float * restrict dev_ch1dxy,
				 float * restrict dev_ch1dyz, float * restrict dev_ch1dxz,
				 float * restrict dev_v2px,   float * restrict dev_v2pz,
				 float * restrict dev_v2sz,   float * restrict dev_v2pn,
				 float * restrict dev_pp,     float * restrict dev_pc,
				 float * restrict dev_qp,     float * restrict dev_qc)
{
    const int ix=blockIdx.x * blockDim.x + threadIdx.x;
    const int iy=blockIdx.y * blockDim.y + threadIdx.y;


    const int strideX=ind(1,0,0)-ind(0,0,0);
    const int strideY=ind(0,1,0)-ind(0,0,0);
    const int strideZ=ind(0,0,1)-ind(0,0,0);

    const float dxxinv=1.0f/(dx*dx);
    const float dyyinv=1.0f/(dy*dy);
    const float dzzinv=1.0f/(dz*dz);
    const float dxyinv=1.0f/(dx*dy);
    const float dxzinv=1.0f/(dx*dz);
    const float dyzinv=1.0f/(dy*dz);

    for (int iz=bord+1; iz<sz-bord-1; iz++) {
        const int i=ind(ix,iy,iz);

        // p derivatives, H1(p) and H2(p)

        const float pxx= Der2(dev_pc, i, strideX, dxxinv);
        const float pyy= Der2(dev_pc, i, strideY, dyyinv);
        const float pzz= Der2(dev_pc, i, strideZ, dzzinv);
        const float pxy= DerCross(dev_pc, i, strideX, strideY, dxyinv);
        const float pyz= DerCross(dev_pc, i, strideY, strideZ, dyzinv);
        const float pxz= DerCross(dev_pc, i, strideX, strideZ, dxzinv);

        const float cpxx=dev_ch1dxx[i]*pxx;
        const float cpyy=dev_ch1dyy[i]*pyy;
        const float cpzz=dev_ch1dzz[i]*pzz;
        const float cpxy=dev_ch1dxy[i]*pxy;
        const float cpxz=dev_ch1dxz[i]*pxz;
        const float cpyz=dev_ch1dyz[i]*pyz;
        const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
        const float h2p=pxx+pyy+pzz-h1p;

        // q derivatives, H1(q) and H2(q)

        const float qxx= Der2(dev_qc, i, strideX, dxxinv);
        const float qyy= Der2(dev_qc, i, strideY, dyyinv);
        const float qzz= Der2(dev_qc, i, strideZ, dzzinv);
        const float qxy= DerCross(dev_qc, i, strideX,  strideY, dxyinv);
        const float qyz= DerCross(dev_qc, i, strideY,  strideZ, dyzinv);
        const float qxz= DerCross(dev_qc, i, strideX,  strideZ, dxzinv);

        const float cqxx=devch1dxx[i]*qxx;
        const float cqyy=devch1dyy[i]*qyy;
        const float cqzz=devch1dzz[i]*qzz;
        const float cqxy=devch1dxy[i]*qxy;
        const float cqxz=devch1dxz[i]*qxz;
        const float cqyz=devch1dyz[i]*qyz;
        const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
        const float h2q=qxx+qyy+qzz-h1q;

        // p-q derivatives, H1(p-q) and H2(p-q)

        const float h1pmq=h1p-h1q;
        const float h2pmq=h2p-h2q;

        // rhs of p and q equations

        const float rhsp=dev_v2px[i]*h2p + dev_v2pz[i]*h1q + dev_v2sz[i]*h1pmq;
        const float rhsq=dev_v2pn[i]*h2p + dev_v2pz[i]*h1q - dev_v2sz[i]*h2pmq;

        // new p and q

        dev_pp[i]=2.0f*dev_pc[i] - dev_pp[i] + rhsp*dt*dt;
        dev_qp[i]=2.0f*dev_qc[i] - dev_qp[i] + rhsq*dt*dt;
    }

}

void propagate(const float val, float *qp, float *qc, int offset) {

    dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
    dim3 numBlocks(sx/threadsPerBlock.x, sy/threadsPerBlock.y);

    kernel_Propagate<<<numBlocks, threadsPerBlock>>> (  sx,   sy,   sz,   bord,
					      dx,   dy,   dz,   dt,   it,
					      dev_ch1dxx,  dev_ch1dyy,  dev_ch1dzz,
					      dev_ch1dxy,  dev_ch1dyz,  dev_ch1dxz,
					      dev_v2px,  dev_v2pz,  dev_v2sz,  dev_v2pn,
					      dev_pp,  dev_pc,  dev_qp,  dev_qc);

    CUDA_SwapArrays(&dev_pp, &dev_pc, &dev_qp, &dev_qc);
    cudaDeviceSynchronize();
}

void CUDA_SwapArrays(float **pp, float **pc, float **qp, float **qc) {
  float *tmp;

  tmp=*pp;
  *pp=*pc;
  *pc=tmp;

  tmp=*qp;
  *qp=*qc;
  *qc=tmp;
}
