  // OpenCL Kernel Function for element by element Matrix addition
__kernel void MatrixMul(__global const float* srcA,__global const int* srcB,__global const float* srcC,__global const int* srcD,__global float* dst,const int M,const int max1)
{

	int tid=get_global_id(0);
	if(tid<M){
		int i,col,perrownonzero;
		float svalue=0.0,value;	
		perrownonzero=srcD[tid];
		for(i=0 ;i<perrownonzero;i++){	
			 value=srcA[i*M+tid];
			 col=srcB[i*M+tid];
			svalue+=value*srcC[col];
		}
		dst[tid]=0.85*svalue;
	}
}
