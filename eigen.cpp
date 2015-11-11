#include<string.h>
#include<math.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <string>
#include<stdio.h>
#include<CL/opencl.h>
#include<math.h>
#include<sys/time.h>
#include<time.h>
#include<ctime>
#include<assert.h>
#include<sys/types.h>
int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char*  str;

    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);

        str = new char[size+1];
        if(!str)
        {
            f.close();
            return 1;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
    
        s = str;
        delete[] str;
        return 0;
    }
    printf("Error: Failed to open file %s\n", filename);
    return 1;
}
float *srcA,*srcC,*dst;
int *srcB,*srcD;
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmDevSrcA;               // OpenCL device source buffer A
cl_mem cmDevSrcB;
cl_mem cmDevSrcC;
cl_mem cmDevSrcD;               // OpenCL device source buffer B 
cl_mem cmDevDst; 
cl_event event;               // OpenCL device destination buffer 
size_t globalworksize;        // 1D var for Total # of work items
size_t localworksize;		// 1D var for # of work items in the work group	
size_t szParmDataBytes;		// Byte size of context information
size_t szKernelLength;		// Byte size of kernel code
cl_int ciErr1;		// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation
void Cleanup (int argc, char **argv, int iExitCode);
//profiling info
long long kernel_execution=0;
long long kernel_queuing=0;
long long kernel_submission=0;
long long launch_overhead=0;
long long data_transfer=0;
long long CPU_time=0;
long long GPU_time=0;
long long start,end;
long long getTime();

using namespace std;
int main(int argc, char **argv)
{
    const char * filename  = "MatrixMul.cl";
    std::string  sourceStr; 
    ciErr1 = convertToString(filename, sourceStr);
    if(ciErr1 != CL_SUCCESS) {
        printf("Error in convertToString, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        return EXIT_FAILURE;
    }

    const char * source    = sourceStr.c_str();
    size_t sourceSize[]    = { strlen(source) };  
//=========================================================================================
     ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got platform\n");

//=========================================================================================
	ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got device\n");
//=========================================================================================
     cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got context\n");
//=========================================================================================
	cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErr1);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got commandqueue\n");
//=========================================================================================
	 cpProgram = clCreateProgramWithSource(cxGPUContext, 1, &source, sourceSize, &ciErr1);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got createprogramwithsource\n");
//=========================================================================================
	ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clBuildProgram, Line %u in file %s !!! Error code = %d\n\n", __LINE__, __FILE__, ciErr1);
        size_t length;
        char buffer[2048];
        clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        printf("--- Build log ---\n%s\n", buffer);
        std::cout << "--- Build log ---\n" << buffer << std::endl;
        Cleanup(argc, argv, EXIT_FAILURE);
       }
        printf("*** Got buildprogram\n");	
//=========================================================================================
	ckKernel = clCreateKernel(cpProgram, "MatrixMul", &ciErr1);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got createkernel\n");
//=========================================================================================


	FILE *fnodes;
	ifstream fin("nodes.txt");
	if(fin.is_open()!=1){
		printf("Cant open nodes1.file");;
		return 0;
	}
int M=0;
string a;
getline(fin,a);
M=atoi(a.c_str());
fin.close();
fnodes=fopen("adj_list.txt","r");
if(fnodes==NULL){
	printf("error in opening the adjacency list file");
		return 0;
}
int **matrix=new int*[M];
for(int x=0;x<M;x++){
	matrix[x]=new int[M];
	for(int k=0;k<M;k++){
		matrix[x][k]=0;
	}
}
int *nonzeros;
nonzeros=new int[M];
int count=0;
int max=0;
int d=0;
fseek( fnodes, 0L, SEEK_SET);
for(int i=0;i<M;i++){
	fscanf( fnodes, "%*d: %d", &d);
	
	while(d!=-1){
		matrix[i][d]=1;
		count+=1;
		fscanf( fnodes, "%d", &d);
	}
	if(max<count)
		max=count;
	nonzeros[i]=count;
	count=0;
}
fclose(fnodes);
float **sparsematrix=new float*[M];
for(int x=0;x<M;x++){
	sparsematrix[x]=new float[M];
	for(int k=0;k<M;k++){
		sparsematrix[x][k]=0.0;
	}
}
for(int i=0;i<M;i++){
	for(int k=0;k<M;k++){
		if(matrix[k][i]==1){
			sparsematrix[i][k]=(1/(float)nonzeros[k]);
			
		}
	}
}
/*float **mat=new float*[M];
for(int x=0;x<M;x++){
	mat[x]=new float[M];
	for(int k=0;k<M;k++){
		mat[x][k]=0.0;
	}
}
float bd=0.85;
float **E=new float*[M];
for(int x=0;x<M;x++){
	E[x]=new float[M];
	for(int k=0;k<M;k++){
		E[x][k]=(1/(float)M);
	}
}
for(int i=0;i<M;i++){
	for(int j=0;j<M;j++){
		mat[i][j]=(bd*sparsematrix[i][j])+(0.15*E[i][j]);
		
	}
}
	
for(int i=0;i<M;i++){
	count=0;
	for(int j=0;j<M;j++){
		if(mat[i][j]!=0.0){
			count+=1;
		}
	}
	if(max<count)
		max=count;
	nonzeros[i]=count;
	
}
for(int i=0;i<M;i++){
	for(int j=0;j<M;j++){
	if(mat[i][j]==0)
		printf("%f ",mat[i][j]);}
	printf("\n");
}*/
int **entries=new int*[M];
for(int x=0;x<M;x++){
	entries[x]=new int[max];
	for(int k=0;k<max;k++){
		entries[x][k]=0;
	}
}
float **mat1=new float*[M];
for(int x=0;x<M;x++){
	mat1[x]=new float[max];
	for(int k=0;k<max;k++){
		mat1[x][k]=0.0;
	}
}

for(int x=0;x<M;x++){
	int a=0;
	for(int k=0;k<M;k++){
		if(sparsematrix[x][k]!=0){
			mat1[x][a]=sparsematrix[x][k];
			entries[x][a]=k;
			a=a+1;
		}
			
	}
}

float yrdiff=0.0;
float Ep=1e-5;
float *R;
R=new float[M];
for(int i=0;i<M;i++)
	R[i]=1/(float)M;
float* y=(float*)malloc(sizeof(float)*M);
for(int i=0;i<M;i++)
	y[i]=0.0;
	//parallel code for page rank:
    //=========================================================================================	
	float diff=0.0,ysum,rsum;
	unsigned int _sizeA=M*max;
	unsigned int _sizeB=M*max;
	unsigned int _sizeC=M*1;
	int size=M/256;
	if(M%256){
		++size;
	}
	globalworksize=size*256;
	localworksize=256;
	srcA=(float *)malloc(sizeof(float)*_sizeA);   //mat1
	srcB=(int *)malloc(sizeof(int)*_sizeB);		//entries
	srcC=(float *)malloc(sizeof(float)*_sizeC);	//R
	srcD=(int *)malloc(sizeof(int)*_sizeC);
	dst=(float *)malloc(sizeof(float)*_sizeC);	//y
	for(int i=0;i<M;i++)
		for(int j=0;j<max;j++)
			srcA[i+j*M]=mat1[i][j];
			
	for(int i=0;i<M;i++)
		for(int j=0;j<max;j++)
			srcB[i+j*M]=entries[i][j];
	for(int i=0;i<M;i++)
			srcC[i]=R[i];
	for(int i=0;i<M;i++)
			srcD[i]=nonzeros[i];
	for(int i=0;i<M;i++)
		dst[i]=0.0;
	
    //=========================================================================================			
	cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * _sizeA,NULL, &ciErr1);
    cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_int) * _sizeB, NULL, &ciErr1);

	cmDevSrcC=clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * _sizeC, NULL, &ciErr1);
	
 	cmDevSrcD=clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_int) * _sizeC, NULL, &ciErr1);

	cmDevDst = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float) * _sizeC, NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS) {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got createbuffer\n");
    //=========================================================================================
    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA);
    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmDevSrcB);
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmDevSrcC);
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&cmDevSrcD);
    ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void*)&cmDevDst);
	ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_int), (void*)&M);
   ciErr1 |= clSetKernelArg(ckKernel,6,sizeof(cl_int), (void*)&max); 
   if (ciErr1 != CL_SUCCESS) {
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
    printf("*** Got setkernelarg\n");

    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * _sizeA, srcA, 0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_int) * _sizeB, srcB, 0, NULL, NULL);
ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcD, CL_FALSE, 0, sizeof(cl_int) * _sizeC,srcD, 0, NULL, NULL);

//=========================================================================================

do{
        ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcC, CL_FALSE, 0, sizeof(cl_float) * _sizeC, srcC, 0, NULL,NULL);
	ciErr1=clFinish(cqCommandQueue);  
      if (ciErr1 != CL_SUCCESS) {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }

  
    printf("*** Got enqueuewritebuffer\n");
//=========================================================================================
	start=getTime();
	ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel,1 , NULL, &globalworksize, &localworksize, 0, NULL, &event);   
	ciErr1=clFinish(cqCommandQueue);
	end=getTime();
	 if (ciErr1 != CL_SUCCESS) {
        	printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        	Cleanup(argc, argv, EXIT_FAILURE);
    		}

    	printf("*** Got enqueuendrangekernel\n");
	long s,e,q,sub;
	ciErr1 = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &s, NULL);

	ciErr1 = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &e, NULL);
	kernel_execution+=static_cast<long long>(e-s)/1000;  //kernel execution time
//=========================================================================================
         ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(float)* _sizeC, dst, 0, NULL, NULL);
ciErr1=clFinish(cqCommandQueue);  
  if (ciErr1 != CL_SUCCESS) {
        printf("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(argc, argv, EXIT_FAILURE);
    }
   
	ysum=0.0;
	rsum=0.0;
	yrdiff=0.0;
	for(int i=0;i<M;i++){
		ysum+=dst[i];
		rsum+=srcC[i];
	}
	diff=ysum-rsum;
	for(int i=0;i<M;i++)
		dst[i]=dst[i]+(diff*(1/(float)M));
	for(int i=0;i<M;i++)
		yrdiff+=(dst[i]-srcC[i]);
	for(int i=0;i<M;i++)
		srcC[i]=dst[i];
	GPU_time+=(end-start);
}while(yrdiff<Ep);
//serial code version
printf("Serial code for page rank calculation:");

do{
	float acc=0.0;
	yrdiff=0.0;
	int b;
	start=getTime();
	for(int i=0;i<M;i++){
		b=srcD[i];
		y[i]=0.0;	
		for(int k=0;k<b;k++){
			
				y[i]+=srcA[i+k*M]*R[srcB[i+k*M]];
		}
		y[i]=y[i]*0.85;
	}
	end=getTime();
	ysum=0.0;
	rsum=0.0;
	for(int i=0;i<M;i++){
		ysum+=y[i];
		rsum+=R[i];
	}
	diff=ysum-rsum;
	for(int i=0;i<M;i++)
		y[i]=y[i]+(diff*(1/(float)M));	
	for(int i=0;i<M;i++)
		yrdiff+=(y[i]-R[i]);
	for(int i=0;i<M;i++)
		R[i]=y[i];
	CPU_time+=(end-start);
}while(yrdiff<Ep);

bool bMatch=true;
for (int i=0; i<M ;i++){
   if(dst[i] !=y[i]){
	bMatch=false;
	printf(" particular %d rank doesnt match with values %f vs %f",i,dst[111],y[111]);
	break;
    }
}
printf("\n\n got check match");
Cleanup(argc,argv,(bMatch==true)? EXIT_SUCCESS : EXIT_FAILURE);


//for(int i=0;i<M;i++)
//	printf("rank of page %d is %lf\n",i+1,y[i]);
	
printf("maximum value of the row %d",max);
cout<<"CPU time serial in micro seconds	"<<CPU_time<<endl;
cout<<"GPU_time in microsecond"<<GPU_time<<endl;
cout<<"kernel execution time"<<kernel_execution<<endl;
for(int i=0;i<M;i++){
	delete[] sparsematrix[i];
	delete[] matrix[i];
}
for(int i=0;i<max;i++){
  delete[] mat1[i];
  delete[] entries[i];
}
delete[] mat1;
delete[] entries;
delete [] sparsematrix;
delete [] matrix;
delete[] y;
delete [] R;
delete[] nonzeros;

/*for(int i=0;i<max;i++){
	//delete[] mat[i];
	delete[] entries[i];
}
//delete [] mat;
delete [] entries;
*/

cout<<"press any key to exit";
return 0;
}
long long getTime(){  //time measured in microseconds
	struct timeval time;
	int err=gettimeofday(&time,NULL);
	return static_cast<long long>(time.tv_sec * 1000000 + time.tv_usec);
} 
void Cleanup (int argc, char **argv, int iExitCode)
{
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
    if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cmDevSrcA)clReleaseMemObject(cmDevSrcA);
    if(cmDevSrcB)clReleaseMemObject(cmDevSrcB);
    if(cmDevSrcC)clReleaseMemObject(cmDevSrcC);
    if(cmDevSrcD)clReleaseMemObject(cmDevSrcD);
    if(cmDevDst)clReleaseMemObject(cmDevDst);

    free(srcA); 
    free(srcB);
    free(srcC);
    free(srcD);
    free(dst);
   

    if(iExitCode == EXIT_SUCCESS)
      printf("\n******* PASSed\n");
    else
      printf("\n******* FAILed\n");
}
