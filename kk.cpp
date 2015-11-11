#include<string.h>
#include<iostream>
#include <fstream>
#include <stdlib.h>
#include<stdio.h>
#include <string>
using namespace std;
int main(int argc, char *argv[]){
  string line; 
ifstream myfile("nodes1.txt");
// if(myfile.is_open()){
	getline(myfile,line);
//	int M=atoi(line.c_str());
// 	printf("%d",M);
	FILE *fnodes;
	int d;
	fnodes=fopen("adj_list1.txt","r");
	if(fnodes==NULL) printf("adj list file not opened");
	while(!feof(fnodes))
	{
		fscanf(fnodes,"%*d: %d",&d);
		printf("%d ",d);
		while(d!=-1){
			fscanf(fnodes,"%d",&d);
			printf("%d ",d);}
		printf("\n");
	} 
	myfile.close();
return 0;	
}
