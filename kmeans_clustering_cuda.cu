#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

__device__ inline float euclid_dist_2(int tid, int numObjects, int numAttributes, const float* __restrict__ attributes,
		int clusterId, const float* __restrict__ clusters){
    float ans=0.0;
    for(int i = 0; i < numAttributes; i++){
		float diff = attributes[tid + i*numObjects] - clusters[i + clusterId*numAttributes];
		ans += diff*diff;
	}

    return ans;
}


__device__ inline int find_nearest_point(int tid, \
		int numObjects,
		int     numAttributes,
		const float* __restrict__ attributess,
                       const float* __restrict__ centers,         
                       int     ncenters)
{
    int index = 0;
    float min_dist = FLT_MAX;

    for (int i = 0; i < ncenters; i++) {
        float dist;
        dist = euclid_dist_2(tid, numObjects, numAttributes, attributess, i, centers);  
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }

    return(index);
}

__global__ void findNewClusterIndex(int numObjects, int numAttributes, const float* __restrict__ attributes, \
		int numClusters, const float* __restrict__ clusters, int* __restrict__ membership, \
		int* __restrict__ new_centers_len, float* __restrict__ new_centers, float* __restrict__ delta){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if(tid < numObjects){
		/* find the index of nestest cluster centers */
		int index = find_nearest_point(tid,  numObjects, numAttributes, attributes, clusters, numClusters);
		/* if membership changes, increase delta by 1 */
		if (membership[tid] != index) atomicAdd(delta, 1.0f);

		/* assign the membership to object i */
		membership[tid] = index;

		/* update new cluster centers : sum of objects located within */
		atomicAdd(new_centers_len+index, 1);
	}
}

__global__ void updateNewCluster(int numObjects, int numAttributes, const float* __restrict__ attributes, \
		int numClusters, const float* __restrict__ clusters, int* __restrict__ membership, \
		int* __restrict__ new_centers_len, float* __restrict__ new_centers, float* __restrict__ delta){

	extern __shared__ float s[];

	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	for(int i = threadIdx.x; i < numClusters*numAttributes; i += blockDim.x){
		s[i] = 0.0f;
	}

	__syncthreads();
	
	if(tid < numObjects){
		int index = membership[tid];
		for(int j = 0; j < numAttributes; j++){ 
			atomicAdd(s+index*numAttributes+j, attributes[tid + numObjects*j]);
		}
	}

	__syncthreads();

	for(int i = 0; i < numClusters; i++){
		for(int j = threadIdx.x; j < numAttributes; j += blockDim.x){ 
			atomicAdd(new_centers+i*numAttributes+j, s[j + numAttributes*i]);
		}
	}
}

__global__ void updateCenter(int numClusters, int numAttributes, float* __restrict__ clusters,\
		int* __restrict__ new_centers_len, float* __restrict__ new_centers){
	/* replace old cluster centers with new_centers */
	for(int i = blockIdx.x; i < numClusters; i += gridDim.x) {
		for(int j = threadIdx.x; j < numAttributes; j += blockDim.x) {
			if (new_centers_len[i] > 0)
				clusters[i*numAttributes + j] = new_centers[i*numAttributes + j] / new_centers_len[i];
			new_centers[i*numAttributes + j] = 0.0;   /* set back to 0 */
		}
		new_centers_len[i] = 0;   /* set back to 0 */
	}
}

/*----< kmeans_clustering() >---------------------------------------------*/
void kmeans_clustering(int     numObjects,
                         int     numAttributes,
						 float *attributes,    /* in: [numObjects][numAttributes] */
						  int    *membership,
                          int     numClusters,
						  float*  clusters,
                          float   threshold){
	
	int     *d_new_centers_len; /* [numClusters]: no. of points in each cluster */
	cudaMalloc((void**)&d_new_centers_len, numClusters*sizeof(int));

    float  *d_new_centers;     /* [numClusters][numAttributes] */
	cudaMalloc((void**)&d_new_centers, numClusters*numAttributes*sizeof(int));

	float    *d_delta;
	cudaMalloc((void**)&d_delta, sizeof(float));

	float *d_attributes;
	cudaMalloc((void**)&d_attributes, numObjects*numAttributes*sizeof(float));
	cudaMemcpy(d_attributes, attributes, numObjects*numAttributes*sizeof(float), cudaMemcpyDefault);

	int *d_membership;
	cudaMalloc((void**)&d_membership, numObjects*sizeof(int));
	cudaMemcpy(d_membership, membership, numObjects*sizeof(int), cudaMemcpyDefault);

	float *d_clusters;
	cudaMalloc((void**)&d_clusters, numClusters*numAttributes*sizeof(float));
	cudaMemcpy(d_clusters, clusters, numClusters*numAttributes*sizeof(float), cudaMemcpyDefault);

	float delta = 0.0;
	
	do {
		cudaMemset(d_new_centers_len, 0, numClusters*sizeof(int));
		cudaMemset(d_new_centers, 0, numClusters*numAttributes*sizeof(int));
		cudaMemset(d_delta, 0, sizeof(int));

		int blockSize = 256;
		int gridSize = (numObjects+blockSize-1)/blockSize;
		findNewClusterIndex<<<gridSize, blockSize>>>(numObjects, numAttributes, d_attributes, numClusters, d_clusters, d_membership, d_new_centers_len, d_new_centers, d_delta);

		updateNewCluster<<<gridSize, blockSize, numClusters*numAttributes*sizeof(float)>>>(numObjects, numAttributes, d_attributes, numClusters, d_clusters, d_membership, d_new_centers_len, d_new_centers, d_delta);
		updateCenter<<<numClusters, blockSize>>>(numClusters, numAttributes, d_clusters, d_new_centers_len, d_new_centers);

		//delta /= numObjects;
		cudaMemcpy(&delta, d_delta, sizeof(float), cudaMemcpyDefault);

//		printf("%.3f %.3f\n", delta, threshold);
	} while (delta > threshold);

	cudaMemcpy(clusters, d_clusters, numClusters*numAttributes*sizeof(float), cudaMemcpyDefault);

	cudaFree(d_new_centers_len);
	cudaFree(d_new_centers);
	cudaFree(d_delta);
	cudaFree(d_attributes);
	cudaFree(d_membership);
	cudaFree(d_clusters);
}

