#include "../include/ShapeAlignment.h"


__global__
void findShapeDiameter(ShapeCoordinates *shape_coordinates, 
                       ShapeCoordinates *point1, 
                       ShapeCoordinates *point2, 
                       float *diameter, 
                       int num_coordinates);

ShapeAlignment::ShapeAlignment(vector<ShapeCoordinates>& shape_coordinates):aligned_shape(shape_coordinates){

    num_coordinates = int(aligned_shape.size());

    // Resize thrust vectors
    d_shape_coordinates.resize(num_coordinates);

    // Copy shape coordinates to the device and host vectors
    //thrust::copy(shape_coordinates.begin(), shape_coordinates.end(), d_shape_coordinates.begin());
    d_shape_coordinates = aligned_shape;

    // Find the coordinates corresponding to the shape diameter
    shapeDiameter();

    // Rotate the shape about the origin to align the shape diameter along the y-axis
    rotateShapeAboutOrigin();
}


void ShapeAlignment::shapeDiameter(){

    int NUM_THREADS = num_coordinates;
    int NUM_THREADS_PER_BLOCK = 32;
    int NUM_BLOCKS = (NUM_THREADS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    d_point1.resize(num_coordinates);
    d_point2.resize(num_coordinates);
    d_diameter.resize(num_coordinates);

    // Launch the kernel to find the shape diameter
    findShapeDiameter<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(d_shape_coordinates.data()), 
                                                             thrust::raw_pointer_cast(d_point1.data()), 
                                                             thrust::raw_pointer_cast(d_point2.data()), 
                                                             thrust::raw_pointer_cast(d_diameter.data()), 
                                                             num_coordinates);

    cudaDeviceSynchronize();

    // Copy the data back to the host vectors
    h_point1 = d_point1;
    h_point2 = d_point2;
    h_diameter = d_diameter;

    // Find the maximum diameter and its corresponding index
    float max_diameter = 0.0f;
    int max_index = 0;
    for (int idx = 0; idx < num_coordinates; idx++) {
        if (h_diameter[idx] > max_diameter) {
            max_diameter = h_diameter[idx];
            max_index = idx;
        }
    }

    // Find the coordinates corresponding to the maximum diameter
    point1 = h_point1[max_index];
    point2 = h_point2[max_index];

    // Save the segment diameter
    segment_diameter = max_diameter;

    // Print the coordinates of the points corresponding to the maximum diameter
    // cout << "Point 1: (" << point1.yVal << ", " << point1.zVal << ")" << endl;
    // cout << "Point 2: (" << point2.yVal << ", " << point2.zVal << ")" << endl;
}

void ShapeAlignment::rotateShapeAboutOrigin(){

    // Calculate slope of the max diameter
    float dx = point2.yVal - point1.yVal;
    float dy = point2.zVal - point1.zVal;
    float theta = std::atan2(dy, dx);

    // build the clockwise rotation matrix by -theta:
    //   [ cos(-θ)  -sin(-θ) ]
    //   [ sin(-θ)   cos(-θ) ]
    // note cos(-θ)=cosθ, sin(-θ)=-sinθ

    float c = cos(-1*theta);
    float s = sin(-1*theta);

    for (auto &pt : aligned_shape){

        float y = pt.yVal;
        float z = pt.zVal;

        pt.yVal = c * y - s * z;
        pt.zVal = s * y + c * z;
    }

    // shift contour coordinates into 1st quadrant + 0.05 margin
    float minY = aligned_shape[0].yVal;
    float minZ = aligned_shape[0].zVal;

    for (auto &pt : aligned_shape) {
        minY = min(minY, pt.yVal);
        minZ = min(minZ, pt.zVal);
    }

    float shiftY = (minY < 0.0f ? -minY : 0.0f) + 0.05f;
    float shiftZ = (minZ < 0.0f ? -minZ : 0.0f) + 0.05f;

    for (auto &pt : aligned_shape) {
        pt.yVal += shiftY;
        pt.zVal += shiftZ;
    }
}


// Kernel to find the coordinates corresponding to the shape diameter
__global__
void findShapeDiameter(ShapeCoordinates *shape_coordinates, 
                       ShapeCoordinates *point1, 
                       ShapeCoordinates *point2, 
                       float *diameter, 
                       int num_coordinates){
    
    // Calculate the global thread ID                    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_coordinates) {

        // Initialize the diameter to 0
        float max_diameter = 0.0f;

        // Loop through all the coordinates to find the maximum distance
        for (int idx = 0; idx < num_coordinates; idx++) {
            
            float dy = shape_coordinates[tid].yVal - shape_coordinates[idx].yVal;
            float dz = shape_coordinates[tid].zVal - shape_coordinates[idx].zVal;
            float dist = sqrtf(dy * dy + dz * dz);

            if (dist > max_diameter) {
                max_diameter = dist;
                point1[tid] = shape_coordinates[tid];
                point2[tid] = shape_coordinates[idx];
            }
        }

        // Store the maximum diameter
        diameter[tid] = max_diameter;
    }
}
