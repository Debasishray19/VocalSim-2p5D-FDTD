#define _USE_MATH_DEFINES

#include "../include/ShapeGenerator.h"
#include <cmath>


// Constructor for ShapeGenerator
ShapeGenerator::ShapeGenerator(int num_tube_segments,
    std::vector<float>& s1_semi_major_axis_len, 
    std::vector<float>& s1_semi_minor_axis_len,
    std::vector<float>& s2_semi_major_axis_len, 
    std::vector<float>& s2_semi_minor_axis_len,
    std::vector<float>& s1_s2_center_distance, 
    std::vector<float>& yIntersect, 
    std::vector<float>& zIntersect,
    float segment_area):
s1_semi_major_axis_len(s1_semi_major_axis_len),
s1_semi_minor_axis_len(s1_semi_minor_axis_len),
s2_semi_major_axis_len(s2_semi_major_axis_len),
s2_semi_minor_axis_len(s2_semi_minor_axis_len),
s1_s2_center_distance(s1_s2_center_distance),
yIntersect(yIntersect),
zIntersect(zIntersect),
segment_area(segment_area){

    s1_semi_major_axis_len.resize(num_tube_segments, 0.0f);
    s1_semi_minor_axis_len.resize(num_tube_segments, 0.0f);
    s2_semi_major_axis_len.resize(num_tube_segments, 0.0f);
    s2_semi_minor_axis_len.resize(num_tube_segments, 0.0f);
    s1_s2_center_distance.resize(num_tube_segments, 0.0f);
    yIntersect.resize(num_tube_segments, 0.0f);
    zIntersect.resize(num_tube_segments, 0.0f);
}

void ShapeGenerator::combineRegularShapes(int shape_idx, int num_coordinates){
    
    // Create a vector of ShapeCoordinates to store the coordinates of the shape
    vector<ShapeCoordinates> half_contour;

    // Create an instance of Utils class to use its functions
    Utils utils;

    // Find two shapes [s1, s2] that intersect each other at two points
    // Their combined area should be equal to the segment area

    // A = Area
    // A1 = Area of s1, A2 = Area of s2
    // A1:A2 = m:1

    // (A1)x + (A2)x = (m+1)x = A
    // x  = A/(m+1)
    // A1 = m*x = m(A/(m+1))
    // A2 = 1*x = 1(A/(m+1))

    // A1+A2 > A [At start]

    // Define m

    float m = 3.0f;
    float x = segment_area / (m + 1.0f);
    float A1 = m * x;
    float A2 = 1.0f * x;

    // Set  major to minor axis length ratio for both shapes
    // Set [1/1 - Circle/Circle], [1/3 = Circle/Ellipse], [3/3 = Ellipse/Ellipse]
    float s1_axis_len_ratio = 3.0f;
    float s2_axis_len_ratio = 1.0f;

    float s1_semi_major_axis_ratio = s1_axis_len_ratio;
    float s1_semi_minor_axis_ratio = 1.0f;

    float s2_semi_major_axis_ratio = s2_axis_len_ratio;
    float s2_semi_minor_axis_ratio = 1.0f;
    
    float s1_lenx = sqrtf(A1 / (M_PI * s1_semi_major_axis_ratio * s1_semi_minor_axis_ratio));
    s1_semi_major_axis_len[shape_idx] = s1_lenx * s1_semi_major_axis_ratio;
    s1_semi_minor_axis_len[shape_idx] = s1_lenx * s1_semi_minor_axis_ratio;

    float s2_lenx = sqrtf(A2 / (M_PI * s2_semi_major_axis_ratio * s2_semi_minor_axis_ratio));
    s2_semi_major_axis_len[shape_idx] = s2_lenx * s2_semi_major_axis_ratio;
    s2_semi_minor_axis_len[shape_idx] = s2_lenx * s2_semi_minor_axis_ratio;

    // Number of coordinates to generate half of the original shape
    int num_half_coordinates = (num_coordinates + 1) / 2;

    // Resize s1_ and s2_coordinates vectors
    s1_coordinates.resize(num_half_coordinates);
    s2_coordinates.resize(num_half_coordinates);

    // Shape [s1, s2] Generation
    // Set the center of s1 to (yCenter1, zCenter1) = (0, 0)
    // Set the center of s2 to (yCenter2, zCenter2) = 
    // yCenter2 = yCenter1 + s1_semi_major_axis_len + 0.003*millimeter | zCenter2 = zCenter1

    // Determine s1 coordinates [y, z]
    float yCenter1 = 0.0f;
    float zCenter1 = 0.0f;

    float s1_starting_coord_y = yCenter1 - s1_semi_major_axis_len[shape_idx];
    float s1_ending_coord_y   = yCenter1 + s1_semi_major_axis_len[shape_idx];
    
    vector<ShapeCoordinates> s1_coordinates = generateEllipseCoordinates(s1_starting_coord_y, s1_ending_coord_y, 
                                                                         s1_semi_major_axis_len[shape_idx], s1_semi_minor_axis_len[shape_idx], 
                                                                         num_half_coordinates, yCenter1);
    
    // Determine s2 coordinates [y, z]
    float yCenter2 = yCenter1 + s1_semi_major_axis_len[shape_idx] + 0.003f * MILLIMETER;
    float zCenter2 = zCenter1;

    float s2_starting_coord_y = yCenter2 - s2_semi_major_axis_len[shape_idx];
    float s2_ending_coord_y   = yCenter2 + s2_semi_major_axis_len[shape_idx];

    vector<ShapeCoordinates> s2_coordinates = generateEllipseCoordinates(s2_starting_coord_y, s2_ending_coord_y, 
                                                                         s2_semi_major_axis_len[shape_idx], s2_semi_minor_axis_len[shape_idx], 
                                                                         num_half_coordinates, yCenter2);

    vector<ShapeCoordinates> intersection_points = utils.findShapeIntersections(s1_coordinates, s2_coordinates);
    
    half_contour = finalContourCoordinates(s1_coordinates, s2_coordinates, intersection_points,
                                           s1_semi_major_axis_len[shape_idx], s1_semi_minor_axis_len[shape_idx],
                                           s2_semi_major_axis_len[shape_idx], s2_semi_minor_axis_len[shape_idx],
                                           yCenter1, yCenter2, s1_starting_coord_y, s2_ending_coord_y, num_half_coordinates);

    // Print final Contour coordinates to a txt file
    // print_section_shape(half_contour, "contour_coordinates.txt");
    
    float generated_contour_area = 2 * utils.computeTotalArea(half_contour);
    
    // Updated the final contour coordinates to generate the desired shape as follows:

    // STEP1: Scale up s1 and s2 coordinates
    // STEP2: Generate the contour coordinates for S1 and S2
    // STEP3: Find the intersection points of the two shapes
    // STEP4: Find the final contour coordinates and compute the area [generated_contour_area]
    // STEP5: if generated_contour_area < segment_area, then repeat the process

    float scale_up_factor = 0.01f;
    float scale_up = MILLIMETER*scale_up_factor;

    while(generated_contour_area < segment_area){

        // STEP1: Scale up major and minor axis of s1 and s2

        s1_semi_major_axis_len[shape_idx] = s1_semi_major_axis_len[shape_idx] + scale_up;
        s1_semi_minor_axis_len[shape_idx] = s1_semi_minor_axis_len[shape_idx] + scale_up;

        s2_semi_major_axis_len[shape_idx] = s2_semi_major_axis_len[shape_idx] + scale_up;
        s2_semi_minor_axis_len[shape_idx] = s2_semi_minor_axis_len[shape_idx] + scale_up;

        // STEP2: Draw both shapes [s1, s2]
        float s1_starting_coord_y = yCenter1 - s1_semi_major_axis_len[shape_idx];
        float s1_ending_coord_y   = yCenter1 + s1_semi_major_axis_len[shape_idx];

        vector<ShapeCoordinates> s1_coordinates = generateEllipseCoordinates(s1_starting_coord_y, s1_ending_coord_y, 
                                                                             s1_semi_major_axis_len[shape_idx], s1_semi_minor_axis_len[shape_idx], 
                                                                             num_half_coordinates, yCenter1);

        float s2_starting_coord_y = yCenter2 - s2_semi_major_axis_len[shape_idx];
        float s2_ending_coord_y   = yCenter2 + s2_semi_major_axis_len[shape_idx];
        vector<ShapeCoordinates> s2_coordinates = generateEllipseCoordinates(s2_starting_coord_y, s2_ending_coord_y, 
                                                                             s2_semi_major_axis_len[shape_idx], s2_semi_minor_axis_len[shape_idx], 
                                                                             num_half_coordinates, yCenter2);
        // STEP3: Find the intersection points of the two shapes
        vector<ShapeCoordinates> intersection_points = utils.findShapeIntersections(s1_coordinates, s2_coordinates);

        // STEP4: Find the contour coordinates for the half-shape and compute the area [generated_contour_area]
        half_contour = finalContourCoordinates(s1_coordinates, s2_coordinates, intersection_points,
                                               s1_semi_major_axis_len[shape_idx], s1_semi_minor_axis_len[shape_idx],
                                               s2_semi_major_axis_len[shape_idx], s2_semi_minor_axis_len[shape_idx],
                                               yCenter1, yCenter2, s1_starting_coord_y, s2_ending_coord_y, num_half_coordinates);
        
        // Compute area of the generated contour
        generated_contour_area = 2 * utils.computeTotalArea(half_contour);

        // Print the area
        // std::cout << "Generated contour area: " << generated_contour_area << endl;
    }

    // Manually set the tube height [zVal] for the first and last coordinate to zero
    half_contour[0].zVal = 0.0f;
    half_contour[half_contour.size() - 1].zVal = 0.0f;

    // Generate the full shape coordinates
    full_shape_coordinates = half_contour;
    
    for(int idx = full_shape_coordinates.size()-2; idx >= 0; idx--){
        full_shape_coordinates.push_back({full_shape_coordinates[idx].yVal, -1 * full_shape_coordinates[idx].zVal});
    }

    // Set the diameter of the shape along the y-axis
    ShapeAlignment shapeAlignment(full_shape_coordinates);
    aligned_full_shape_coordinates = shapeAlignment.aligned_shape;
    
    segment_diameter = shapeAlignment.segment_diameter;

    // print_section_shape(full_shape_coordinates, "contour_coordinates.txt");
}

// Print coordinates of the shape to a file
void ShapeGenerator::print_section_shape(vector<ShapeCoordinates> shape, string file_name){

    string file_path = "../../results/" + file_name;

    ofstream out(file_path);
    if (!out) {
        std::cerr << "Error: could not open the file for writing\n";
    } else {
        // column header (optional)
        out << "# yVal    zVal\n";
        for (const auto &pt : shape) {
          out << pt.yVal  << ' '<< pt.zVal << '\n';
        }  
    }
    out.close();
}

// Generate half‐ellipse coordinates centered at yCenter
vector<ShapeCoordinates> ShapeGenerator::generateEllipseCoordinates(float starting_coord_y, float ending_coord_y, 
                                                                    float semi_major_axis_len, float semi_minor_axis_len, 
                                                                    int num_points, float yCenter){
    
    // Create an instance Utils class to use its functions
    Utils utils;

    vector<ShapeCoordinates> shape(num_points); 

    vector<float> y_coords = utils.linspace(starting_coord_y, ending_coord_y, num_points);
    vector<float> distance_from_center_sqr(num_points);

    for (int idx = 0; idx < num_points; idx++) {
        float y = y_coords[idx] - yCenter;
        distance_from_center_sqr[idx] = y * y;
    }

    // Square the major and minor axis lengths
    float semi_major_axis_len_sqr = semi_major_axis_len * semi_major_axis_len;
    float semi_minor_axis_len_sqr = semi_minor_axis_len * semi_minor_axis_len;

    // To avoid numerical error: s1_distance_from_center_sqr[0]^2 = semi_major_axis_len_sqr
    // Do the same for other indices in shape
    distance_from_center_sqr[0] = semi_major_axis_len_sqr;

    for(int idx = 0; idx < static_cast<int>(num_points/2); idx++){
        distance_from_center_sqr[num_points - idx - 1] = distance_from_center_sqr[idx];
    }

    // Populate the shape vector with y and z coordinates
    for (int i = 0; i < num_points; ++i) {
        float y = y_coords[i];
        float z = sqrtf(max(0.0f, semi_minor_axis_len_sqr * (1.0f - distance_from_center_sqr[i] / semi_major_axis_len_sqr)));

        // Assign the coordinates to the shape vector
        shape[i].yVal = y;
        shape[i].zVal = z;
    }

    return shape;

}

vector<ShapeCoordinates> ShapeGenerator::finalContourCoordinates(vector<ShapeCoordinates> s1_coordinates, vector<ShapeCoordinates> s2_coordinates, vector<ShapeCoordinates> intersection_points,
                                                                 float s1_semi_major_axis_len, float s1_semi_minor_axis_len,
                                                                 float s2_semi_major_axis_len, float s2_semi_minor_axis_len,
                                                                 float yCenter1, float yCenter2, float s1_starting_coord_y, float s2_ending_coord_y, 
                                                                 int num_half_coordinates){
    
    // number of points in each half                                                                
    int seg = num_half_coordinates - 1;
    
    // Create an instance Utils class to use its functions
    Utils utils;

    // Generate y-coordinates of two halves of the shape
    vector<float> yValShape1 = utils.linspace(s1_starting_coord_y, intersection_points[0].yVal, seg);
    vector<float> yValShape2 = utils.linspace(intersection_points[0].yVal, s2_ending_coord_y, seg);                                                                
    
    // Compute the corresponding z-coordinates for the two halves
    vector<float> zValShape1(seg);
    vector<float> zValShape2(seg);

    for (int i = 0; i < seg; ++i) {
        zValShape1[i] = sqrtf(max(0.0f, (s1_semi_minor_axis_len * s1_semi_minor_axis_len) * (1.0f - ((yValShape1[i] - yCenter1) * (yValShape1[i] - yCenter1)) / (s1_semi_major_axis_len * s1_semi_major_axis_len))));
        zValShape2[i] = sqrtf(max(0.0f, (s2_semi_minor_axis_len * s2_semi_minor_axis_len) * (1.0f - ((yValShape2[i] - yCenter2) * (yValShape2[i] - yCenter2)) / (s2_semi_major_axis_len * s2_semi_major_axis_len))));
    }

    // Splice them into finalContour so that the intersection appears exactly once
    int totalPts = 2 * num_half_coordinates - 1;
    vector<ShapeCoordinates> finalContour(totalPts);

    // first half
    int ptr_idx = 0;
    for (int idx = 0; idx < seg; ++idx, ++ptr_idx) {
        finalContour[ptr_idx].yVal = yValShape1[idx];
        finalContour[ptr_idx].zVal = zValShape1[idx];
    }

    // the shared intersection point 
    finalContour[ptr_idx].yVal = intersection_points[0].yVal;
    finalContour[ptr_idx].zVal = intersection_points[0].zVal;
    ++ptr_idx;

    // second half
    for (int idx = 0; idx < seg; ++idx, ++ptr_idx) {
        finalContour[ptr_idx].yVal = yValShape2[idx];
        finalContour[ptr_idx].zVal = zValShape2[idx];
    }

    return finalContour;
}
