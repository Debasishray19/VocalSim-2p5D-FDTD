# 2.5D FDTD Wave Solver
:sunflower:<b>Project Details:</b>
<br><br> The 2.5D Finite-Difference Time-Domian (FDTD) wave solver blends the computational 
efficiency of low-dimensional models with the accuracy of 3D approaches tailored for simulating 
acoustic tube geometries similar to vocal tracts and wind instruments. The existing 1D and 
2D solvers can only simulate cylindrical tubes, whereas the 2.5D model can approximate wave
propagation in mirror-symmetric tube geometries.

:triangular_flag_on_post:<b>External Dependencies:</b>
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - For parallelization of the FDTD wavesolver.
* [Vcpkg](https://vcpkg.io/en/) - A free C++ package manager.
* [Chocolatey](https://chocolatey.org/) - A windows package manager.
* [Boost libraries](https://www.boost.org/) - Powerful open source libraries for C++ programming.
* [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page) - A C++ template library for linear algebra.

<b>Note</b>: Watch the following [YouTube](https://www.youtube.com/watch?v=gsLIUtmTs8Q&t=60s) video (Time: 1.42) to install 
<b>Vcpkg</b> and <b>boost libraries</b>. You can install <b>Eigen</b> library using the <b>Chocolatey</b> package manager.

:rocket: <b>Simulation:</b>
* Create a <b>build</b> folder inside the project folder.
* Run the following commands:
  * cmake .. -G "Visual Studio 17 2022" -A x64
  * cmake --build . --config Release --verbose


:bookmark: <b>References:</b>
<br><br>If you use the code for your research work, please cite the following papers:

[1] <a href ="https://www.isca-speech.org/archive/interspeech_2019/mohapatra19_interspeech.html">"An Extended Two-Dimensional Vocal Tract Model for Fast Acoustic Simulation of Single-Axis Symmetric Three-Dimensional Tubes"</a>  by Mohapatra et al.
```
@inproceedings{mohapatra19_interspeech,
  author={Debasish Ray Mohapatra and Victor Zappi and Sidney Fels},
  title={{An Extended Two-Dimensional Vocal Tract Model for Fast Acoustic Simulation of Single-Axis Symmetric Three-Dimensional Tubes}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={3760--3764},
  doi={10.21437/Interspeech.2019-1764}
}
```

<br>[2] <a href ="https://www.isca-archive.org/interspeech_2024/mohapatra24b_interspeech.html">"2.5D Vocal Tract Modeling: Bridging Low-Dimensional Efficiency with 3D Accuracy"</a>  by Mohapatra et al.
```
@inproceedings{mohapatra24b_interspeech,
  title     = {2.5D Vocal Tract Modeling: Bridging Low-Dimensional Efficiency with 3D Accuracy},
  author    = {Debasish Ray Mohapatra and Victor Zappi and Sidney Fels},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {17--21},
  doi       = {10.21437/Interspeech.2024-1749},
  issn      = {2958-1796},
}
```
