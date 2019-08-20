# CUDA-GL-RayTracer
A CUDA accelerated ray tracer using OpenGL to present the final output.

The code uses Peter Shirley's <a href="http://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html">Ray Tracing in one weekend code<a> as a base for getting the rendered spheres. I also used Roger Allen's article <a href="https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/">Accelerated Ray Tracing in One Weekend in CUDA <a> as a reference for using CUDA to translate the original CPU based implementation.

I used CUDA to speed up the process of tracing rays for each pixel which is a task that could be parallelized to improve performance for a rendered frame. OpenGL was used to create a texture and send it to the GPU for read/write via OpenGL and CUDA interop. 

