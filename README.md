# Image-Upscaler

## Description

I've chosen to study one of the fundamental problems of image processing: 
How can we upscale the resolution of a given image such that its visual quality 
will also increase. 

In order to conclude an efficient research, I've studied the already known and 
available interpolation methods, and tried to understand each method's 
mechanism, flaws and how to improve them. After learning in depth information 
about Nearest Neighbours, Bilinear, Bicubic and Lancoz interpolations, I managed
to craft three efficient ways that try to search and apply filters such that the
images' visuals would be more appealing. 

The first method trained a Neural Network to spot and correct the differences 
between the original image, and the resized image with a given method. 

The second method used a pipeline of multiple Neural Networks, such that each 
NN would be assigned a very specific scale factor to perfect. 

The last method suggested to learn a mathematical correction function by 
enforcing a specific mathematical template, and fine-tuning the parameters 
such that the loss would be minimal. 

On the theoretical side, I've also discussed the possibility of using Genetic 
Algorithms to enhance the results of the correction function, and the fact that 
we can always try to vectorize simple visual images such as cartoon images for 
better results, and to use image-segmentation in order to have clear objects 
which would later be given to specialized algorithms for specific colour 
pallets.

## Useful Links

Paper: 
https://www.overleaf.com/read/cwttgyvqpfjh

Presentation: 
https://docs.google.com/presentation/d/1Enc_lKh4Wt9dDzl9_d5tOG-wcq2Sxv07fDRAphH8J6E/edit?usp=sharing