# Neural SDF Approximation

## An attempt at making a ray marching based renderer

![Render1](/images/suzanne1.png)
![Render2](/images/suzanne2.png)

The main idea of this project is approximating the signed distance function of any arbitrary object using a small neural network and then rendering the object using raymarching.

This project borrows a lot from janivanecky's [nsdf](https://github.com/janivanecky/nsdf) project, specifically the neural network architecture used to approximate the SDFs.

## Usage (Needs a lot of cleanup)

Generate a neural network model by running the train.py script from nsdf.

`python train.py $YOUR_MESH_FILE --output $OUTPUT_MODEL_FILE --model_size {small, normal, bigly}`

Then export the weights by running the ExportWeights jupyter notebook (make sure to change the model path to the output model file from the train script output)

Change the path to the weights file in RayMarching.cs script in the Unity project.

## Performance

The project currently runs at 30-50 FPS on my old rx 580 but you may experience FPS drops when you close up on the model.

## Next steps
The current neural network architecture seems to be too heavy to be run once per pixel in the compute shader. The next steps would be experimenting with subdividing the model into smaller simple models that require a simpler neural network (probably needs to be coupled with an acceleration structure such as octrees)
