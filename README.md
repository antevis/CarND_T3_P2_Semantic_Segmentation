## Udacity's Self-Driving Car Engineer Nanodegree
### Term 3, Project 2
### Semantic Segmentation

### Labelling the drivable area of a road in images using a Fully Convolutional Network (FCN).

The implementation is extensively based the **Project walkthrough** given at classes.
I've added some functionality to save and reuse the trained model with the best `loss`, though
it prevents the `train_nn` method from passing the tests, so it is commented out.
I've also added the video processing functionality. These are two sample results:

[Belorussian-Lithuanian border queue of trucks](https://youtu.be/tOUDx5okQ7c)

[Random drive through Moscow traffic in September 2012](https://youtu.be/nInh6jpBchU)

I've tested image horizontal flipping and random brightness adjustment as the **data augmentation techniques**, but it didn't seem to yield any visible improvement.

To run the project:
```
python main.py
```
