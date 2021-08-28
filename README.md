# parking_cv
Parking cv is a machine learning project that uses an Xception DNN to classify images extracted from a parking lot camera. 

## How to use it (With existing .xml files)
First we need to traverse all the folders inside the PKLot database (https://web.inf.ufpr.br/vri/databases/parking-lot-database/) by using the
```traverse_and_segment('route_to_root')```. 

This will extract all of the parking spaces inside each of the pictures.
Once finished we will need to train the neural network on the pictures we have just extracted (723790 parking spots), we do so by issuing the
following command:
```
train.start()
```

This will take hours or minutes depending on your hardware (Training on an 12 thread, i7 8750H took 2 days), when finished it will generate a file
named 'model3.h5'. 

We need to spawn a parking instance by using the coordinates of an existing parking lot:
```
p1 = Parking("PKLot/PKLot/UFPR05/Rainy/2013-03-13/2013-03-13_13_05_08.xml")
```
And then update its parking spots with a new picture of the same parking lot:
```
 p1.update_state_from_photo("PKLot/PKLot/UFPR05/Sunny/2013-03-12/2013-03-12_08_40_03.jpg")
 p1.draw_boxes()
 cv2.waitKey(0)
