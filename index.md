# CS 4641 Project Midterm Report
 <img src = "https://github.com/Aaronwork1205/Machine_learning/blob/gh-pages/assets/css/info.jpg?raw=true"> 

## Introduction and background 

### Vehicle plate recognition system: 

The vehicle plate recognition system is used to recognize the vehicle plate number and store the information for parking charging.
### Background: 
The vehicle plate number recognition system is still limited and has low efficiency when handling cases, which includes but is not limited to event parking, regular permit parking illegal parking. This incapability often will incur the congestion of the main traffic route and increase the risk of traffic incidents. In order to improve the correctness of plate recognition and thus improve the vehicle throughput, we need a better model to handle the plate recognition and save unit labor cost.

## Methodology

 The dataset we will utilize is based on images of vehicle plates. To handle the influence of a variety of plates, the influence of different weather, lighting conditions, and the angle between the camera and the plate objects, at the very first step, we include 433 images of plates under different lighting conditions. 400 of these images will be served as the training dataset, and 33 of these images will be served as the test part. 
 
We will consider each image as each data point and the features are the RGB color space and XY coordinates. Also, we will consider the testing part of the neural network as the target, which is also the goal that the model is aiming for. The target also includes the raw image data processing, which contains the processing of raw environments and angles from which the images were taken. Basically, the images were treated the same way as the training data, which is to be binarized and tiled to the same angle for the neural network to identify the plate number. The images include the plates with backgrounds like parking lots or traffic scenes.  We might also need to include more image datasets in the future if needed to encounter emerging challenges. The dataset is obtained through roboflow.com and Kaggle.com for academic purposes. Each image in the datasets has already been preprocessed, zipped and cropped to similar image size and training efficiency. 

Due to the nature of our project, we didn’t need to clean up, standardize the data since all images are all RGB pictures with the 3 dimensions (length, width, and RGB value). We also confirmed with Dr. Valente to make sure that it is acceptable to not include feature engineering or selection for our project.

 Before we start preprocessing the data, we have performed Principal Component Analysis which is a linear dimensionality reduction technique to compress our image for the purpose of rapid iterations and computer processing. We utilized the PCA methods from sklearn library. After the proper library was imported, we first split the image into three channels (Blue, Green, and Red) and then normalized the data in each set by dividing all data of all channels by 255. Then, we fit and transform the data by only considering 50 dimensions for PCA. Then, we checked how much variance was accounted for after the reduction by calling the method explained_variance_ratio_. Then, we reconstructed the image by reverse transforming the data (which is a process of taking information lost data points from the reduced space and taking it back to the original space) and then merged all three channels into one to obtain a colorful reduced image.

 
After the dimensional reduction, we converted our reduced image to a grayscale image and did binarization to the data , which abandons the RGB color space and only keeps the XY coordinates and obtained a simpler two dimensional array that contains only whiute and black pixels, with a certain threshold value. Then, we composed an array (N by 2) from the binary image of which N represents the number of black pixels and 2 is the x and y coordinates.

We had a few approaches to segmented the image to identify the vehicle plates area using different methods of unsupervised learning. We tried directly performing GMM on the black and white image. We also tried using a label method from skimage.measure library to first identify all the logos in the image. Then, we performed kmeans and set k as 2 to obtain two clusters - a dark background cluster and one where the license plate is located. Then, we find the coordinate of the region by finding row and col from the resulting array where the value is bigger than at least 10 (since the background will be equal to 0. Then, we found the location of the licence plate and we crop the image using the x y coordinates. However, the result from these two methods were consistent with every pictures; therefore, we performed the third option of unsupervised learning, which is the Density-based spatial clustering.

We first Run DBSCAN on the data points (xy coordinates of the black pixels) with eps equals to 1.6, and min number to be 5. Since points in plate area should be closely packed with no gap between adjacent points, given any point on the plate there would be 9 adjacent points and they are all within 1.414 in distance, because euclidean distance is used in this algorithm. Then, we select the clusters that satisfy criterias like a certain length to width ratio, percentage of points to total data points, and percentage of area occupied by the points, because most of the target clusters have very similar values for these parameters. After this step, the raw result from DBSCAN can narrow down to just a few clusters and Beta-CV measure is implemented to identify the cluster most likely to be the target. Eventually, we calculate the rectangle position where the plate cluster is and find the region on the original image. 




## Results

Our project focused on car plate recognition, which dataset consists of pixels of a given image. Before preprocessing the image, we run PCA on several images. We found that most pictures work best with a dimension of 50 as the parameter for PCA. The average variance that was accounted for after PCA is around 98.6% percent which is relatively high. With the compressed image, our processing speed will dramatically increase when it comes to future training since each channel’s dimension is decreased to 50.

<img src = "https://github.com/Aaronwork1205/Machine_learning/blob/gh-pages/assets/css/1.png?raw=true">


<img src = "https://github.com/Aaronwork1205/Machine_learning/blob/gh-pages/assets/css/2.png?raw=true">

Due to this dataset's nature, each data point's features are RGB and XY coordinates. Therefore, feature selection is not applicable in our problem space since we must transform our features to binary values (black and white) and XY coordinates using preprocessing methods. To decrease the number of datapoints in each image, we extracted only the black pixels and obtained the following image.

<img src = "https://github.com/Aaronwork1205/Machine_learning/blob/gh-pages/assets/css/3.png?raw=true">

The unsupervised learning methods we tried to perform on the dataset are K-means, GMM, and DBSCAN. For some of the car images, GMM clustering was able to produce few clusters , and one of the images has the possibility to contain the license plate. For some other car images, Kmeans method is really good at predicting a clear area of the location of licence plate; however, it doesn’t work on most of the pictures. By comparing the results, we can see that the DBSCAN method yields the best result. That is because K-means and GMM algorithms rely on calculating the distance between each data point and the center of each cluster. However, due to the irregular shapes resulting from preprocessing, the messiness of the positioning of clusters, and the uncertainty of the number of clusters, K-means and GMM algorithms did not yield satisfactory results for most of the picture. Sometimes K-means and GMM will produce weird clustering results, and it highly depends on pre-selecting the best number of clusters as well as the quality of the picture.

<img src = "https://github.com/Aaronwork1205/Machine_learning/blob/gh-pages/assets/css/4.png?raw=true">

<img src = "https://github.com/Aaronwork1205/Machine_learning/blob/gh-pages/assets/css/5.png?raw=true">

However, since the DBSCAN algorithm relies on grouping the data points right next to each other, it works well on our problem. As we can see from the plot, DBSCAN algotithm will produce multiple uncentern number of clusters, however, one of them will be the number plate. The black pixels of the number plate (blank places with no number present) are all connected. As a result, the DBSCAN algorithm is capable of grouping the adjacent black pixels, and one of the clusters is the cluster of our plate. However, DBSCAN heavily relies on preprocessing and number of thresholds, making sure that each number plate will be a clear and solid quadrilateral, not dilated to make the boundaries unclear. With every aspect considered, DBSCAN is the best-performing algorithm.

<img src = "https://github.com/Aaronwork1205/Machine_learning/blob/gh-pages/assets/css/6.png?raw=true">




## Discussion

### Feature selection, engineering, and dimensionality reduction

Since our database only contains images, which itself is a dataset composed by m by n by 3 array. Due to the nature of our project, we didn’t need to use feature selection or engineering for the current step as we only have two features (length and height) after we eliminated the 3rd dimension (rgb value). We confirmed with the Dr. Valente and was told that we were allowed to skip feature selection and engineering for this project. 

For dimensionality reduction, we decided to use Principal Component Analysis(PCA) to transform the values in each blue, green, and red set into a smaller k=50 number of correlated variables while keeping as much variability in the original data as possible. In this case, we used PCA to compress our image by minimizing the size in bytes of an image. As we can see from the result section, the compressed image isn’t as clear as the original image; however, it preserved most of the shape and variation, and we were able to transform the compressed image to grayscale and further black and white picture. 

As we decrease the n_components we put in the PCA, we can see that the picture becomes more and more compressed and blurry. After many testings, we found out that a value of 50 is the best for most datasets. We were able to achieve around on average 98.5% of the variance in the data with only 50 components. Moreover, we preserved the size of the original image even though we reduced the dimension individually for each channel to 50 from its original size (around 500 on average).

With the implementation of PCA, the computer will be able to process the reduced image much faster. 

 
### Unsupervised learning methods (density estimation, clustering, etc)

Since the input data points are the pixels from the binary image, the data points are distributed to capture the shapes from the original image, such as the front/rear bumper, headlights, and windshields, as well as the license plate. All other areas are primarily white and will be ignored by the algorithm. Most preprocessed images contain the area of the license plate that is isolated from other subjects, in other words, there is a recognizable boundary around the plate. 

For evaluating the resulting clusters, serval characteristics are looked at. The length and width ratio is one of the characteristics of interest because the cluster of the plate usually has the ratio within a certain range. The target cluster contains significantly more points than other clusters, suggesting by results from 10 images. Comparing the percentages of points in the cluster will help shrink all raw results into a few clusters that likely contain the target. To further narrow down the clusters, Beta-CV measures are used to compute how each cluster is isolated from the others. As mentioned above, the target cluster, the plate, has a clear boundary to all its nearby shapes or clusters. Also, the target is primarily black and has a higher density than all other clusters. Based on the two distinctions, the Beta-CV measure will mostly likely correctly identify the target cluster as it computes the ratio of mean intracluster distance to the mean intercluster distance. 

By comparing the accuracy of the final results by GMM and DBSCAN, DBSCAN is clearly the method with the best performance. The plate’s data points appear to be a rectangle or trapezoid on the binary image and are located next to each other with no gap in between. In most cases, the GMM algorithm tries to cluster the data points in elliptic shapes, because the nature of GMM is to find and fit data points onto gaussian distributions in the same dimensional space as the features. Therefore, the GMM algorithm cannot accurately recognize shapes like rectangles or trapezoids and it usually includes much more excessive and unnecessary points outside the plate or split the plate into multiple clusters in other cases. On the other hand, DBSCAN is more effective on the data points. The result generated by DBSCAN mostly consists of clusters of shapes that could be recognized by eye and found on the original image. Unlike GMM splitting a shape into different clusters, DBSCAN includes an entire shape into one cluster. DBSCAN connects points on a density and distance basis so that it can cluster points together that are closely located which makes it able to detect arbitrarily-shaped clusters and robust to noise. Therefore, the rectangle-like plate can be easily identified by DBSCAN. 

## Future

For the next step, we might also incorporate pure plates for the supervised learning part to increase the detection accuracy further. In the future, the process will be divided into two parts to guarantee the performance of the recognition model, which are the training part and the test part. In the training dataset, we will train the model based on the label for each plate image. The test dataset mainly serves as the evaluation of the performance for the successful recognition rate. The new feature we will include in this project will be the involvement of deep learning technology. In the past decades, the vehicle plate recognition system is mainly based on the preprocessing of the captured plate images and identifying each character based on the character database. However, with the help of the deep learning method and enormous datasets, the process efficiency and correctness will be dramatically improved. The potential risk for this method includes failing to find a hyper parameter set to make the time cost in a reasonable range, and failing to improve the overall recognition rate based on the limited training and testing datasets at hand. It also might increase the data search and fetching budget to meet these accuracy criteria. A reasonable budget will include the usage of GPU with high performance and time searching for useful datasets. The basic estimate time for each epoch varies based on the size of each batch of datasets, and the configuration of each hyper parameter. It can range from several minutes to several hours to complete this task.

