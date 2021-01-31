# Recognition of bees and types of cells on the frame <!-- omit in toc -->

## Index <!-- omit in toc -->

- [Acknowledgment](#acknowledgment)
- [Introduction](#introduction)
- [Possibile approaches](#possibile-approaches)
  - [Recognition by color](#recognition-by-color)
  - [Recognition by histogram](#recognition-by-histogram)
  - [Clustering](#clustering)
- [Implementation](#implementation)
  - [Frame recognition](#frame-recognition)
    - [Manual frame selection](#manual-frame-selection)
    - [Automatic frame recognition](#automatic-frame-recognition)
  - [Classes discrimination](#classes-discrimination)
- [Results](#results)
  - [Frame recognition](#frame-recognition-1)
  - [Color mean](#color-mean)
  - [Histogram comparison](#histogram-comparison)
  - [K-Means clustering](#k-means-clustering)
  - [Region growing](#region-growing)
  - [Color means vs histogram comparison](#color-means-vs-histogram-comparison)

<div class="page-break"></div>

# Acknowledgment

This is a project for the course: **Signal, Image and Video** of the University of Trento.

Professors:
- Francesco De Natale
- Andrea Rosani

Students:
- Giovanni Lorenzini
- Simone Luchetta
- Diego Planchenstainer

<div class="page-break"></div>

# Introduction

![Beehive](/images/beehive.png)

For beekeeping, assessing the health status of a honeybee colony is very important.
There can be many methods in order to collect this kind of information and each of those has various levels of accuracy; among them, there is a simple and fast procedure for retrieving the health indicator which is based on looking at the sixths parts of the whole frame.

Essentially, the surface of the frame is divided in sixths; then, one should attribute to each sixth what its associated predominant characteristic is.
For this process the beekeeper classifies each sixth to one amongst the following  classes:
- Bees;
- Capped brood (close cells);
- Uncapped brood (open cells).

Sometimes it is useful to consider also half of a sixth for better precision.
Once this process is completed, one can then assess the health status of the hive, based on the amount of bees, close cells and open cells.
Then by looking at a specific table it is possible to retrieve a health indicator.

Amongst all species, it is the “Apis mellifera ligustica” the one who is the most widely distributed in Italy.
These bees can have strains that have different colors: most of them are yellow in color, while some others are very pale yellow. Nonetheless, each of those has a tendency to darken as the bee brood is carried out over time.
As a matter of fact, older generations of bee colonies have indeed a lower capacity of producing wax and they are associated to having a browner color.

In order to create a colony, bees must produce what is called the beeswax.
The hive workers collect and use it to form cells for honey storage and larval or pupal protection within the beehive.
The beeswax is a complex chemically compound made of fatty acids, alcohol chains, hydrocarbons and other substances.
The beeswax is essential in order to ensure a correct life cycle for all the units that populate the nest itself.
Over the time, the beeswax becomes darker due to the droppings of the larvae, and some other residues that might be left behind, which may cause severe problems: beeswax could be damaged by the wax moth, or residues could lead to the growth of smaller bees, not to mention the fact that a bad beeswax could make honey crystallize faster and therefore bees may not get through the winter.

The majority of the examples in this document are based on this image, keep it as a reference.

![Sixths numeration](/images/sixths-numeration.png)  
*Adopted numeration of sixths*

<div class="page-break"></div>

# Possibile approaches

It is possible to classify the content of the frame basing on various methodologies.
Herein are proposed three main paths that lead to the classification of each part of a frame.
Before presenting each method, we wanted to give a general overview on how it’s possible to classify a subdivision of the frame.

<div class="page-break"></div>

## Recognition by color

The underlying idea for this approach is to select an area of variable dimension that is associated with a matching class.
Within this area it is possible to compute the sample mean.
Then, we also compute the sample mean for each part of the frame.

Once we have these means measurements, we exploit these information by computing the mean squared error between the means of the selected samples and the means of the sixths.
In this way it is possible to associate what is mostly dominant in each of the subdivisions of the frame by taking the lowest value of the means squared error.

<div class="page-break"></div>

## Recognition by histogram

Another similar strategy is to select a sample of the classes of interests, and extract their related histograms.
Then, for each of the sixths it could be possible to compare the histogram concerning the class with the ones regarding the sixths areas. The sixth that reaches the closest correlation will then be associated to that particular class.

The measure of correlation will be determined by computing the histogram correlations between the one of the classes and the one associated to each sixth.
The subdivision that gets an higher correlation with the corresponding class histogram will be then associated to that particular class.
For further information see the [opencv documentation](https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html).

In this example by looking at the histograms can be noticed that:
- In sixth<sub>0,0</sub>, the top left one, there is a predominance of close cells;
- In sixth<sub>2,1</sub>, the bottom right one, there is a predominance of bees.

![Histogram Bee Close Open](/images/histogram-bee-close-open.png)  
*Histograms of a bee, close and open areas respectively*

![Histogram S00 S21](/images/histogram-S-0-0-S-2-1.png)  
*Histograms of the sixth<sub>0,0</sub> and of the sixth<sub>2,1</sub>*

<div class="page-break"></div>

## Clustering

The simplest approach to clustering is the k-means technique.
It assumes that you decide at priori the number of clusters (k).
Each example (or each cluster) will be represented by its mean μ<sub>i</sub>.
Basically you want examples to be associated with the cluster with the closest mean.

With K-means the intrinsic idea is that we treat the whole image with unsupervised learning (as we of course have no label for each of the classes).
Thus, it will be up to the algorithm to find whether certain types of examples are more similar to a particular cluster rather than another one.

Once the process is complete, the result should be able to classify correctly the three main classes, them being bees, opened cells and closed cells.
It is important to reiterate that the program does not know which class is which, so it is possible that one class will have some classes inside it (for example capped brood and the stock of honey can be seen as "class 3", even if the program does not know what "class 3" is).

<div class="page-break"></div>

# Implementation

The program that has been implemented is subdivided into two principal parts.
In the first one the recognition of the frame is performed, either in automatic mode or in manual mode and the image is geometrically corrected.
The aim of the second one is to discriminate between the three classes present in the image.

<div class="page-break"></div>

## Frame recognition

This step consists in cutting out the non relevant areas that are not needed in order to compute the measurements.
The whole process is aimed at identifying the contours of the bee frame so that we could discard the background content, so to say.

### Manual frame selection

For the manual recognition of the frame the user has to select four corners of the frame. Then the image is cropped and corrected geometrically.

![Manual frame selection](/images/manual-frame-selection.png)  
*Selection of four points to discriminate the corners*

### Automatic frame recognition

For the automatic recognition of the frame the user has to select a region on the frame in order to obtain the color of it.
Then the image is converted to HSV format for representative purposes.
In fact in this color space it is possible to extract the different ranges of colors in a better way.
Then the region corresponding to the selected range of colors is extracted.

```python
mask = cv2.inRange(hsvImage, (hMin, sMin, vMin), (hMax, sMax, vMax))
filtered = cv2.bitwise_and(hsvImage, hsvImage, mask=mask)
gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
```

![HSV filtered frame](/images/hsv-filtered-frame.png)  
*Image after color extraction*

Morphological operators are applied to filter the noise and connect the possible gaps between the pieces of the extracted frame.

```python
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(y / 75), round(y / 75))))
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(y/100), round(y/100))), iterations=3)
```

![Morpholohical operators frame](/images/morphological-operators-frame.png)  
*Image after morphological operations*

Afterwards, probabilistic Hough line transform is applied to extract the lines that characterize the frame.

```python
lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
```

Since the Hough lines are many, they need to be grouped by similarity and therefore a mean line is calculated for every group.
Similarity criteria are defined as follows: each of the lines must not differ from a certain angle to the other, and on top of that, their distance must not be over a certain threshold.

Hopefully we will obtain four groups, as the sides of the frame, and we can compute the intersection between the lines. These points represent the corners of the frame that can be corrected geometrically and cropped.

![Grouped Hough Lines](/images/grouped-hough-lines.png)  
*Extraction of the lines that characterize the frame*

A perspective transform is calculated from four points and then a perspective transformation is applied to the image.

Afterwards the process of class discrimination can begin.

<div class="page-break"></div>

## Classes discrimination

The operator has to identify the area corresponding to each class (with the possibility to skip one if it is not present) by clicking the left button of the mouse.
The size of the rectangle that corresponds to the selected area is adjustable using the center wheel of the mouse in order to be able to select the wider possible area for each class.
When the selection is done the corresponding areas are cropped and the relative histogram is computed.

![Class Selection](/images/class-selection.png)  
*Selection of the matching class*

The image is then cropped into six pieces as it would be traditionally done and an histogram for each sixth is computed.

The predominant class is selected by computing the correlation between the sixth histogram and the sample histograms, choosing the one with the higher value.

```python
beeHistCorrelation = cv2.compareHist(beeHistogram, meanHistogram, cv2.HISTCMP_CORREL)
openCellHistCorrelation = cv2.compareHist(openCellHistogram, meanHistogram, cv2.HISTCMP_CORREL)
closeCellHistCorrelation = cv2.compareHist(closeCellHistogram, meanHistogram, cv2.HISTCMP_CORREL)

if (beeHistCorrelation > openCellHistCorrelation and beeHistCorrelation > closeCellHistCorrelation):
  results.append("Bee")
elif (openCellHistCorrelation > beeHistCorrelation and openCellHistCorrelation > closeCellHistCorrelation):
  results.append("Open")
elif (closeCellHistCorrelation > beeHistCorrelation and closeCellHistCorrelation > openCellHistCorrelation):
  results.append("Close")
```

Then an overlay of the results is displayed assigning different colors to the classes for better visual distinction. Related counts-percentages are also shown in the terminal window.

<div class="page-break"></div>

# Results

The result of the various methodologies will be described below.
Before doing that a premise is needed. As explained before traditionally the frame is subdivided into six pieces to make manual classification easier.
Since for a program there is no substantial difference into making 6 or 100 parts we have embedded the possibility to modify the number of sections of the frame, thus their dimensions.
In the discussion below there will be some images subdivided in sixths and other in more parts in order to better analyze the behavior of the algorithms applied.

<div class="page-break"></div>

## Frame recognition

The automatic frame recognition does not work on all images, mostly due to bad image quality.
We proposed a table in which we show the effective validity of the automatic frame recognition algorithm.

| Image | Manual frame selection | Automatic frame recognition |
| :---: | :--------------------: | :-------------------------: |
|  01   |        &#10003;        |          &#10003;           |
|  02   |        &#10003;        |          &#10003;           |
|  03   |        &#10003;        |          &#10003;           |
|  04   |        &#10003;        |          &#10005;           |
|  05   |        &#10003;        |          &#10005;           |
|  06   |        &#10003;        |          &#10005;           |
|  07   |        &#10003;        |          &#10003;           |
|  08   |        &#10003;        |          &#10003;           |
|  09   |        &#10003;        |          &#10003;           |
|  10   |        &#10003;        |          &#10005;           |
|  11   |        &#10003;        |          &#10003;           |
|  12   |        &#10003;        |          &#10005;           |
|  13   |        &#10003;        |          &#10005;           |
|  14   |        &#10003;        |          &#10005;           |
|  15   |        &#10003;        |          &#10005;           |
|  16   |        &#10003;        |          &#10005;           |
|  17   |        &#10003;        |          &#10003;           |
|  18   |        &#10003;        |          &#10005;           |
|  19   |        &#10003;        |          &#10005;           |
|  20   |        &#10003;        |          &#10005;           |
|  21   |        &#10003;        |          &#10005;           |
|  22   |        &#10003;        |          &#10005;           |

<div class="page-break"></div>

## Color mean

In the first place the methodology that has been used is the one based on the exploitation of the color means of the classes.
Unfortunately, this method has not obtained enough satisfactory results.
The main problem of this approach is given by the fact that one of the mean class color could be the mean color between the other two, leading the classifier into a wrong discrimination.
This problem has been encountered in many images, in particular in the image that is present in all the document as example, where the mean color of the bees is the mean between the color of the opened cells and the closed ones.
As it can be seen in the image below the first sixth is wrongly classified as "Bee" even though it is the less predominant.

![Classification Color Mean](/images/classification-color-6.png)  
*Results of color mean method applied on sixths*

![Classification Color Mean](/images/classification-color-1000.png)  
*Results of color mean method applied on small dimension areas*

<div class="page-break"></div>

## Histogram comparison

The second methodology bases its approach on the exploitation of the histograms of the classes.
It is a more raffinate way to compare the colors between the different classes.
We consider this approach as the most efficient between the tried ones.

The main defect in this approach is the dependency on the area selected by the user.
If this area is not chosen in a proper manner the results can be disrupted.
For example, if in the selection of the open cell area we select an area with one bee over the cell, the histogram will be tricked and the classifier will be misled.

![Classification Histogram](/images/classification-histogram-6.png)  
*Results of histogram method applied on sixth*

![Classification Histogram](/images/classification-histogram-1000.png)  
*Results of histogram method applied on small dimension areas*

<div class="page-break"></div>

## K-Means clustering

Some other techniques that could be applied in order to classify the various areas consists in applying an unsupervised algorithm.
The idea here was to treat the whole pixels in the image as if they were points in a cartesian space of the colors, and then let the K-Means clustering algorithm find associations, hence clustering them into three groups (one for each of the possible corresponding class).

We tested the algorithm with various parameters for the converging criterion (number of iterations, final accuracy...), but none of the settings separated the regions successfully enough.
This algorithm gave very poor results in the end. Still, we included the file for completeness, but we found it unusable.

Here we tried the K-Means algorithm from the OpenCV framework:

```python
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 2, 1.0)
K = 3
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
```

![K-means CV2 TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER max_iter=2 epsilon=1.0](/images/kmeans-CV2-TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER-2-1.0.png)  
*K-means CV2 TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER max_iter=2 epsilon=1.0*

Here instead we used the K-Means algorithm from the SciKit framework:

```python
kmeans = KMeans(n_clusters=3, random_state=0, tol=1e-1).fit(pic_n)
```

![K-means scikit n_clusters=3 random_state=0 tol=1e-1](/images/kmeans-scikit-n_clusters=3-random_state=0-tol=1e-1.png)  
*K-means scikit n_clusters=3 random_state=0 tol=1e-1*

Things change a little bit (but are still non satisfactory) if we try to blur a bit the image before applying the algorithm.

![K-means scikit n_clusters=3 random_state=0 tol=1e-1 with blurred image](/images/kmeans-scikit-n_clusters=3-random_state=0-tol=1e-3-blurred-5-5.png)  
*K-means scikit n_clusters=3 random_state=0 tol=1e-1 with blurred image*

<div class="page-break"></div>

## Region growing 

We also tried another kind of image segmentation approach, which is the region growing. Essentially, an initial set of small areas is iteratively merged according to similarity constraints. The algorithm starts by choosing an arbitrary set of seed pixels, that will be compared with their own neighboring pixels. A region is grown from the seed pixel by adding in neighboring pixels that are similar, increasing the size of the region. This whole process is continued until all pixels belong to some region.

```shell
$ python ./RegionGrowing.py dataset/02.png 12
```

![Region Growing](/images/region-growing.png)  
*Region Growing*

```shell
$ python ./RegionGrowingSV.py dataset/02.png 12
```

![Region Growing SV](/images/region-growing-sv.png)  
*Region Growing SV*

This algorithm has proven to be not only computationally hungry, but gave very poor results in the end.

<div class="page-break"></div>

## Color means vs histogram comparison

The differences between the method that exploits the mean color and the one in which the histograms are compared can be appreciated in the images showed below.
As it can be seen the histogram method is able to better discriminate the different classes.
In these two images the size of the areas analyzed is small.
Clearly there is a limit on the minimum size of the area in which the histogram operates in an efficient way.
The fact is that the number of occurrences of a particular color can drastically drop as the area becomes tinier, leading the classifier to an uncertain decision, because it does not have enough information (think about the bees stripes).

![Classification Color Mean](/images/classification-color-200.png)  
*Results of color mean method applied on medium dimension area*

![Classification Histogram](/images/classification-histogram-200.png)  
*Results of histogram method applied on medium dimension area*

Herein is reported a table containing all the comparisons between the two main algorithms for all the images in our dataset.

<div class="page-break"></div>

<table>
  <tr>
    <th>Image</th>
    <th colspan="4">Color mean</th>
    <th colspan="4">Histogram comparison</th>
  </tr>
  <tr>
    <th>File</th>
    <th>Bees</th>
    <th>Open cells</th>
    <th>Close cells</th>
    <th>Accuracy</th>
    <th>Bees</th>
    <th>Open cells</th>
    <th>Close cells</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>01</td>
    <td>4</td>
    <td>0</td>
    <td>8</td>
    <td>67%</td>
    <td>0</td>
    <td>1</td>
    <td>11</td>
    <td>92%</td>
  </tr>
  <tr>
    <td>02</td>
    <td>7</td>
    <td>1</td>
    <td>4</td>
    <td>67%</td>
    <td>3</td>
    <td>1</td>
    <td>8</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>03</td>
    <td>5</td>
    <td>0</td>
    <td>7</td>
    <td>25%</td>
    <td>9</td>
    <td>3</td>
    <td>0</td>
    <td>83%</td>
  </tr>
  <tr>
    <td>04</td>
    <td>5</td>
    <td>3</td>
    <td>4</td>
    <td>58%</td>
    <td>5</td>
    <td>2</td>
    <td>5</td>
    <td>75%</td>
  </tr>
  <tr>
    <td>05</td>
    <td>11</td>
    <td>0</td>
    <td>1</td>
    <td>33%</td>
    <td>8</td>
    <td>4</td>
    <td>0</td>
    <td>67%</td>
  </tr>
  <tr>
    <td>06</td>
    <td>9</td>
    <td>1</td>
    <td>2</td>
    <td>42%</td>
    <td>9</td>
    <td>0</td>
    <td>3</td>
    <td>67%</td>
  </tr>
  <tr>
    <td>07</td>
    <td>11</td>
    <td>1</td>
    <td>0</td>
    <td>83%</td>
    <td>12</td>
    <td>0</td>
    <td>0</td>
    <td>92%</td>
  </tr>
  <tr>
    <td>08</td>
    <td>10</td>
    <td>1</td>
    <td>1</td>
    <td>92%</td>
    <td>11</td>
    <td>1</td>
    <td>0</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>09</td>
    <td>0</td>
    <td>11</td>
    <td>1</td>
    <td>25%</td>
    <td>3</td>
    <td>2</td>
    <td>7</td>
    <td>92%</td>
  </tr>
  <tr>
    <td>10</td>
    <td>9</td>
    <td>0</td>
    <td>3</td>
    <td>58%</td>
    <td>4</td>
    <td>0</td>
    <td>8</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>11</td>
    <td>1</td>
    <td>0</td>
    <td>11</td>
    <td>58%</td>
    <td>6</td>
    <td>0</td>
    <td>6</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>12</td>
    <td>12</td>
    <td>0</td>
    <td>0</td>
    <td>92%</td>
    <td>11</td>
    <td>0</td>
    <td>1</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>13</td>
    <td>0</td>
    <td>0</td>
    <td>12</td>
    <td>92%</td>
    <td>1</td>
    <td>0</td>
    <td>11</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>14</td>
    <td>12</td>
    <td>0</td>
    <td>0</td>
    <td>92%</td>
    <td>10</td>
    <td>2</td>
    <td>0</td>
    <td>92%</td>
  </tr>
  <tr>
    <td>15</td>
    <td>6</td>
    <td>6</td>
    <td>0</td>
    <td>92%</td>
    <td>8</td>
    <td>4</td>
    <td>0</td>
    <td>92%</td>
  </tr>
  <tr>
    <td>16</td>
    <td>11</td>
    <td>0</td>
    <td>1</td>
    <td>100%</td>
    <td>11</td>
    <td>0</td>
    <td>1</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>17</td>
    <td>12</td>
    <td>0</td>
    <td>0</td>
    <td>92%</td>
    <td>11</td>
    <td>1</td>
    <td>0</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>18</td>
    <td>9</td>
    <td>2</td>
    <td>1</td>
    <td>92%</td>
    <td>8</td>
    <td>4</td>
    <td>0</td>
    <td>92%</td>
  </tr>
  <tr>
    <td>19</td>
    <td>9</td>
    <td>0</td>
    <td>3</td>
    <td>67%</td>
    <td>6</td>
    <td>1</td>
    <td>5</td>
    <td>92%</td>
  </tr>
  <tr>
    <td>20</td>
    <td>12</td>
    <td>0</td>
    <td>0</td>
    <td>100%</td>
    <td>12</td>
    <td>0</td>
    <td>0</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>21</td>
    <td>12</td>
    <td>0</td>
    <td>0</td>
    <td>83%</td>
    <td>12</td>
    <td>0</td>
    <td>0</td>
    <td>83%</td>
  </tr>
  <tr>
    <td>22</td>
    <td>11</td>
    <td>0</td>
    <td>1</td>
    <td>83%</td>
    <td>10</td>
    <td>0</td>
    <td>2</td>
    <td>92%</td>
  </tr>
  <tr>
    <th></th>
    <th colspan="3">Accuracy mean</th>
    <th>72%</th>
    <th colspan="3"th>Accuracy mean</th>
    <th>91%</th>
  </tr>
</table>

It can be easily seen that the histogram comparison algorithm is far better than the color means algorithm.
