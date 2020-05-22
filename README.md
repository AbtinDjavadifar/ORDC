# ORDC
Object Recognition Datasets and Challenges: A Review

### Table 1 - Early Milestone object recognition datasets

|                           Dataset                            | # of classes | # of images | Annotation type | Year |                         Description                          |
| :----------------------------------------------------------: | :-------------: | :-----: | :-------------: | :--: | :----------------------------------------------------------: |
|   [COIL-100](https://www.kaggle.com/jessicali9530/coil100)   |       100       |  7,200  | Classification  | 1996 | Single-object  images with black background – 72 poses for each object. |
| [FERET](https://www.nist.gov/programs-projects/face-recognition-technology-feret) |      1,199      | 14,126  | Classification  | 1997 | Large-Scale  face recognition dataset and testing framework. |
|           [BSDS](https://github.com/BIDS/BSDS500)            |        -        |   500   |  Segmentation   | 2001 |  Category  agnostic segmentation of natural context images.  |
| [Caltech-101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) |       102       |  9,144  |  Bounding Box   | 2003 |                101  common object categories.                |
| [LabelMe](https://github.com/wkentaro/labelme/blob/master/README.md) |       182       | 62,197  |    Polygons     | 2005 | Public  Online Annotation Tool  Polygons  instead of classification annotation. |
| [Caltech-256](https://www.kaggle.com/jessicali9530/caltech256) |       257       | 30,307  |  Bounding  Box  | 2006 |                An  extension for Caltech-101.                |
| [Tiny  Images](http://horatio.cs.nyu.edu/mit/tiny/data/index.html) |     75,062      |  80  m  | Classification  | 2009 | 32×32  images hierarchically annotated based on the Wordnet Lexical database. |


### Table 2 - Dataset statistic for PASCAL VOC, ImageNet, MS COCO, and Open Images 

|                           Dataset                            | # of Classes | # of Images | Average  Objects Per Image | First  Introduced |
| :----------------------------------------------------------: | :----------------: | :---------------: | :------------------------: | :---------------: |
|    [PASCAL  VOC](http://host.robots.ox.ac.uk/pascal/VOC/)    |         20         |      22,591       |            2.3             |       2005        |
|            [ImageNet](http://www.image-net.org/)             |       21,841       |    14,197,122     |             3              |       2009        |
|       [Microsoft  COCO](http://cocodataset.org/#home)        |         91         |      328,000      |            7.7             |       2014        |
| [Open  Images](https://opensource.google/projects/open-images-dataset) |        600         |     9,178,275     |            8.1             |       2017        |


### Table 3 - Challenge Description for PASCAL VOC, ILSVRC, MS COCO, and Open Images

|                          Challenge                           |         Tasks  Covered         | # of Classes |   # of Images       |           # of Annotated  Objects |  Years  active  |                      Task  Description                       |                      Evaluation  Metric                      |
| :----------------------------------------------------------: | :----------------------------: | :--------: | :---------------------: | -----------------------------: | :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    [PASCAL  VOC](http://host.robots.ox.ac.uk/pascal/VOC/)    |     Image  Classification      |     20     |         11,540          |                         27,450 |  2005  - 2012   | Absence/presence  prediction of at least one instance of every class in each image |                      Average  Precision                      |
|                                                              |           Detection            |     20     |         11,540          |                         27,450 |  2005  - 2012   | Bounding  box prediction for every instance of the challenge classes present in images |              Average  Precision with IoU > 0.5               |
|                                                              |          Segmentation          |     20     |          2913           |                           6929 |  2007  - 2012   |        Semantic  segmentation for the object classes         |                             IoU                              |
|                                                              |     Action  Classification     |     10     |          4588           |                           6278 |  2010  - 2012   | bounding  box prediction or single points for persons performing an action and annotate  with the corresponding action label |             AP  over action class classification             |
|                                                              |     Person  Layout Taster      |     3      |           609           |                            850 |  2007  - 2012   | Body  part (hands, head, feet) detection with bounding boxes |     AP  calculated separately for parts, with IoU > 0.5      |
|     [ILSVRC](http://www.image-net.org/challenges/LSVRC/)     |     Image  Classification      |    1000    |        1,331,167        |                      1,331,167 |  2010  - 2014   |      Classification  for one annotated class per image       |   Binary  class error over the top 5 predictions per image   |
|                                                              |      Object  Localization      |    1000    |         573,966         |                        657,231 |  2011  - 2017   |    Bounding  box detection for only one object per image     | Binary  class and bounding box IoU error over the top 5 predictions |
|                                                              |       Object  Detection        |    200     |         476,688         |                        534,309 |  2013  - 2017   |     Bounding  box prediction for all instances per image     | AP flexible recall threshold varied  proportional to bounding box size |
|                                                              |  Object  Detection from Video  |     30     | 5,314  (video snippets) |                              - |  2015  - 2017   | Continuous  bounding box prediction throughout video sequences | AP flexible recall threshold varied  proportional to bounding box size |
|       [Microsoft  COCO](http://cocodataset.org/#home)        |           Detection            |     80     |        123,000+         |                       500,000+ | 2015  - present |    Instance  Segmentation over object classes (*things*)     |                 AP  at IoU = [0.5:0.05:0.95]                 |
|                                                              |           Keypoints            |     17     |        123,000+         |                       250,000+ | 2017  - present |   Simultaneous  object detection and keypoint localization   |        AP  based on Object Keypoint Similarity (OKS)         |
|                                                              |             Stuff              |     91     |        123,000+         |                              - | 2017  – present |       Pixelwise  segmentation of background categories       |                          Mean  IoU                           |
|                                                              |            Panoptic            |    171     |        123,000+         |                       500,000+ | 2018  - present |     Full  segmentation of images (*stuff* and *things*)      |                      Panoptic  Quality                       |
|                                                              |           DensePose            |     -      |         39,000          |                         56,000 |  2019-present   | Human  body segmentation and mapping all the pixels of the body to a template 3D  model |         AP  based on Geodesic Point Similarity (GPS)         |
| [Open Images](https://storage.googleapis.com/openimages/web/challenge2019.html) |       Object  Detection        |    500     |        1,743,042        |                     12,421,955 | 2018  - present |          Hierarchical-based  bounding box detection          |                             mAP                              |
|                                                              |     Instance  Segmentation     |    300     |       ~  848,000        |                      2,148,896 | 2018  - present | Instance  Segmentation over object classes, negative labels included to refine training |                       mAP  at IoU>0.5                        |
|                                                              | Visual  Relationship Detection |     57     |        1,743,042        | 380,000  relationship triplets | 2018  - present | Labeling  images with relationship triplets containing the interacting objects and the  action class | A  weighted sum of mAP and recall of number of relationships at IoU>0.5 |


### Table 4 – Generic object detection datasets 

|                           Dataset                            | # of Images | # of Classes | # of Bounding  Boxes | Year |
| :----------------------------------------------------------: | :-------: | :-----: | :---------------: | :--: |
| [Caltech  101 ](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) |   9,144   |   102   |       9144        | 2003 |
|                          MIT  CSAIL                          |   2,500   |   21    |       2500        | 2004 |
| [Caltech  256](https://www.kaggle.com/jessicali9530/caltech256) |  30,307   |   257   |      30,307       | 2006 |
|          [Visual Genome](https://visualgenome.org/)          |  108,000  | 76,340  |     4,102,818     | 2016 |
|    [YouTube BB](https://research.google.com/youtube-bb/)     |  5.6  m   |   23    |      5.6  m       | 2017 |
|   [Objects 365](https://www.objects365.org/overview.html)    |  638,000  |   365   |      10.1  m      | 2019 |


### Table 5 – Object Segmentation datasets

|                           Dataset                            | # of Images | # of Classes | # of Objects | Year | Challenge |                         Description                          |
| :----------------------------------------------------------: | :-------------: | :-----: | :--------: | :--: | :-------: | :----------------------------------------------------------: |
|       [SUN](https://groups.csail.mit.edu/vision/SUN/)        |     130,519     |  3819   |  313,884   | 2010 |    No     | The  main purpose of the dataset is scene recognition, however instance-level  segmentation masks have also been provided |
| [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) |     10,000      |   20    |   20,000   | 2011 |    No     | Object  contours on the train/validation images of PASCAL VOC |
| [Pascal Part](http://roozbehm.info/pascal-parts/pascal-parts.html) |     11,540      |   191   |   27,450   | 2014 |    No     | Object  part segmentations for all the 20 class in the PASCAL VOC dataset |
|             [DAVIS](https://davischallenge.org/)             |  150  (videos)  |    4    |    449     | 2016 |    Yes    | A  video object segmentation dataset and challenge focused on semi-supervised  and unsupervised segmentation tasks |
|       [YouTube-VOS](https://youtube-vos.org/dataset/)        | 4,453  (videos) |   94    |   7,755    | 2018 |    Yes    | videos  object segmentation dataset collected of short (3s-6s) video snippets |
|             [LVIS](https://www.lvisdataset.org/)             |     164,000     |  1000   |    2  m    | 2019 |    Yes    | Instance  segmentation annotations for a long-tail of classes with few samples |
| [LabelMe](http://labelme.csail.mit.edu/Release3.0/browserTools/php/dataset.php) |     62,197      |   182   |  250,250   | 2005 |    No     | Instance-level  segmentations, some of the background classes have also been annotated |


### Table 6 – Popular scene recognition datasets

|                           Dataset                            | # of Images | # of Classes | Additional  Annotations | Year | Description                                                  |
| :----------------------------------------------------------: | :-------: | :-----: | :------------------------------: | ---- | ------------------------------------------------------------ |
| [15-Scene](https://www.kaggle.com/zaiyankhan/15scene-dataset) |   4,485   |   15    |                -                 | 2006 | One  of the earliest major scene classification datasets     |
| [MIT Indoor67](https://www.kaggle.com/naveenkavuru/mit-indoor-67) |  15,620   |   67    |                -                 | 2009 | Indoor  scene classification in 5 main groups: Store, Home, Public Space, Leisure,  and Working Place |
|       [SUN](https://groups.csail.mit.edu/vision/SUN/)        |  130,519  |   899   |      313,844  SM (Objects)       | 2010 | Classification  dataset of navigable scenes with additional object recognition annotations |
| [SUN Attribute](http://cs.brown.edu/~gmpatter/sunattributes.html) |  14,000   |   700   | 102  binary attributes per image | 2012 | attribute-based  representation of scenes for a subset of the original SUN database |
|     [Open Surfaces](http://opensurfaces.cs.cornell.edu/)     |  25,357   |   160   |     71,460  SM   (Surfaces)      | 2013 | Segmented  surfaces in interior scenes with texture and material information |
|           [Places2](http://places2.csail.mit.edu/)           |   10  m   |   476   |                -                 | 2017 | Classification  of scenes bounded by spaces a human body would fit, with binary attributes |


### Table 7 – Scene parsing datasets

|                           Dataset                            | # of Images | Stuff  Classes | Object  Classes | Year | Challenge |                          Highlights                          |
| :----------------------------------------------------------: | :-------: | :------------: | :-------------: | :--: | :-------: | :----------------------------------------------------------: |
| [MSRC 21](https://www.microsoft.com/en-us/research/project/image-understanding/?from=http%3A%2F%2Fresearch.microsoft.com%2Fvision%2Fcambridge%2Frecognition%2F#!downloads) |    591    |       6        |       15        | 2006 |    No     | One of the earliest semantic scene parsing datasets, Images were later used in [71], [101] |
| [Stanford  Background](http://dags.stanford.edu/projects/scenedataset.html) |    715    |       7        |        1        | 2009 |    No     | Outdoor  scene parsing dataset collected from LabelMe, MSRC, and PASCAL VOC. Geometric  features also included |
|   [SiftFlow](https://people.csail.mit.edu/celiu/SIFTflow/)   |   2688    |       18       |       15        | 2009 |    No     | An  early dataset on outdoor environment scene parsing labeled using LabelMe |
| [Barcelona](https://www.kaggle.com/xvivancos/barcelona-data-sets) |  15,150   |       31       |       139       | 2010 |    No     |               A  subset of the LabelMe dataset               |
|   [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/)    |   1,449   |       26       |       893       | 2012 |    No     | Parsing  of 464 cluttered indoor scenes, depth maps also included. Semantic  segmentation for objects |
|    [SUN+LM](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/)    |  45,676   |       52       |       180       | 2013 |    No     | A  fully annotated subset of LabelMe and SUN datasets with both indoor and  outdoor images |
| [PASCAL  Context](https://cs.stanford.edu/~roozbeh/pascal-context/) |  10,103   |      152       |       388       | 2014 |    No     | Pixel-wise  semantic segmentation on the PASCAL VOC dataset. 520 new object and stuff  categories were added to the original dataset. |
|         [SUN  RGB-D](http://rgbd.cs.princeton.edu/)          |  10,335   |       47       |       800       | 2015 |    Yes    | Indoor  scene parsing dataset and benchmark, 3D bounding boxes also provided |
|      [Cityscapes](https://www.cityscapes-dataset.com/)       |  25,000   |       14       |       13        | 2016 |    No     | Images  captured from a vehicle driving in urban environments across 50 cities in  different weather conditions in Europe. Instance-level segmentations |
| [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) |  25,210   |     1,242      |      1,451      | 2017 |    Yes    | Includes  object part labels, and attributes. Instance-level segmentations |
|      [Synscapes](https://7dlabs.com/synscapes-overview)      |  25,000   |       14       |       13        | 2018 |    No     | Photo-realistic  synthetic scene parsing of urban environments. Annotation categories are the  same as Cityscapes. Instance-level segmentations |
|   [MS  COCO Stuff](https://github.com/nightrome/cocostuff)   |  163,957  |       91       |       80        | 2018 |    Yes    | Pixel-wise  semantic segmentation for the entire MS COCO dataset |


### Table 8 – Popular Street-view autonomous driving datasets

|                         **Dataset**                          | **Year** |    **Location**     | **Annotated  frames** |    ** # of Classes **     | **3D  Boxes** |                        **Highlights**                        |
| :----------------------------------------------------------: | :------: | :-----------------: | :-------------------: | :-------------------: | :-----------: | :----------------------------------------------------------: |
|        [KITTI](http://www.cvlibs.net/datasets/kitti/)        |   2012   | Karlsruhe,  Germany |          15k          |           8           |     200k      | Pioneer  benchmark dataset for 3D object detection, multimodal |
|      [Cityscapes](https://www.cityscapes-dataset.com/)       |   2016   |  50  cities in EU   |          25k          |          27           |       0       | annotation  richness, scene variability and complexity  Provided  with depth information with stereo image and sensors |
|  [BDD 100k](https://bair.berkeley.edu/blog/2018/05/30/bdd/)  |   2017   |       NY, SF        |         100k          | 40  Objects  8  Lanes |       0       | crowd  sourcing to reflect the diversity  LANE  classified into different layers |
|          [KAIST](https://irap.kaist.ac.kr/dataset/)          |   2018   |        Seoul        |         8.9k          |           3           |       0       | a new  type of large-scale dataset covering various time slots in drivable areas, Thermal  image sensor used as a secondary sensor |
|           [ApolloScape](http://apolloscape.auto/)            |   2018   |      4x  China      |         144k          |  25  Object28 Lanes   |      70k      | Contains  lane markings based on the lane colours and styles, Instance level  annotations are available  ,  Tricycles are also annotated |
|          [A*3D](https://github.com/I2RDL2/ASTAR-3D)          |   2019   |      Singapore      |          39k          |           7           |     230k      | Focused  on pedestrian detection  High  driving speed and low annotation speed |
|       [Argoverse](https://www.argoverse.org/data.html)       |   2019   | Miami,  Pittsburgh  |          22k          |          15           |     993k      | Focused  on 3D object tracking and motion forecasting, Annotated HD semantic maps  included |
|                      Automative  RADAR                       |   2019   |       Germany       |          500          |           7           |     3000      |     RADAR  data and object detection based on RADAR data     |
|             [H3D](https://usa.honda-ri.com/H3D)              |   2019   |         SF          |          27k          |           8           |     1.1M      | to  stimulate research on full-surround 3D multi object detection and tracking |
|            [nuScenes](https://www.nuscenes.org/)             |   2019   |     Boston,  SG     |          40k          |          23           |     1.4M      | First  dataset provided 3D dataset with attribute annotations, first to provide  RADAR data, rich multimodal information |
|        [Semantic  KITTI](http://semantic-kitti.org/)         |   2019   |      Karlsruhe      |           -           |          25           |       0       | Semantic  segmentation using multiple scans  datasets  with pointwise annotation of 3D point clouds, Sequential dataset |
|               [Waymo](https://waymo.com/open/)               |   2019   |       3x USA        |         200k          |           4           |      12M      | 15  times diverse than any available data,  First  dataset- such low-level synchronized info available, making it easier to  conduct research on LiDAR input representation other than the popular 3D  point set format |
| [Mapillary  Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g) |   2017   |       Global        |          25k          |          152          |       0       | Scene-parsing  with instance-level object segmentation with a diverse geographic, weather,  season and daytime extent |
|         [Lyft  L5](https://level5.lyft.com/dataset/)         |   2019   |     Paolo  Alto     |          46k          |           9           |     1.3 m     | Multimodal  captured by a fleet of vehicles, an annotated LiDAR semantic map is provided, |
| [D²-City](https://outreach.didichuxing.com/d2city/Overview)  |   2019   |        China        |         700k          |          12           |       -       | Sampled  from dashcam video sequences, Bounding cube annotations, Tricycles are also  annotated |

### 



### Table 9 – Pedestrian Detection Datasets. Number of images does not include unannotated images. Unique pedestrians are considered for the number of pedestrians.

|                           Dataset                            | Year |  # of Cities   | # of Images | # of Pedestrians |                          Highlights                          |
| :----------------------------------------------------------: | :--: | :--------------: | :-------: | :-----------: | :----------------------------------------------------------: |
|       [CityPersons](https://arxiv.org/abs/1702.05693)        | 2017 | 27  cities in EU |   5000    |     35016     |           Built  on top of the Cityscapes dataset            |
|    [INRIA](https://project.inria.fr/aerialimagelabeling/)    | 2005 |        -         |    614    |      902      |                  Occlusion  labels included                  |
| [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) | 2009 |        1         |  250,000  |     2300      | Temporal  correspondence and occlusion labels included, Sampled from 10 hours of video |
| [MIT  Ped.](http://cbcl.mit.edu/software-datasets/PedestrianData.html) | 2000 |        -         |   1800    |     1800      |         Labelled  using the LabelMe annotation tool          |
|       [EuroCity](https://eurocity-dataset.tudelft.nl/)       | 2018 | 31  cities in EU |  47,000   |    238,000    |        Largest  pedestrian detection dataset to date         |
|       [NightOwls](https://www.nightowls-dataset.org/)        | 2018 |        7         |    32     |    55,000     | Pedestrian  detection at night time, detailed annotations attributes: pose, occlusion,  and height |
| [Daimler](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html) | 2009 |        1         |  21,790   |    56,492     |       Occlusion  attributes provided, monocular images       |





### Table 10 – Bird’s eye view datasets

|                         **Dataset**                          | **Year** |          **Location**           |                     **Road  span/Area**                      |        **Size  of data**         |                        **Highlights**                        |
| :----------------------------------------------------------: | :------: | :-----------------------------: | :----------------------------------------------------------: | :------------------------------: | :----------------------------------------------------------: |
| [NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm) |   2005   |               USA               |                   500-640m  Span  of road                    |             90  min              | Video  cameras attached to the adjacent buildings  Speed  levels more than 75km/h are not included in the dataset  Very  less amount of truck class |
|           [HighD](https://www.highd-dataset.com/)            |   2017   |             Germany             |                      420m  Span of road                      |           16.5  hours            | Drone  based dataset with five scenario description layers, the first 3 layers  include static scenario description, 4th layer includes dynamic  description,5th layer includes environment conditions |
| [The inD  (Intersection  Drone Dataset)](https://www.ind-dataset.com/) |   2017   | 4  locations in Aachen, Germany |        Altitude 100m   80x40  meters to 140x70 meters        |  10 hours  Of  video recording   | dataset  contains more than 11500 road users including vehicles, bicyclists and  pedestrians at intersections |
|       [INTERACTION](https://interaction-dataset.com/)        |   2019   | USA,  China, Bulgaria, Germany  |                             n/a                              | 365min+  433min+  133min+  60min | Data  collected from drones and traffic cameras  Multimodal,  driving behavior |
|       [AU-AIR](https://bozcani.github.io/auairdataset)       |   2019   |        Aarhus,  Denmark         | Flight altitude (5m to 30m) and camera angle  45 to 90 degree |             2  hours             | multi-modal  sensor data (i.e., visual, time, location, altitude, IMU, velocity)   differences  between natural and aerial images in the context of object detection task |





### Table 11 - AV-related object recognition and scene understanding challenges

| **Challenge/Benchmark**                                      | **Year** | **Task**                                                     | **Dataset**                           | **Metric**                                                   | **Highlight**                                                |
| ------------------------------------------------------------ | -------- | ------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [CVPR 2018 - Video Segmentation Challenge](https://www.kaggle.com/c/cvpr-2018-autonomous-driving) | 2018     | Video Segmentation                                           | -                                     | mAP & IoU                                                    | Segmentation of movable object from video frames.            |
| [CVPR 2018 - Berkeley DeepDrive challenges](http://wad.ai/2018/challenge.html) | 2018     | Road Object Detection & Drivable Area    Segmentation & Domain adaptation | BDD 100K dataset                      | AP & IoU                                                     | Multi-tasks.                                                 |
| [nuScenecs 3D detection challenge](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) | 2019     | 3D model generation                                          | nuScenes dataset                      | mAP & TP                                                     | Generate 3D model of the environment.   Using sensor data retrieved from camera, lidar, and radar. |
| [Lyft 3D Detection for Autonomous Vehicle](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles) | 2019     | Object detection                                             | Lyft Level 5 dataset                  | IoU                                                          | 3D object detection over semantic maps.                      |
| [NightOwls Pedestrian Detection Challenge](https://www.nightowls-dataset.org/nightowls-competition-2019/) | 2019     | Pedestrian detection                                         | NightOwls dataset                     | Standard average missing rate                                | RGB pictures of pedestrians in dim environment.              |
| [D²-City Detection Domain Adaptation Challenge](https://outreach.didichuxing.com/d2city/Overview) | 2019     | Object detection & Domain adaptation                         | Image-Net & BDD 100K datasets         | AP & IoU                                                     | Transfer learning. Domain adaptation   for datasets from two different countries. |
| [WIDER Face & Person Challenge](https://wider-challenge.org/2019.html) | 2019     | Pedestrian detection                                         | WIDER dataset                         | mAP & IoU                                                    | Detection of pedestrians and cyclist in unconstrained  environment. |
| [CVPR 2019 - Beyond Single-frame Perception](http://wad.ai/2019/index.html) | 2019     | 3D object detection                                          | -                                     | mAP & IoU                                                    | Using 3D lidar scanned point clouds.   High quality dataset with different environment conditions. |
| [The KITTI 3D Object Evaluation Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) | 2017     | Object detection                                             | KITTI dataset                         | precision-recall curve & AP                                  | Dataset consists of images with their point clouds.          |
| [GM-ATCI Rear-view pedestrians dataset Benchmark](https://sites.google.com/site/rearviewpeds1/) | 2016     | Pedestrian detection                                         | GM-ATCI Rear-view pedestrians dataset | IoU                                                          | Study of position and occlusion pattern of pedestrian        |
| [Caltech Pedestrian Detection Benchmark](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) | 2012     | Pedestrian detection                                         | Caltech Pedestrian dataset            | IoU                                                          | -                                                            |
| [The KITTI 2D Object Evaluation Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) | 2012     | Object detection & Object Orientation                        | KITTI dataset                         | precision-recall curve & AP    & average orientation similarity | Objection detection from 2D RGB images                       |


### Table 12 - Medical imaging datasets

| **Dataset**                                                  | **Size**      | **Year** | **Target disease/organ** | **Content**                       | **Challenge/**  **Benchmark** | **Description**                                              |
| ------------------------------------------------------------ | ------------- | -------- | ------------------------ | --------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| [NLM's MedPix Database](https://medpix.nlm.nih.gov/home)     | 59000 images  | -        | -                        | Integrated images                 | no                            | A free online dataset contains more than 12000 patient cases |
| [STARE Database](https://cecas.clemson.edu/~ahoover/stare/)  | ~400 cases    | -        | Eye                      | retinal images                    | no                            | Blood vessel segmentation images                             |
| [SMIR](https://www.smir.ch/)                                 | 350425 images | -        | -                        | CT scans                          | yes                           | 51 subjects of whole-body postmortem CT scans                |
| [EchoNet-Dynamic](https://echonet.github.io/dynamic/)        | 10030 images  | 2020     | Heart                    | Echocardiographic video frames    | yes                           | An expert labeled dataset for the study of cardiac motion and  chamber size. |
| [Atlas of Digital Pathology](http://www.dsp.utoronto.ca/projects/ADP/) | 17668 images  | 2020     | Radiological   diagnosis | Histological patch images         | yes                           | Images of different organs with 57 types of hierarchical tissue  annotated |
| [COVID-CT Dataset](https://arxiv.org/abs/2003.13865)         | 349 images    | 2020     | COVID19                  | CT scans                          | no                            | Specifically targeting the worldwide  pandemic virus.        |
| [SARAS-ESAD Dataset](https://saras-esad.grand-challenge.org/Dataset/) | 22601 frames  | 2020     | Prostatectomy procedure  | Video frames                      | yes                           | A dataset of videos showing the full prostatectomy procedure by  surgeons |
| [The StructSeg 2019 Dataset](https://structseg2019.grand-challenge.org/Dataset/) | 120 cases     | 2019     | Radiotherapy planning    | CT scans                          | yes                           | A dataset for the treatment of cancers                       |
| [ODIR-5K](http://academictorrents.com/details/cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72) | 5000 images   | 2019     | Eye                      | fundus photographs                | yes                           | Fundus images taken by various cameras with different resolutions |
| [DRIVE](https://drive.grand-challenge.org/)                  | 400 cases     | 2019     | Eye                      | Retinal images                    | yes                           | Images of 400 different patients between 25-90 years of age. |
| [The RSNA Brain Hemorrhage CT Dataset](https://pubs.rsna.org/doi/10.1148/ryai.2020190211) | 874035 images | 2019     | Brain Hemorrhage         | CT scans                          | yes                           | Images gathered from 2 medical societies and 60  neuroradiologists |
| [The KiTs19 Challenge Dataset](https://github.com/neheller/kits19) | 300 cases     | 2019     | Kidney tumor             | CT scans                          | yes                           | A dataset of multi-phase CT imaging with segmentation masks  |
| [SegTHOR](https://arxiv.org/abs/1912.05950)                  | 60 scans      | 2019     | Lung                     | CT scans                          | No                            | A dataset focused on the segmentation of organs at risk in the  thorax |
| [The EAD Challenge Dataset](https://arxiv.org/abs/1905.03209) | 2700 images   | 2019     | Hollow organs            | Endoscopic video frames           | yes                           | Images collected from 6 different data centers               |
| [Oasis Brains Dataset](https://www.oasis-brains.org/)        | ~1000 cases   | 2019     | Brain                    | MRI & PET images                  | no                            | A dataset collected over 30 years                            |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | 224316 images | 2019     | Chest                    | Chest radiographs                 | yes                           | A dataset labeled by an automatic labeler                    |
| [LERA](https://aimi.stanford.edu/lera-lower-extremity-radiographs-2) | 182 patients  | 2019     | Musculoskeletal disorder | Radiographs                       | yes                           | Images of hip, foot, ankle and knee of patients for the study of  musculoskeletal disorders |
| [CAMEL colorectal adenoma Dataset](https://github.com/ThoroughImages/CAMEL) | 177 cases     | 2019     | Cancer                   | Histology images                  | no                            | A dataset for segmentation of cancerous parts in organ       |
| [BACH Dataset](https://iciar2018-challenge.grand-challenge.org/dataset/) | 430 images    | 2019     | Breast cancer            | Microscopy & whole-slide images   | yes                           | Microscopy images labelled by 2 experts                      |
| [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) | 1370 patients | 2018     | Knee                     | MRI                               | yes                           | A dataset for autonomous MRI diagnosis                       |
| [The REFUGE Challenge Dataset](https://refuge.grand-challenge.org/) | 1200 images   | 2018     | Glaucoma                 | Fundus photographs                | yes                           | The dataset was  collected using two types of devices.       |
| [MURA](https://stanfordmlgroup.github.io/competitions/mura/) | 40561 images  | 2018     | Bone                     | musculoskeletal radiographs       | yes                           | A manually labeled dataset by board-certificated Stanford  radiologists, containing 7 body types: finger, hand, elbow, forearm, humerus,  wrist and shoulder |
| [Calgary-Campinas Public Brain MR Dataset](https://sites.google.com/view/calgary-campinas-dataset/home) | 167 scans     | 2018     | Brain                    | MRI                               | no                            | A dataset for analysis of brain MRI                          |
| [HAM 10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | 10015 images  | 2018     | Skin lesions             | Dermatoscopic images              | yes                           | A multi-modal and multi-population dataset                   |
| [NIH Chest X-ray Dataset](https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest) | 100000 images | 2017     | Chest                    | X-ray images                      | no                            | A dataset of x-ray images                                    |
| [RESECT](https://archive.norstore.no/pages/public/datasetDetail.jsf?id=10.11582/2017.00004) | 23 patients   | 2017     | Cerebral Tumor           | MRI & intra-operative ultrasound  | yes                           | A dataset of homologous landmarks                            |
| [Cancer Digital Slide Archive](https://cancer.digitalslidearchive.org/) | -             | 2017     | Cancers                  | Glass slides of histologic images | no                            | High resolution detailed images of tissue microenvironments and  cytologic details |
| [609 Spinal  Anterior-posterior X-ray Dataset](http://spineweb.digitalimaginggroup.ca/spineweb/index.php?n=Main.Datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images) | 609 images    | 2017     | Spine                    | X-ray images                      | No                            | Each vertebra was located  by a landmark and the landmark is used to calculate Cobb angles. |
| [Cholec80](https://docs.google.com/forms/d/1GwZFM3-GhEduBs1d5QzbfFksKmS1OqXZAz8keYi-wKI/viewform?edit_requested=true) | 80 videos     | 2016     | Surgery                  | Video frames                      | no                            | A dataset containing 80 videos of surgeries performed by 13  different surgeons |
| [CRCHistoPhenotypes - Labeled Cell Nuclei Data](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/) | 100 images    | 2016     | Cell                     | Histology images                  | no                            | 100 H&E stained histology images of colorectal  adenocarcinomas |
| [CSI 2014 Vertebra Segmentation  Challenge Dataset](http://spineweb.digitalimaginggroup.ca/spineweb/index.php?n=Main.Datasets#Dataset_13.3A_xVertSeg_-_Segmentation_and_Classification_of_Fractured_Vertebrae) | 10 scans      | 2016     | Spine                    | CT scan                           | yes                           | Entire thoracic and lumbar  spine were covered by the images. The in-plane resolution is from 0.31 to  0.45mm. The slice-thickness is 1mm or 2mm. |
| [Multi-Modality Vertebra  Dataset](http://spineweb.digitalimaginggroup.ca/spineweb/index.php?n=Main.Datasets#Dataset_13.3A_xVertSeg_-_Segmentation_and_Classification_of_Fractured_Vertebrae) | 20 cases      | 2015     | Vertebra                 | MRI & CT scan                     | no                            | The 3D vertebra centre  location and orientation are annotated. |
| [CVC colon DB](http://www.cvc.uab.es/CVC-Colon/index.php/databases/) | 1200 frames   | 2012     | colon & rectum           | Colonoscopy video frames          | no                            | The dataset's region of  interest has been annotated.   The video frames were specifically chosen for maximum visual distinction  among them. |
| [LIDC-IDRI Database](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) | 1018 cases    | 2011     | Lung nodule              | CT scans                          | yes                           | A database created by 7 academic centers and 8 medical imaging  companies |
| [Computed Tomography Emphysema Dataset](https://lauge-soerensen.github.io/emphysema-database/) | 115 slices    | 2010     | COPD                     | CT scans                          | no                            | High-resolution CT scans                                     |
| [DIARETDB1](http://www2.it.lut.fi/project/imageret/diaretdb1/) | 89 images     | 2007     | Diabetic retinopathy     | fundus photographs                | no                            | A database for benchmarking the detection of diabetic  retinography |
| [ELCAP Public Lung Image Database](http://www.via.cornell.edu/lungdb.html) | 50 sets       | 2003     | Lung                     | CT scans                          | no                            | 50 low-dose documented CT scans for lungs containing nodules |
| [The Digital Database for Screening Mammography](http://www.eng.usf.edu/cvprg/Mammography/Database.html) | 2620 cases    | 1998     | Breast                   | Mammography images                | no                            | The database has the  function for user to search classes among normal, benign and cancer. |


### Table 14 – Well-known face recognition datasets. Abbreviations in the table: Oclusion (O), Pose (P), Age (A), Expression (E), Skin color (S), Gender (G), Bounding Boxes (BB), Keypoints (KP), V (video)

|                         **Dataset**                          | **Year** |  # of Subjects | # of Images | **Additional  Information** | **Highlights**                                               |
| :----------------------------------------------------------: | -------- | :-------------: | :-----------: | :-------------------------: | ------------------------------------------------------------ |
|  [VGGFace](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/)  | 2015     |      2,622      |    2.6  M     |              A              | Large-scale  celebrity recognition with high intra-class variations |
| [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)  | 2018     |      9,131      |     3.31M     |            A, P             | Diversified  pose, age, and ethnicity of celebrity faces     |
|           [LFW](http://vis-www.cs.umass.edu/lfw/)            | 2007     |      5,749      |    13,233     |              -              | The  first unconstrained FR dataset                          |
|        [MegaFace](http://megaface.cs.washington.edu/)        | 2016     |     672,052     |    4.7  M     |              -              | Raised  difficulty by including 1 M distractors, non-celebrity subjects |
|                             YTF                              | 2011     |      1,595      |   3,425  V    |              -              | Designed  for face verification in videos; same format as LFW |
| [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) | 2014     |     10,577      |    494,414    |              -              | First  publicly available large-scale FR dataset             |
| [IJB-A](https://www.nist.gov/programs-projects/face-challenges) | 2015     |       500       |     5,712     |           BB,  KP           | Manually  verified bounding boxes for face detection, nose and eye keypoints included |
| [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) | 2016     |     100,000     |     10  M     |              -              | Celebrity  identification dataset and benchmark with a linked celebrity knowledge base |
| [Pubfig](https://www.cs.columbia.edu/CAVE/databases/pubfig/) | 2009     |       200       |    60,000     |       A,  E, G, P, BB       | 73  automatically generated attributes provided, same format as LFW |
|  [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  | 2015     |     10,177      |    202,599    |             KP              | Designed  for face attribute prediction in the wild, 40 binary attributes included |
| [DiF](https://www.research.ibm.com/artificial-intelligence/publications/paper/?id=Diversity-in-Faces) | 2019     |        -        |    0.97  M    |      A,  P, S, BB, KP       | Quantitative  facial features included to reduce recognition bias across different demographics |
|      [IMDB-Face](https://github.com/fwang91/IMDb-Face)       | 2015     |     100,000     |    460,723    |            A,  G            | Age  and gender prediction on a set of celebrities collected from IMDB |
|               [UMDFaces](http://umdfaces.io/)                | 2016     |      8,501      |    367,920    |      A,  P, G, BB, KP       | Detailed  human-verified attributes and annotations          |
| [IJB-B](https://www.nist.gov/programs-projects/face-challenges) | 2015     |      1,845      |    21,798     |         A,  G, P, S         | A  superset of IJB-A with additional occlusion, illumination |
| [IJB-C](https://www.nist.gov/programs-projects/face-challenges) | 2018     |      3,531      |    31,334     |         A,  G, P, S         | An  improvement upon IJB-B with a focus on diversifying the geographic coverage  of subjects |
|  [FaceScrub](http://vintage.winklerbros.net/facescrub.html)  | 2014     |       695       |    141,130    |              G              | A  broad dataset of movie celebrities gathered from IMDB     |
|         [CACD](https://bcsiriuschen.github.io/CARC/)         | 2014     |      2,000      |    163,446    |              A              | Images  include age variations for each subject for cross-age face recognition and  retrieval, only 200 subjects are manually annotated. |


### Table 15 – remote sensing object detection datasets. Dataset size is the number of images unless states otherwise

|                         **Dataset**                          | **Year** |   **Annotation**   | **Size** | **Spatial  Resolution (cm per pixel)** |                       **Description**                        |
| :----------------------------------------------------------: | :------: | :----------------: | :------: | :------------------------------------: | :----------------------------------------------------------: |
|  [SpaceNet  C.1&C.2](https://spacenetchallenge.github.io/)   |   2019   | 685,000  buildings |  5,555   |                 30-50                  |   building  footprints annotated using polygons, 5 cities    |
|    [SpaceNet  C.3](https://spacenetchallenge.github.io/)     |   2019   |  8,676  *km* road  |  5,555   |                 30-50                  | Road  centerlines labeled based on the OpenStreetMap scheme  |
|            [COWC](https://gdo152.llnl.gov/cowc/)             |   2016   |  32,716  vehicles  |    -     |                   15                   | Car  detection dataset gathered from 6 cities in North America and Europe, cars  annotated with points on centroids |
|              [xView](http://xviewdataset.org/)               |   2018   |    1M  objects     |  1,400   |                   30                   | Large-scale  object overhead object detection dataset with bounding box annotations |
| [FMoW](https://spacenetchallenge.github.io/datasets/fmow_summary.html) |   2017   |  132,700  objects  |    1M    |                   -                    | Temporal  image sequences from over 200 countries with the purpose visual reasoning  about location, time, and sun angles. Bounding box annotations |
| [NWPU-RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) |   2017   |   31,500  scenes   |  31,500  |                20-3000                 | Aerial  scene classification dataset with variations in spatial resolution,  illumination, object pose, occlusion |
|       [TorontoCity](https://arxiv.org/abs/1612.00423)        |   2016   | 400,000  buildings |   712    |                   10                   | RGB  and LiDAR Aerial imagery of the greater Toronto area augmented with and  street-view stereo and LiDAR |
|   [DOTA](https://captain-whu.github.io/DOTA/dataset.html)    |   2018   |  188,282  objects  |  2,806   |                   -                    | Rotated  bounding box annotations verified by expert annotators, 15 common classes |
|     [TAS](http://ai.stanford.edu/~gaheitz/Research/TAS/)     |   2008   |  1,319  vehicles   |    30    |                   -                    | An  early annotated remote sensing dataset from collected from google earth,  bounding boxes |
|                            DLR3K                             |   2013   |  3,472  vehicles   |    20    |                   13                   | Rotated  bounding boxes with additional orientation annotations |
| [NWPU  VHR-10](http://www.escience.cn/people/gongcheng/NWPU-VHR-10.html) |   2016   |   3,775  objects   |   715    |                 50-200                 | generic  object detection dataset with 10 classes, bounding box annotations |
|                            LEVIR                             |   2018   |  11,000  objects   |  22,000  |                 20-100                 | Bounding  boxes, annotations provided for airplanes, ships, and oilpots |
|          [VEDAI](https://downloads.greyc.fr/vedai/)          |   2016   |  3,600  vehicles   |  1,210   |                  12.5                  | Small  vehicle detection consisting of 9 vehicle classes, rotated bounding boxes |
|          [UCAS-AOD](https://hyper.ai/datasets/5419)          |   2015   |   6,000  objects   |   910    |                   -                    | Rotated  bounding box annotations, vehicle and airplane detection, taken from Google  Earth |
|          [AID](https://captain-whu.github.io/AID/)           |   2016   |   10,000  scenes   |  10,000  |                 50-800                 |         Aerial  scene classification with 30 classes         |



### Table 16 – Remote sensing challenges. * the number of classes for the land cover classification task.

|                        **Challenge**                         | **Year** | **Dataset  size** | # of Classes |                    **Evaluation  Metric**                    |                           **Task**                           |
| :----------------------------------------------------------: | :------: | :---------------: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [SpaceNet  C.1&C.2](https://spacenetchallenge.github.io/Challenges/challengesSummary.html) |   2019   |       5,555       |       2        |                            score                             |          Building  footprint detection in 5 cities           |
| [SpaceNet  C.3](https://spacenetchallenge.github.io/Challenges/challengesSummary.html) |   2019   |       5,555       |       1        |               Average  Path Length Similarity                |                   Road  network extraction                   |
| [FMoW](https://spacenetchallenge.github.io/Challenges/challengesSummary.html) |   2017   |       1  M        |       63       |                            score                             |                    Object  Classification                    |
| [DSTL](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection) |   2017   |        57         |       10       |                             IoU                              |                    Semantic  Segmentation                    |
| [NWPU-  RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) |   2017   |      31,500       |       45       |                           Accuracy                           |                    Scene  Classification                     |
| [DIUx  xView](https://insights.sei.cmu.edu/sei_blog/2019/01/deep-learning-and-satellite-imagery-diux-xview-challenge.html) |   2018   |       1,400       |       60       |                             IoU                              |                      Object  detection                       |
|              [DeepGlobe](http://deepglobe.org/)              |   2018   |      10,000       |      7  *      | IoU,                                                                         score | Building  segmentation, road extraction, land cover classification |



### Table 17 – Species Recognition Datasets

|                           Dataset                            | # of Images | # of Classes | # of Annotations | Year | Challenge |                         Description                          |
| :----------------------------------------------------------: | :-------: | :-----: | :---------------------: | :--: | :-------: | :----------------------------------------------------------: |
| [Flower  102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) |   8,189   |   103   |        8,189  SM        | 2008 |    No     | Flower  recognition dataset of 103 flower categories common in the United Kingdom |
| [Caltech-Birds  2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) |  11,788   |   200   |       11,788  BB        | 2011 |    No     |      15  part locations and 28 attributes for each bird      |
| [Stanford  Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) |  22,000   |   120   |       22,000  BB        | 2011 |    No     |  Single-object  per image dataset for dog breed recognition  |
|            [F4K](http://groups.inf.ed.ac.uk/f4k/)            |  27,370   |   23    |       27,370  CL        | 2012 |    No     | Fish  recognition dataset annotated by following marine biologists |
| [Snapshot  Serengeti](https://www.zooniverse.org/projects/zooniverse/snapshot-serengeti) |  1.2  m   |   61    | 406,433  CL, 150,000 BB | 2014 |    No     | Wild  animal classification dataset gathered using 225 camera-traps in Serengeti  National Park in Africa |
|       [NABirds](https://dl.allaboutbirds.org/nabirds)        |  48,562   |   555   |       48,562  BB        | 2015 |    No     | Expert-curated  dataset of North American birds, 11 bird parts annotated in every image |
|     [PlantCLEF](https://www.imageclef.org/PlantCLEF2020)     |  434,251  | 10,000  |       10,000  CL        | 2015 |    Yes    | Plant  classification dataset gathered in the Amazon rainforest |
|      [iNat](https://www.kaggle.com/c/inaturalist-2018)       |  675,175  |  5,089  |       561,767  BB       | 2017 |    Yes    | Manually  collected dataset of 13 super-class and 5k sub-class species, organized in a hierarchical taxonomy, highly imbalanced |
| [Dogs-in-the-Wild](https://ai.baidu.com/broad/introduction?dataset=canine) |  300,000  |   362   |       300,000  CL       | 2018 |    No     | A  large dataset for dog breed classification in natural environments |
|        [AnimalWeb](https://arxiv.org/abs/1909.04951)         |  21,900   |   334   |        198k  KP         | 2019 |    No     | Hierarchically  categorized dataset for animal face recognition with 9 keypoint annotations  per face |
|           [IP102](https://github.com/xpwu95/IP102)           |  75,000   |   102   | 75,000  CL,  19,000  BB | 2019 |    No     | Hierarchically  categorized dataset for insect pest recognition |



### Table 18 - Clothing Detection Datasets

|                           Dataset                            | # of Images | # of Classes | # of Annotated  Clothing instances | Annotation  Type | Year | Challenge/Benchmark | # of Attributes |
| :----------------------------------------------------------: | :-------: | :-----: | :-----------------------------: | :--------------: | :--: | :-----------------: | :-----------: |
|                             DARN                             |  182,000  |   20    |             182,000             |        BB        | 2015 |         No          |       9       |
|    [Street2Shop](http://www.tamaraberg.com/street2shop/)     |  404,000  |   11    |             20,357              |        BB        | 2015 |         No          |       -       |
| [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) |  800,000  |   50    |             180,000             |        KP        | 2016 |         Yes         |       5       |
|          [ModaNet](https://github.com/eBay/modanet)          |  55,000   |   13    |             240,000             |     BB,  SM      | 2018 |         No          |       -       |
| [FashionAI](https://tianchi.aliyun.com/markets/tianchi/FashionAI) |  324,000  |   41    |             324,000             |        KP        | 2018 |         No          |      68       |
| [Deepfashion2](https://github.com/switchablenorms/DeepFashion2) |  491,000  |   13    |             801,000             |   BB,  SM, KP    | 2019 |         Yes         |       4       |
