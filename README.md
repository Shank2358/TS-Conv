### 更新啦~~
# TS-Conv: Task-wise Sampling Convolutions for Arbitrary-Oriented Object Detection in Aerial Images  

  <a href="https://github.com/Shank2358/GGHL/">
    <img alt="Version" src="https://img.shields.io/badge/Version-1.3.0-blue" />
  </a>
  
  <a href="https://github.com/Shank2358/GGHL/blob/main/LICENSE">
    <img alt="GPLv3.0 License" src="https://img.shields.io/badge/License-GPLv3.0-blue" />
  </a>
  
  <a href="https://github.com/Shank2358" target="_blank">
  <img src="https://visitor-badge.glitch.me/badge?page_id=gghl.visitor-badge&right_color=blue"
  alt="Visitor" />
</a> 

<a href="mailto:zhanchao.h@outlook.com" target="_blank">
   <img alt="E-mail" src="https://img.shields.io/badge/To-Email-blue" />
</a> 
Code is coming soon.

## This is the implementation of TS-Conv 👋👋👋
[[Arxiv](https://arxiv.org/abs/2209.02200)]

### Please give a ⭐️ if this project helped you. If you use it, please consider citing:
  ```Arxiv
  @ARTICLE{9709203,
  author={Huang, Zhanchao and Li, Wei and Xia, Xiang-Gen，Hao Wang and Tao, Ran},
  journal={arXiv}, 
  title={Task-wise Sampling Convolutions for Arbitrary-Oriented Object Detection in Aerial Images}, 
  year={2022},
  volume={},
  number={},
  pages={1-16},
  doi={10.48550/arXiv.2209.02200}}
  ```
### 🤡🤡🤡 Clone不Star,都是耍流氓    


### 👹👹👹 不出意外的话是毕业前最后一个工作了，可能也是学术圈的最后一个工作（即将失业，，，），有点水大家见谅。   


## 0. Something Important 🦞 🦀 🦑 

* ### 🎃🎃🎃 The usage of the TS-Conv repository is the same as that of the ancestral repository [GGHL](https://github.com/Shank2358/GGHL). If you have any questions, please see the issues there.  
    #### 用法和祖传的[GGHL](https://github.com/Shank2358/GGHL)仓库一样，有问题可以看那边的issues。MMRotate版本也在写着。TS-Conv还将持续更新一段时间，现在更新完的是主体模型的代码，重点在head，DCN，以及dataload的标签分配那部分，其他和GGHL差不多。可视化和更多其它部分的功能和实验我也在抓紧更新中。   

* ### 💖💖💖 Thanks to [Crescent-Ao](https://github. com/Crescent-Ao) and [haohaoolalahao](https://github.com/haohaoolalahao) for contributions to the GGHL repository, thanks to [Crescent-Ao](https://github.com/Crescent-Ao) for the GGHL deployment Version. Relevant warehouses will continue to be updated, so stay tuned.  
    #### 打个广告，GGHL部署版本[GGHL-Deployment](https://github.com/Crescent-Ao/GGHL-Deployment)已经上线，欢迎大家使用~~ 感谢我最亲爱的师弟[Crescent-Ao](https://github.com/Crescent-Ao)和[haohaolalahao](https://github.com/haohaolalahao)对GGHL仓库的贡献，感谢[Crescent-Ao](https://github.com/Crescent-Ao)完成的GGHL部署版本。相关仓库还会持续更新中，敬请期待。

* ### 😺😺😺 Welcome everyone to pay attention to the MGAR completed by [haohaoolalahao](https://github.com/haohaoolalahao) in cooperation with me, which has been accepted by [IEEE TGRS](https://ieeexplore.ieee.org/document/9912396).  
    ### 再打个广告，欢迎大家关注[haohaolalahao](https://github.com/haohaolalahao)与我合作完成的遥感图像目标检测工作 MGAR: Multi-Grained Angle Representation for Remote Sensing Object Detection，论文已经正式接收[IEEE TGRS](https://ieeexplore.ieee.org/document/9912396) [Arxiv](https://arxiv.org/abs/2209.02884), 感谢大家引用：
  ```IEEE TGRS
    @ARTICLE{9912396,
      author={Wang, Hao and Huang, Zhanchao and Chen, Zhengchao and Song, Ying and Li, Wei},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Multi-Grained Angle Representation for Remote Sensing Object Detection}, 
      year={2022},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TGRS.2022.3212592}}
  ```  

## 🌈 1.Environments
Linux (Ubuntu 18.04, GCC>=5.4) & Windows (Win10)   
CUDA > 11.1, Cudnn > 8.0.4

First, install CUDA, Cudnn, and Pytorch.
Second, install the dependent libraries in [requirements.txt](https://github.com/Shank2358/GGHL/blob/main/requirements.txt). 

```python
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
pip install -r requirements.txt  
```
    
## 🌟 2.Installation  
1. git clone this repository    

2. Polygen NMS  
The poly_nms in this version is implemented using shapely and numpy libraries to ensure that it can work in different systems and environments without other dependencies. But doing so will slow down the detection speed in dense object scenes. If you want faster speed, you can compile and use the poly_iou library (C++ implementation version) in datasets_tools/DOTA_devkit. The compilation method is described in detail in [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) .

```bash
cd datasets_tools/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace 
```   
  
## 🎃 3.Datasets

1. [DOTA dataset](https://captain-whu.github.io/DOTA/dataset.html) and its [devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)  

#### (1) Training Format  
You need to write a script to convert them into the train.txt file required by this repository and put them in the ./dataR folder.  
For the specific format of the train.txt file, see the example in the /dataR folder.   

```txt
image_path xmin,ymin,xmax,ymax,class_id,x1,y1,x2,y2,x3,y3,x4,y4,area_ratio,angle[0,180) xmin,ymin,xmax,ymax,class_id,x1,y1,x2,y2,x3,y3,x4,y4,area_ratio,angle[0,180)...
```  
The calculation method of angle is explained in [Issues #1](https://github.com/Shank2358/GGHL/issues/1) and our paper.

#### (2) Validation & Testing Format
The same as the Pascal VOC Format

#### (3) DataSets Files Structure
  ```
  cfg.DATA_PATH = "/opt/datasets/DOTA/"
  ├── ...
  ├── JPEGImages
  |   ├── 000001.png
  |   ├── 000002.png
  |   └── ...
  ├── Annotations (DOTA Dataset Format)
  |   ├── 000001.txt (class_idx x1 y1 x2 y2 x3 y3 x4 y4)
  |   ├── 000002.txt
  |   └── ...
  ├── ImageSets
      ├── test.txt (testing filename)
          ├── 000001
          ├── 000002
          └── ...
  ```  
There is a DOTA2Train.py file in the datasets_tools folder that can be used to generate training and test format labels.
First, you need to use [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) , the official tools of the DOTA dataset, for image and label splitting. Then, run DOTA2Train.py to convert them to the format required by GGHL. For the use of DOTA_devkit, please refer to the tutorial in the official repository.

## 🌠🌠🌠 4.Usage Example
#### (1) Training  
```bash
sh train_GGHL_dist.sh
```

#### (2) Testing  
```python
python test.py
```
## 📝 License  
Copyright © 2021 [Shank2358](https://github.com/Shank2358).<br />
This project is [GNU General Public License v3.0](https://github.com/Shank2358/GGHL/blob/main/LICENSE) licensed.

## 🤐 To be continued 
