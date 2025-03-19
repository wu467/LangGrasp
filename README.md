## LangGrasp: Leveraging Fine-Tuned LLMs for Language Interactive  Robot Grasping with Ambiguous Instructions

## Video
[![LangGrasp: Leveraging Fine-Tuned LLMs for Language Interactive Robot Grasping with Ambiguous Instructions](https://img.youtube.com/vi/qX2jbXbaZb0/maxresdefault.jpg)](https://youtu.be/qX2jbXbaZb0 "LangGrasp: Leveraging Fine-Tuned LLMs for Language Interactive Robot Grasping with Ambiguous Instructions")



## LangGrasp
LangGrasp is a research project that combines open-vocabulary part segmentation (via VLPart) with 3D grasp detection (via the Graspness Implementation). It allows you to segment object parts in images and then plan robotic grasps on those parts using 3D point cloud data.

### Install VLPart and Graspness Implementation (Prerequisites)
This project relies on two base repositories, VLPart (for part segmentation) and graspness_implementation (for grasp detection). Please install these first:

* **VLPart:** Requires PyTorch and Detectron2. Follow the official instructions to set up VLPart:

```bash
# Install Detectron2 (required by VLPart)
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 
pip install -e .            # install detectron2 in development mode
cd ..

# Clone and install VLPart
git clone https://github.com/facebookresearch/VLPart.git
cd VLPart
pip install -r requirements.txt  # install VLPart dependencies
# (Optional) Install VLPart package
pip install -e .
```

***Note***: VLPart is built on **Detectron2** and PyTorch (≥1.9). Make sure PyTorch and TorchVision are installed and match your CUDA version

* **Graspness Implementation:** Requires PyTorch and several 3D libraries. Install the graspness project as follows:
```bash
# Clone the Graspness implementation repository
git clone https://github.com/rhett-chen/graspness_implementation.git
cd graspness_implementation
pip install -r requirements.txt  # install graspness dependencies

# Compile and install custom PointNet++ operators
cd pointnet2
python setup.py install  && cd ..

# Compile and install the CUDA KNN operator
cd knn
python setup.py install  && cd ..

# Install GraspNet API (for dataset and evaluation utilities)
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI 
pip install .  && cd ..

# (Optional) Install graspness_implementation as a package
pip install -e .

```

***Note***: The graspness project relies on MinkowskiEngine for 3D sparse convolution. You may need to install MinkowskiEngine separately, as its pip installation depends on your system’s CUDA and PyTorch version. Please refer to the MinkowskiEngine repository for instructions. Typically, you can install a matching wheel, for example: pip install MinkowskiEngine==0.5.4 -f https://developer.nvidia.com/compute/minkowski-engine (use the version and link appropriate for your CUDA/PyTorch combo). Ensure you also have a C++ compiler and CUDA toolkit installed for building the custom operators above.