#!/bin/bash
cp patch/Depth-Anything-V2/run.py tools/Depth-Anything-V2/run.py
cp patch/Video-Depth-Anything/run.py tools/Video-Depth-Anything/run.py
cp patch/utils3d/utils.py tools/utils3d/utils3d/numpy/utils.py
cp patch/TrajectoryAttention/my_svd_pipeline_clean.py tools/TrajectoryAttention/models/my_svd_pipeline_clean.py
cp patch/DiffusionAsShader/pipelines.py tools/DiffusionAsShader/models/pipelines.py
