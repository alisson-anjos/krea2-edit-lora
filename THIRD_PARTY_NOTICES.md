# Third-Party Notices

This project interoperates with and optionally downloads third-party code and model weights. Those components retain their own licenses.

## Krea 2

- Source: https://github.com/krea-ai/krea-2
- Model: https://huggingface.co/krea/Krea-2-Raw

Use of the model and derived adapters is subject to the Krea 2 license.

## Qwen

- Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Qwen Image: https://huggingface.co/Qwen/Qwen-Image

These components provide vision-language conditioning and the image VAE.

## ArcFace / InsightFace

- Architecture source: https://github.com/deepinsight/insightface
- PyTorch conversion used by this project: https://github.com/cosanlab/py-feat
- Weights: https://huggingface.co/py-feat/arcface_r50

The IResNet implementation in `src/identity.py` follows the MIT-licensed fused-BN conversion in py-feat. The downloaded pretrained weights retain the InsightFace non-commercial research restriction. Enabling the optional identity loss does not change that restriction.

## OpenCV YuNet

- Model: https://huggingface.co/opencv/face_detection_yunet
- Project: https://github.com/opencv/opencv_zoo

YuNet is used only while building the optional identity cache.

## ComfyUI Krea2Edit

- Node pack: https://github.com/lbouaraba/comfyui-krea2edit

The repository includes a small compatibility patch for matching `t=0` reference modulation at inference. The node pack itself is not redistributed here.
