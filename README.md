# Autonomous Vehicle Path Planning Using Parallel Processing

## Overview
This project focuses on **real-time path planning** for autonomous vehicles using deep learning models, with an emphasis on **parallel processing** to improve training efficiency and performance. The project implements **CNN** and **ResNet-18** models to predict steering angles based on raw input images from the CARLA dataset. By leveraging PyTorch's parallelization capabilities, we achieve significant reductions in computation time and enhance model accuracy. This project is done for the course Parallel Processing for Data Analytics at Northeastern University by Professor Stratis Ioannadis

## Key Features
- **Parallel vs. Serial Training**: Implemented parallel processing to utilize multiple CPU cores and GPUs, reducing training time by 90% compared to serial execution.
- **Deep Learning Models**:
  - **Convolutional Neural Network (CNN)**: Inspired by NVIDIA's end-to-end self-driving car research.
  - **ResNet-18**: Fine-tuned for regression tasks to predict steering angles.
- **Dataset**: The CARLA simulator dataset consisting of **186k road images** with corresponding steering angle data across varying weather conditions.
- **Performance Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Epoch Duration

## Results
### CNN Model
| Implementation | Training Loss | Validation Loss | R² Score | Epoch Duration (s) |
|-----------------|--------------|-----------------|----------|-------------------|
| Serial         | 0.0041       | 0.0044          | 0.4898   | ~3000             |
| Parallel       | 0.0029       | 0.0027          | 0.6570   | ~336              |

### ResNet-18 Model
| Implementation | Validation Loss | Validation RMSE |
|-----------------|-----------------|-----------------|
| Serial         | 0.0007          | 0.0265          |
| Parallel       | 0.0003          | 0.0214          |

**Key Insight**: Parallelization led to **90% faster training** and more accurate steering predictions.

## Dependencies
To run the project, install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

### Key Libraries:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- CARLA Simulator (optional, for dataset generation)


## Results Visualization
The training loss, validation loss, and R² score are logged during training. Use the provided script to visualize the metrics:
```bash
python plot_results.py
```

## Future Work
- Explore more advanced architectures (e.g., Transformer-based models).
- Implement real-time testing on physical hardware.
- Extend the system to handle dynamic and unstructured environments.

## References
- [NVIDIA Self-Driving Car Research](https://arxiv.org/pdf/1604.07316)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch Multithreading](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
- [CARLA Simulator](https://carla.org/)
