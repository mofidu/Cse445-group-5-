# Face Recognition System (Group - 5)

## Project Overview
This project aims to build a **Face Recognition System** capable of identifying 10 individuals using their facial images. The system will recognize known individuals accurately and reject unknown faces by employing machine learning techniques.

---

## Table of Contents
- [Introduction](#introduction)
- [Literature Review and Initial Research](#literature-review-and-initial-research)
- [Dataset Exploration](#dataset-exploration)
- [Model Selection: Why KNN?](#model-selection-why-knn)
- [Thresholding and Unknown Detection](#thresholding-and-unknown-detection)
- [Evaluation Metrics](#evaluation-metrics)
- [Future Work and Deployment Plan](#future-work-and-deployment-plan)
- [Summary](#summary)

---

## Introduction
The primary objective of this project is to develop a system that accurately identifies 10 individuals based on their facial images. The system should also be capable of rejecting unknown faces, ensuring robustness in real-world scenarios.

---

## Literature Review and Initial Research
- Explored face recognition methodologies using online resources and academic papers.
- Identified a typical face recognition pipeline:
  - Dataset collection and preparation
  - Image preprocessing
  - Model selection and training
- Recognized the importance of multiple images per person, covering various expressions and angles.
- Understood that proper labeling, scaling, and annotation of the dataset are critical.

---

## Dataset Exploration
- Considered two approaches:
  - Use publicly available datasets from the internet.
  - Create a custom dataset manually if suitable datasets are unavailable.
- The target is 10 individuals with 15-20 images each for sufficient variation.

---

## Model Selection: Why KNN?
- Dataset size is relatively small (~200 images).
- K-Nearest Neighbors (KNN) chosen due to:
  - Simplicity and effectiveness on small datasets.
  - No lengthy training time required.
  - Reliable performance with quality feature extraction.
- Incorporated Principal Component Analysis (PCA) to reduce dimensionality while preserving essential facial features.
- Literature shows PCA + KNN can achieve around 94% accuracy.

---

## Thresholding and Unknown Detection
- KNN always predicts a class, which can misclassify unknown faces.
- Implemented a distance-based threshold mechanism:
  - If the nearest neighbor’s distance exceeds the threshold, classify as "Unknown."
- This method is widely accepted in face recognition tasks for handling unknowns.

---

## Evaluation Metrics
- Accuracy: Measure correct identification of known faces.
- Precision, Recall, F1-Score: For overall recognition quality.
- False Acceptance Rate (FAR) and False Rejection Rate (FRR): Especially for unknown face rejection performance.

---

## Future Work and Deployment Plan
- Finalize dataset collection and preprocessing.
- Train and fine-tune the model.
- Develop a simple web application for deployment:
  - Frontend: HTML/CSS
  - Backend: Django
  - Model integration through serialized `.pkl` files

---

## Contributors : (Group-5)
- Md Mofidul Hassan
- Md. Shahadat Hossain
- Barshon Basak
- KM Rayed Zakwan
