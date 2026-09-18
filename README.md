# Autonomous-Robot-for-Airport-Runway-FOD-Management

# ✈️ Airport Runway FOD Detection & Removal System

## 📌 Project Overview

A **vision-based Foreign Object Debris (FOD) detection and robotic removal system** developed for runway surface inspection. The system combines **deep-learning-based object detection, real-time image processing, Raspberry Pi-based video acquisition, and a 5-DOF robotic arm** to identify and remove runway debris.

The vision pipeline uses **YOLO11n with OpenCV** for FOD detection and classification. The detected FOD information is processed for targeted robotic pick-and-place, followed by **post-removal visual verification** along a predefined inspection path.

---

## 🚨 Problem Statement

Foreign Object Debris (FOD) on airport runways can include stones, metal fragments, plastic components, and other loose objects that may pose a risk to aircraft operations.

Manual inspection requires continuous human monitoring and physical intervention for debris removal. Conventional approaches may also lack an integrated mechanism for **automated detection, targeted removal, and confirmation of runway cleanliness**.

### Proposed Approach

Develop a computer-vision-driven system capable of:

**Detecting → Classifying → Localizing → Removing → Re-inspecting**

FOD objects on the runway surface.

---

## 🎯 Objectives

* Develop a deep-learning-based FOD detection pipeline.
* Detect and classify **31 FOD categories** using YOLO11n.
* Process live camera frames using OpenCV.
* Transfer video from the Raspberry Pi to a laptop-based vision system.
* Extract FOD detection information from the vision pipeline.
* Perform targeted FOD removal using a **5-DOF robotic arm**.
* Operate the vehicle along a **predefined inspection path**.
* Perform post-removal inspection to verify debris clearance.

---

## 🧠 Computer Vision Pipeline

The core detection pipeline consists of:

```text
USB Camera
     ↓
Raspberry Pi 3A+
     ↓
Video Streaming
     ↓
Laptop / Processing System
     ↓
OpenCV Frame Processing
     ↓
YOLO11n Inference
     ↓
FOD Detection
     ↓
Class + Bounding Box + Confidence
     ↓
Robotic Removal
```

### YOLO11n Detection

YOLO11n is used as the primary object detection model because of its lightweight architecture and suitability for real-time inference.

For each detected object, the model provides:

* **FOD class**
* **Bounding-box coordinates**
* **Detection confidence**
* **Object location within the frame**

These outputs are used to determine the target debris for the robotic removal process.

---

## 📊 Dataset & Model Performance

| Parameter           |              Value |
| ------------------- | -----------------: |
| Model               |            YOLO11n |
| Dataset Size        |  **33,793 images** |
| Number of Classes   | **31 FOD classes** |
| Validation Accuracy |          **99.6%** |
| Processing          |    Python + OpenCV |
| Detection Type      |   Object Detection |

The model was trained using an annotated FOD dataset containing **33,793 images distributed across 31 object categories**.

---

## 🤖 Robotic Removal Subsystem

The robotic subsystem integrates a **5-DOF robotic arm** with the FOD detection pipeline.

### Pick-and-Place Sequence

```text
FOD Detection
     ↓
Target Identification
     ↓
Arm Positioning
     ↓
Gripper Alignment
     ↓
Object Grasp
     ↓
Object Transfer
     ↓
Collection Bin
     ↓
Release
```

The robotic arm is responsible for physically removing the detected debris and placing it into the designated collection area.

---

## 🚗 Vehicle & Inspection Mechanism

The inspection vehicle uses a **predefined path** to systematically cover the designated runway inspection area.

The system does not depend on dynamic autonomous navigation. Instead, the predefined movement pattern provides a controlled inspection trajectory while the computer vision subsystem identifies FOD encountered along the path.

After the removal operation, the vehicle repeats the inspection path to perform **post-removal verification**.

---

## 🔄 Complete System Workflow

```text
┌──────────────────────┐
│    USB Camera        │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Raspberry Pi 3A+     │
│ Video Acquisition    │
└──────────┬───────────┘
           ↓
      Video Stream
           ↓
┌──────────────────────┐
│ Laptop Vision System │
│ Python + OpenCV      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      YOLO11n         │
│ FOD Detection        │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Class / Bounding Box │
│ / Confidence         │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Targeted Robotic     │
│ Removal – 5 DOF Arm  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Collection Bin       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Repeat Inspection    │
│ & Verification       │
└──────────────────────┘
```

---

## ⚙️ Hardware Components

| Component                | Purpose                                 |
| ------------------------ | --------------------------------------- |
| **Raspberry Pi 3A+**     | Camera interfacing & video transmission |
| **USB Webcam**           | Runway image/video acquisition          |
| **5-DOF Robotic Arm**    | FOD pick-and-place                      |
| **MG996R Servo**         | High-torque arm actuation               |
| **MG90S Servos**         | Robotic arm joints                      |
| **L298N Motor Driver**   | DC motor control                        |
| **12V DC Geared Motors** | Vehicle propulsion                      |
| **Buck Converter**       | Voltage regulation                      |
| **Li-ion Battery Pack**  | System power                            |

---

## 💻 Software & Technologies

### Programming

* Python

### Computer Vision

* OpenCV

### Deep Learning

* YOLO11n
* Ultralytics

### Embedded System

* Raspberry Pi OS
* Raspberry Pi 3A+

### Robotics

* Servo-based 5-DOF robotic arm
* DC geared motor drive
* L298N motor driver

---

## 🔬 Technical Architecture

The system is divided into four major modules:

### 1. Video Acquisition Module

The USB webcam captures the runway surface continuously. The Raspberry Pi acts as the edge-side acquisition unit and streams the captured video to the laptop.

### 2. FOD Detection Module

The laptop receives the video stream and processes frames using OpenCV. YOLO11n performs object detection and returns the FOD class, bounding box, and confidence information.

### 3. Robotic Removal Module

The detected target is passed to the robotic handling stage. The 5-DOF arm performs the required pick-and-place sequence to remove the debris.

### 4. Verification Module

Following the removal operation, the vehicle repeats the predefined inspection trajectory. The vision system performs another detection cycle to verify whether the previously identified debris has been cleared.

---

## 📈 Results

The trained YOLO11n model achieved:

> **99.6% validation accuracy**

on a dataset consisting of:

> **33,793 annotated images across 31 FOD classes**

The integrated prototype demonstrates a complete workflow from **visual FOD detection to robotic removal and post-removal verification**.

---

## 🔮 Future Scope

* Investigate **electro-adhesive grippers** for handling FOD with varying shapes, materials, and surface characteristics.
* Improve camera-to-arm coordinate mapping for more precise object grasping.
* Expand the dataset with different runway surfaces, illumination conditions, and object orientations.
* Optimize inference latency for improved real-time performance.
* Improve robotic manipulation and object-handling reliability.
* Extend runway coverage and inspection capabilities.

---

## 🛠️ Tech Stack

```text
Python
YOLO11n
OpenCV
Ultralytics
Deep Learning
Computer Vision
Raspberry Pi 3A+
5-DOF Robotics
Servo Control
DC Motor Control
L298N
```

---

## 👨‍💻 Project Information

**Project:** Airport Runway FOD Detection & Removal System
**Domain:** Computer Vision | Deep Learning | Robotics | Embedded Systems
**Application:** Airport Runway Safety & FOD Management
