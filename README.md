# Semi-Automated Magnetic Particle Inspection Bench with Embedded AI

Final-year engineering project (2025) at **Marlier Maroc**, a non-destructive testing (NDT) company
serving the aerospace, automotive and heavy industry sectors.

![3D model of the inspection bench](images/bench_3d.png)

## Problem
Magnetic particle inspection (MPI) finds surface defects on ferromagnetic parts. Done by hand, it is
slow, depends on the operator's eye, and leaves little traceability.

## Solution
A semi-automated bench that chains magnetisation, developer spraying and demagnetisation, and
analyses the UV images with embedded AI.

```mermaid
flowchart LR
    A[Part loaded] --> B[Magnetisation<br/>solenoid tunnel]
    B --> C[Developer spraying]
    C --> D[UV image capture]
    D --> E[CNN on Jetson Nano<br/>OK / NOT OK]
    E --> F[Demagnetisation]
    E --> G[(SQL database)]
    G --> H[Power BI dashboard]
```

## 3D mechanical design
Full bench modelled in **Autodesk Inventor**: modular frame, part holder, pole pieces,
custom-sized solenoid tunnel, spraying system and UV inspection tunnel.

![Rear view of the bench](images/bench_3d_rear.png)

## Automation & HMI
- **Siemens S7-1200** program in **TIA Portal**: general GRAFCET, emergency-stop GRAFCET and one
  macro-step per stage (magnetisation, spraying, inspection, demagnetisation), translated to Ladder
- **Touch HMI**: home, control and supervision screens, with the four magnetisation methods (M1–M4)
  and live temperature, lux, UV and field intensity readings

![HMI control screen](images/hmi_control.png)

## Embedded AI
- CNN models (**VGG16**, **EfficientNetB0**, Keras) trained with data augmentation on UV inspection images
- Deployed on an **NVIDIA Jetson Nano** for real-time analysis: **96 % accuracy**
- **Gradio** interface to review predictions, correct labels and retrain

## Traceability
Inspection results stored in a **SQL database**, with a **Power BI** dashboard for reporting.

## Stack
Autodesk Inventor · Siemens S7-1200 · TIA Portal · GRAFCET / Ladder · HMI · Python · Keras · OpenCV ·
Jetson Nano · Gradio · SQL · Power BI
