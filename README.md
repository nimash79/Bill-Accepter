# Bill-Accepter — Banknote Recognition & Authentication

A Python-based banknote inspection system that combines computer vision, image processing, and camera integration to recognize US dollar banknotes, extract their serial numbers, and evaluate their authenticity.

Bill-Accepter provides a desktop interface for monitoring the inspection process, viewing captured images, and reviewing recognition results. It is designed to explore automated banknote verification by combining visual pattern matching with counterfeit-detection techniques.

## Overview

Manually checking banknotes can be time-consuming, particularly when denomination recognition, serial number extraction, and authenticity assessment must be performed together.

**Bill-Accepter aims to automate these inspection tasks** through a camera-assisted workflow. The application captures banknote images, processes them using computer vision techniques, and displays the resulting information through a graphical interface.

The project brings together image acquisition, banknote recognition, serial number processing, and authenticity assessment in one application.

## Key Features

- **Banknote recognition:** Analyzes captured banknote images to identify the denomination.
- **Serial number extraction:** Processes banknote images to recognize and retrieve serial number information.
- **Counterfeit detection:** Uses image-based analysis to assess whether a banknote appears authentic or potentially counterfeit.
- **Template matching:** Compares image features against reference patterns to support banknote recognition.
- **Camera integration:** Captures images from a network-accessible camera.
- **Desktop interface:** Provides an interactive interface for controlling inspections and viewing results.
- **Live inspection status:** Displays connection status, processing progress, and operation logs.
- **Visual result presentation:** Shows captured images and inspection information, including denomination, serial number, match score, and authenticity assessment.
- **Serial communication:** Supports communication with a connected serial device as part of the inspection workflow.

## How It Works

The application combines several processing stages to inspect a banknote.

1. **Image acquisition:** The application retrieves a frame from the configured camera.
2. **Image preprocessing:** The captured image is cropped and prepared for subsequent analysis.
3. **Banknote recognition:** Image-matching techniques compare the captured banknote against available reference patterns to help identify its denomination.
4. **Serial number processing:** Dedicated processing logic analyzes the relevant banknote image regions to obtain serial number information.
5. **Authenticity assessment:** The counterfeit-detection module evaluates visual characteristics to produce an authenticity assessment.
6. **Result presentation:** The desktop interface displays the inspection results, images, progress information, and diagnostic logs.

The exact recognition accuracy depends on image quality, reference data, supported banknote types, and the performance of the implemented detection methods.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| OpenCV | Image acquisition and computer vision |
| PyQt5 | Desktop graphical user interface |
| asyncio | Asynchronous operations |
| threading | Background task execution |
| pyserial / serial_asyncio | Serial-port communication |
| Template matching | Visual pattern comparison |

The repository also includes separate Python modules for camera handling, image cropping, serial number processing, template matching, and counterfeit detection.

## Project Structure

```text
Bill-Accepter/
├── app screenshots/       # Application screenshots
├── dataset/               # Banknote image dataset
├── dataset_old/           # Earlier dataset files
├── documents/              # Project documentation
├── models/                 # Model-related files
├── app.py                  # Main desktop application
├── camera.py               # Camera integration
├── config.py               # Application configuration
├── crop.py                 # Image cropping utilities
├── debug.py                # Debugging utilities
├── fake_detection.py       # Counterfeit detection
├── serial_number.py        # Serial number processing
├── sift.py                 # SIFT-based image processing
├── template_matching.py    # Template matching
├── picture.py              # Image-related utilities
├── stream.py               # Stream-related functionality
├── requirements.txt        # Python dependencies
├── LICENSE                 # Project license
└── ...
```

The repository also contains several older implementation files, including `fake_detection_old.py`, `main_old.py`, and `template_matching_old.py`, which preserve earlier development iterations.

## Getting Started

### Prerequisites

Before running the application, ensure you have:

- Python installed on your system.
- A working camera accessible through the configured network endpoint.
- The required Python packages.
- A serial device and appropriate connection settings if your inspection setup requires serial communication.
- The project's reference images and any required model or dataset files.

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/nimash79/Bill-Accepter.git
cd Bill-Accepter
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

Activate it using the appropriate command for your operating system.

Windows:

```bash
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**4. Configure the camera**

Review `config.py` and set the camera's IP address to match your environment. The application constructs its video endpoint using this configuration.

Make sure the camera is reachable and that the configured endpoint is accessible before starting an inspection.

**5. Run the application**

```bash
python app.py
```

These instructions describe the expected setup workflow. Additional configuration or hardware-specific adjustments may be necessary depending on your environment.

## Using the Application

Once the desktop application starts:

1. Verify that the camera and any required serial device are available.
2. Connect to the serial device if required by your setup.
3. Start the inspection process from the main interface.
4. Follow the displayed progress and diagnostic messages.
5. Review the captured images and the resulting denomination, serial number, match score, and authenticity assessment.
6. Stop the inspection or clear the displayed logs when needed.

The available controls and results depend on the current application configuration and the connected hardware.

## Datasets and Reference Images

The repository contains `dataset/` and `dataset_old/` directories, as well as a `models/` directory. These resources support the project's image-processing and recognition workflow.

The usefulness of the recognition and authenticity-assessment methods depends on the quality, coverage, and labeling of the available reference data.

For reliable evaluation, testing should include different banknote conditions, image orientations, lighting environments, and representative examples of both genuine and counterfeit notes.

## Limitations

- Image-based authenticity assessment is not a guarantee that a banknote is genuine or counterfeit.
- Recognition performance may be affected by lighting, blur, image resolution, banknote wear, and camera positioning.
- The system depends on appropriate reference images and configuration.
- Camera and serial communication require compatible hardware and correct connection settings.
- The repository contains legacy implementation files, so not every module necessarily represents the current execution path.

This project should be treated as a computer vision and banknote-inspection application rather than a certified currency-authentication device.

## Potential Improvements

Possible future enhancements include:

- Expanding support for additional currencies and denominations.
- Improving serial number recognition under challenging imaging conditions.
- Training and evaluating dedicated machine-learning models for banknote classification and counterfeit detection.
- Adding confidence scores and clearer explanations for authenticity assessments.
- Creating a standardized evaluation pipeline with precision, recall, F1-score, and confusion matrices.
- Improving configuration management and documenting hardware requirements.
- Adding automated tests and more comprehensive installation documentation.

## License

This project is distributed under the [Apache License 2.0](LICENSE).

## Author

**Nima Sharifinia**

- GitHub: [@nimash79](https://github.com/nimash79)
- Repository: [Bill-Accepter](https://github.com/nimash79/Bill-Accepter)

---

*Bill-Accepter explores the application of computer vision and image processing to automated banknote recognition and authenticity assessment.*
