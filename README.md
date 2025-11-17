Features:
- Detects green trees, autumn leaves, and shaded underbrush
- Supports Blob and Contour detection methods
- Automatically displays output windows with detected objects highlighted
- Saves output images and mask previews

Requirements:
- Python 3.12+
- OpenCV (opencv-python)
- NumPy (numpy)
- imutils (imutils)

Install required packages:
pip install -r requirements.txt

Example requirements.txt content:
numpy
opencv-python
imutils

Project Structure:
tree-counter/
│
├─ count_trees.py         # Main script to run tree counting
├─ utils.py               # Helper functions for image processing
├─ images/                # Input images
│   └─ forest.jpeg
├─ results/               # Output images
├─ venv/                  # Python virtual environment
└─ README.txt             # Project instructions

Usage Instructions:

1. Clone the repository

2. Create a virtual environment (optional but recommended)
python -m venv venv

Activate it:
Windows:
.\venv\Scripts\activate

Linux/macOS:
source venv/bin/activate

3. Install required packages
pip install -r requirements.txt

4. Run the project

Contour detection (recommended for trees, leaves, underbrush):
python "count_trees.py" --image "images/forest.jpeg" --method contour --out "results/contour_output.png"
- Opens two windows:
  - Tree Detection (Contour) → shows green circles around detected objects
  - Mask Preview → shows detected areas
- Press any key to close the windows

Blob detection (alternative method):
python "count_trees.py" --image "images/forest.jpeg" --method blob --out "results/blob_output.png"
- Opens a window highlighting detected blobs

5. Check output
- Output images are saved in the results/ folder.
- Example:
results/contour_output.png
results/contour_output_mask.png
results/blob_output.png

6. Add your own images
- Place images inside the images/ folder
- Update the --image argument in the command:
python "count_trees.py" --image "images/your_image.jpeg" --method contour --out "results/your_output.png"

Notes:
- Adjust detection parameters inside count_trees.py for better results:
  - min_area → minimum size of detected objects
  - max_area → maximum size of detected objects
  - HSV ranges → tune for different forest types
