# PageScan - Book Page Dewarping Application

A standalone Python desktop application for dewarping curved book pages and correcting perspective distortion from scanned book images. Designed for V-cradle book scanning setups with DSLR cameras like the Canon R8.

## Features

### Image Loading
- Support for multiple formats: JPEG, PNG, TIFF, Canon CR3 RAW
- Batch import from folder
- Drag-and-drop support
- Thumbnail queue sidebar with reordering

### Page Dewarping (Curve Correction)
- Automatic detection of curved distortion along book spine/gutter
- Mesh-based dewarping to flatten curved text lines
- Manual curve adjustment with control points
- Real-time preview of adjustments

### Perspective/Keystone Correction
- Auto-detect page quadrilateral boundaries
- Correct trapezoidal distortion to rectangular output
- Manual corner adjustment with draggable points
- Sub-pixel corner refinement

### Additional Processing
- **Auto-crop**: Remove borders and crop to page boundaries
- **Deskew**: Rotation correction for tilted pages
- **White Balance**: Gray world, white patch, or manual adjustment
- **Exposure/Contrast**: Adjust brightness and contrast
- **Binarization**: Convert to black & white (Otsu, Adaptive, Sauvola methods)

### Preview & Comparison
- Side-by-side before/after view
- Toggle between original and processed
- Zoom and pan controls
- Real-time preview of adjustments

### Export
- Output formats: JPEG, PNG, TIFF
- Quality/compression settings
- Batch export with configurable naming convention
- EXIF metadata preservation option
- Custom output folder selection

### User Interface
- Modern PyQt6 desktop UI
- Dark mode support
- Keyboard shortcuts for common actions
- Progress bar for batch operations

## Installation

### Requirements
- Python 3.9 or higher
- PyQt6
- OpenCV
- NumPy
- SciPy
- Pillow

### Install Dependencies

```bash
pip install -r requirements.txt
```

For Canon CR3 RAW support:
```bash
pip install rawpy
```

For EXIF metadata preservation:
```bash
pip install piexif
```

Or install all optional dependencies:
```bash
pip install -r requirements.txt
pip install rawpy piexif imageio
```

### Install as Package

```bash
pip install -e .
```

## Usage

### Run the Application

```bash
# Using the module
python -m pagescan

# Or using the run script
python run.py

# Or if installed as a package
pagescan
```

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Open Images | Ctrl+O |
| Open Folder | Ctrl+Shift+O |
| Export Current | Ctrl+E |
| Export All | Ctrl+Shift+E |
| Process Current | Ctrl+P |
| Process All | Ctrl+Shift+P |
| Reset | Ctrl+R |
| Previous Image | Left Arrow |
| Next Image | Right Arrow |
| Zoom In | Ctrl++ |
| Zoom Out | Ctrl+- |
| Fit to Window | Ctrl+0 |
| Toggle Comparison | Space |
| Quit | Ctrl+Q |

### Workflow

1. **Load Images**: Drag and drop images or use File > Open to load your scanned book pages
2. **Review Analysis**: The app automatically analyzes each image and suggests corrections
3. **Adjust Settings**: Enable/disable processing steps and fine-tune parameters
4. **Preview**: See real-time preview of corrections
5. **Process**: Click "Process Current" or "Process All" to apply corrections
6. **Export**: Export processed images in your preferred format

## Processing Pipeline

The application processes images through the following pipeline (in order):

1. **Dewarping** - Corrects curved text lines from book spine
2. **Perspective Correction** - Fixes keystone/trapezoidal distortion
3. **Deskew** - Corrects page rotation
4. **Crop** - Removes borders and whitespace
5. **White Balance** - Corrects color cast
6. **Exposure** - Adjusts brightness and contrast
7. **Binarization** - Converts to black and white (optional)

Each step can be enabled/disabled independently.

## Architecture

```
pagescan/
├── core/
│   ├── image_loader.py   # Image loading with RAW support
│   ├── dewarper.py       # Curve detection and dewarping
│   ├── perspective.py    # Perspective correction
│   ├── processor.py      # Main processing pipeline
│   └── exporter.py       # Image export functionality
├── ui/
│   ├── main_window.py    # Main application window
│   ├── thumbnail_panel.py # Image queue sidebar
│   ├── image_viewer.py   # Image display with overlays
│   ├── control_panel.py  # Processing controls
│   └── export_dialog.py  # Export settings dialog
└── __main__.py           # Entry point
```

## Supported Formats

### Input
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tif, .tiff)
- BMP (.bmp)
- Canon RAW (.cr3, .cr2) - requires rawpy
- Nikon RAW (.nef) - requires rawpy
- Sony RAW (.arw) - requires rawpy
- Adobe DNG (.dng) - requires rawpy

### Output
- JPEG (configurable quality 1-100)
- PNG (configurable compression 0-9)
- TIFF (none, LZW, or Deflate compression)

## License

MIT License
