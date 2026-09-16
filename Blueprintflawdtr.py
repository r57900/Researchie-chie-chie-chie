
### 2. Save as `app.py`

```python
import cv2
import json
import math
import numpy as np
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes


# ============================================================
# Configuration
# ============================================================

MIN_LINE_LENGTH = 30
MAX_LINE_GAP = 15
ANGLE_TOLERANCE = 5

# Lines shorter than this are considered potentially suspicious
SHORT_LINE_LENGTH = 50

# Distance used when checking whether line endpoints are close
ENDPOINT_DISTANCE = 25


# ============================================================
# Utility functions
# ============================================================

def load_image(uploaded_file):
    """
    Load PNG/JPG or convert the first page of a PDF to an image.
    """

    file_bytes = uploaded_file.read()

    if uploaded_file.name.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=200)

        if not pages:
            raise ValueError("Could not read PDF.")

        image = np.array(pages[0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    else:
        image = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not read image.")

    return image


def preprocess(image):
    """
    Convert blueprint to a clean binary edge image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce scanning noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Connect nearby line segments
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return gray, edges


# ============================================================
# Line detection
# ============================================================

def detect_lines(edges):
    """
    Detect blueprint lines using probabilistic Hough transform.
    """

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    if lines is None:
        return []

    result = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        length = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        angle = math.degrees(
            math.atan2(y2 - y1, x2 - x1)
        )

        # Normalize angle to 0-180
        if angle < 0:
            angle += 180

        result.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "length": round(length, 2),
            "angle": round(angle, 2)
        })

    return result


# ============================================================
# Fault detection
# ============================================================

def detect_short_lines(lines):
    """
    Find unusually short line segments.

    These can indicate:
    - broken walls
    - incomplete drafting
    - accidental marks
    """

    faults = []

    for line in lines:

        if line["length"] < SHORT_LINE_LENGTH:

            faults.append({
                "type": "Possible broken/incomplete line",
                "severity": "medium",
                "confidence": 0.65,
                "location": {
                    "x": int((line["x1"] + line["x2"]) / 2),
                    "y": int((line["y1"] + line["y2"]) / 2)
                },
                "details": (
                    f'Line length is {line["length"]} pixels, '
                    f'which is unusually short.'
                )
            })

    return faults


def endpoint_distance(a, b):
    """
    Minimum distance between any endpoints of two lines.
    """

    points_a = [
        (a["x1"], a["y1"]),
        (a["x2"], a["y2"])
    ]

    points_b = [
        (b["x1"], b["y1"]),
        (b["x2"], b["y2"])
    ]

    minimum = float("inf")

    for p1 in points_a:
        for p2 in points_b:

            distance = math.sqrt(
                (p1[0] - p2[0]) ** 2 +
                (p1[1] - p2[1]) ** 2
            )

            minimum = min(minimum, distance)

    return minimum


def detect_possible_gaps(lines):
    """
    Find pairs of nearly aligned lines whose endpoints are
    close together.

    This can indicate a possible gap in a wall or drawing.
    """

    faults = []

    for i in range(len(lines)):

        a = lines[i]

        for j in range(i + 1, len(lines)):

            b = lines[j]

            # Compare only lines with similar orientation
            angle_difference = abs(a["angle"] - b["angle"])

            if angle_difference > ANGLE_TOLERANCE:
                continue

            distance = endpoint_distance(a, b)

            if distance <= ENDPOINT_DISTANCE:

                # Ignore extremely short lines
                if (
                    a["length"] < MIN_LINE_LENGTH
                    or b["length"] < MIN_LINE_LENGTH
                ):
                    continue

                x = int(
                    (
                        a["x1"]
                        + a["x2"]
                        + b["x1"]
                        + b["x2"]
                    ) / 4
                )

                y = int(
                    (
                        a["y1"]
                        + a["y2"]
                        + b["y1"]
                        + b["y2"]
                    ) / 4
                )

                faults.append({
                    "type": "Possible line/wall gap",
                    "severity": "high",
                    "confidence": 0.72,
                    "location": {
                        "x": x,
                        "y": y
                    },
                    "details": (
                        f"Two similarly oriented lines have "
                        f"a {round(distance, 1)} pixel endpoint gap."
                    )
                })

    return faults


def detect_unusual_angles(lines):
    """
    Detect lines that are not approximately horizontal or vertical.

    Useful for finding accidental/skewed drafting marks.
    """

    faults = []

    for line in lines:

        angle = line["angle"]

        horizontal = min(
            abs(angle),
            abs(angle - 180)
        )

        vertical = abs(angle - 90)

        if (
            horizontal > ANGLE_TOLERANCE
            and vertical > ANGLE_TOLERANCE
            and line["length"] > 80
        ):

            faults.append({
                "type": "Unusual line angle",
                "severity": "low",
                "confidence": 0.55,
                "location": {
                    "x": int((line["x1"] + line["x2"]) / 2),
                    "y": int((line["y1"] + line["y2"]) / 2)
                },
                "details": (
                    f'Line angle is {line["angle"]} degrees.'
                )
            })

    return faults


# ============================================================
# Annotation
# ============================================================

def annotate_image(image, faults, lines):
    """
    Draw detected faults and detected lines onto blueprint.
    """

    output = image.copy()

    # Draw detected lines in blue
    for line in lines:

        cv2.line(
            output,
            (line["x1"], line["y1"]),
            (line["x2"], line["y2"]),
            (255, 150, 0),
            1
        )

    # Fault colors
    colors = {
        "high": (0, 0, 255),       # red
        "medium": (0, 165, 255),   # orange
        "low": (0, 255, 255)       # yellow
    }

    for index, fault in enumerate(faults, start=1):

        x = fault["location"]["x"]
        y = fault["location"]["y"]

        color = colors.get(
            fault["severity"],
            (0, 0, 255)
        )

        # Highlight location
        cv2.circle(
            output,
            (x, y),
            15,
            color,
            3
        )

        cv2.putText(
            output,
            str(index),
            (x + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return output


# ============================================================
# Main analysis pipeline
# ============================================================

def analyze_blueprint(image):

    gray, edges = preprocess(image)

    lines = detect_lines(edges)

    faults = []

    # Fault checks
    faults.extend(
        detect_short_lines(lines)
    )

    faults.extend(
        detect_possible_gaps(lines)
    )

    faults.extend(
        detect_unusual_angles(lines)
    )

    annotated = annotate_image(
        image,
        faults,
        lines
    )

    return {
        "lines": lines,
        "faults": faults,
        "annotated": annotated,
        "edges": edges
    }


# ============================================================
# Streamlit interface
# ============================================================

st.set_page_config(
    page_title="AI Blueprint Fault Detector",
    layout="wide"
)

st.title("🏗️ AI Blueprint Fault Detector")

st.write(
    """
    Upload an architectural or engineering blueprint.
    The system analyzes the drawing for potentially suspicious
    linework, gaps, and unusual geometry.
    """
)

uploaded_file = st.file_uploader(
    "Upload Blueprint",
    type=[
        "png",
        "jpg",
        "jpeg",
        "pdf"
    ]
)

if uploaded_file:

    try:

        image = load_image(uploaded_file)

        st.subheader("Original Blueprint")

        display_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            display_image,
            use_container_width=True
        )

        with st.spinner("Analyzing blueprint..."):

            result = analyze_blueprint(image)

        faults = result["faults"]

        st.subheader("Analysis Results")

        col1, col2, col3 = st.columns(3)

        high = sum(
            1 for f in faults
            if f["severity"] == "high"
        )

        medium = sum(
            1 for f in faults
            if f["severity"] == "medium"
        )

        low = sum(
            1 for f in faults
            if f["severity"] == "low"
        )

        col1.metric(
            "High",
            high
        )

        col2.metric(
            "Medium",
            medium
        )

        col3.metric(
            "Low",
            low
        )

        # Annotated image
        st.subheader("Detected Faults")

        annotated_rgb = cv2.cvtColor(
            result["annotated"],
            cv2.COLOR_BGR2RGB
        )

        st.image(
            annotated_rgb,
            use_container_width=True
        )

        # Fault list
        if faults:

            st.subheader("Fault Report")

            for i, fault in enumerate(
                faults,
                start=1
            ):

                st.write(
                    f"""
                    **#{i} — {fault['type']}**

                    Severity: **{fault['severity']}**

                    Confidence: **{fault['confidence'] * 100:.0f}%**

                    Location:
                    ({fault['location']['x']},
                    {fault['location']['y']})

                    {fault['details']}
                    """
                )

        else:

            st.success(
                "No suspicious linework was detected."
            )

        # JSON report
        report = {
            "blueprint": uploaded_file.name,
            "total_lines_detected": len(
                result["lines"]
            ),
            "total_faults": len(faults),
            "faults": faults
        }

        json_data = json.dumps(
            report,
            indent=4
        )

        st.download_button(
            label="Download Fault Report",
            data=json_data,
            file_name="blueprint_fault_report.json",
            mime="application/json"
        )

    except Exception as e:

        st.error(
            f"Could not analyze blueprint: {str(e)}"
        )
```

