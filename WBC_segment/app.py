from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
from matplotlib import pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

def watershed_segmentation(image_content):
   
    nparr = np.frombuffer(image_content.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    median_filtered_image = cv2.medianBlur(binary_image, ksize=5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_image = cv2.morphologyEx(median_filtered_image, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened_image, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opened_image, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown_region = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers[unknown_region == 255] = 0

    segmented_cells = cv2.watershed(image, markers)
    image[segmented_cells == -1] = [255, 0, 0]

    num_cells_detected = np.max(segmented_cells) - 1

    fig, ax = plt.subplots(figsize=(7.5, 6))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
   # plt.title('Original Image')
    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    

    # Convert the plot to a base64-encoded image
    buffered = BytesIO()
    plt.savefig(buffered, format="png")
    plt.close()

    original_image_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Display the segmented image in a separate plot
    plt.imshow(segmented_cells, cmap='jet')
   # plt.title('Segmented Image')
    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    buffered_segmented = BytesIO()
    plt.savefig(buffered_segmented, format="png")
    plt.close()

    segmented_image_str = "data:image/png;base64," + base64.b64encode(buffered_segmented.getvalue()).decode("utf-8")

    return original_image_str, segmented_image_str, num_cells_detected

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['image']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            try:
                original_image_str, segmented_image_str, num_cells_detected = watershed_segmentation(file)
                return render_template('index.html', original_image=original_image_str, segmented_image=segmented_image_str, num_cells=num_cells_detected)
            except Exception as e:
                return render_template('index.html', error=f'Error processing the image: {str(e)}')

    return render_template('index.html', original_image=None, segmented_image=None, num_cells=None, error=None)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
