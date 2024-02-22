import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class ColorSegmentationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.image = None
        self.lower_color = np.array([0, 0, 0], dtype=np.uint8)
        self.upper_color = np.array([180, 255, 255], dtype=np.uint8)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Original Image Label
        self.original_label = QLabel(self)
        self.original_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.original_label)

        # Segmented Image Label
        self.segmented_label = QLabel(self)
        self.segmented_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.segmented_label)

        # Button to choose an image
        choose_image_button = QPushButton('Choose Image', self)
        choose_image_button.clicked.connect(self.choose_image)
        layout.addWidget(choose_image_button)

        # Sliders for H_MIN, S_MIN, V_MIN, H_MAX, S_MAX
        sliders_layout = QHBoxLayout()
        self.create_slider('H_MIN', sliders_layout, 0, 180)
        self.create_slider('S_MIN', sliders_layout, 0, 255)
        self.create_slider('V_MIN', sliders_layout, 0, 255)
        self.create_slider('H_MAX', sliders_layout, 0, 180)
        self.create_slider('S_MAX', sliders_layout, 0, 255)
        self.create_slider('V_MAX', sliders_layout, 0, 255)
        layout.addLayout(sliders_layout)

        # Save Mask Button
        save_button = QPushButton('Save Mask', self)
        save_button.clicked.connect(self.save_mask)
        layout.addWidget(save_button)

        self.setLayout(layout)
        self.update_images()

    def create_slider(self, label_text, layout, min_val, max_val):
        label = QLabel(f'{label_text}:', self)
        slider = QSlider(Qt.Horizontal)
        slider.setObjectName(label_text)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.valueChanged.connect(self.slider_value_changed)
        layout.addWidget(label)
        layout.addWidget(slider)

    def slider_value_changed(self):
        sender = self.sender()
        if isinstance(sender, QSlider):
            value = sender.value()
            if 'H_MIN' in sender.objectName():
                self.lower_color[0] = value
            elif 'S_MIN' in sender.objectName():
                self.lower_color[1] = value
            elif 'V_MIN' in sender.objectName():
                self.lower_color[2] = value
            elif 'H_MAX' in sender.objectName():
                self.upper_color[0] = value
            elif 'S_MAX' in sender.objectName():
                self.upper_color[1] = value
            elif 'V_MAX' in sender.objectName():
                self.upper_color[2] = value

        self.update_images()

    def update_images(self):
        if self.image is not None:
            # Perform color segmentation
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image, self.lower_color, self.upper_color)
            segmented_image = cv2.bitwise_and(self.image, self.image, mask=mask)

            # Display images in labels
            self.display_image(self.original_label, self.image)
            self.display_image(self.segmented_label, segmented_image)

            # Save the mask for later use
            self.mask = mask

    def display_image(self, label, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)

    def choose_image(self):
        # Use QFileDialog to get the image file path
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Choose Image', '', 'Images (*.png *.jpg *.bmp);;All Files (*)')

        if file_path:
            # Load the chosen image
            self.image = cv2.imread(file_path)
            self.image= cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.update_images()

    def save_mask(self):
        if hasattr(self, 'mask'):
            # Use QFileDialog to get the save path
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, 'Save Mask', '', 'Images (*.png);;All Files (*)')

            if file_path:
                # Save the mask as an image
                cv2.imwrite(file_path, self.mask)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    segmentation_app = ColorSegmentationApp()
    segmentation_app.show()
    sys.exit(app.exec_())
