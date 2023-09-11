#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

// Define the Trash Detection Class.
class TrashDetection {

public:
  TrashDetection(const String& trash_classifier_path) {
    trash_classifier = cv::CascadeClassifier(trash_classifier_path);
  }

  // Detect trash in an image.
  vector<Rect> detect_trash(const Mat& image) {
    // Convert the image to grayscale.
    Mat grayscale_image;
    cv::cvtColor(image, grayscale_image, COLOR_BGR2GRAY);

    // Apply thresholding to the image to binarize it.
    Mat thresholded_image;
    cv::threshold(grayscale_image, thresholded_image, 127, 255, THRESH_BINARY);

    // Find the contours in the thresholded image.
    vector<vector<Point>> contours;
    cv::findContours(thresholded_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Discard small contours.
    vector<vector<Point>> filtered_contours;
    for (const auto& contour : contours) {
      if (cv::contourArea(contour) > 100) {
        filtered_contours.push_back(contour);
      }
    }

    // Classify the contours as trash or not trash.
    vector<Rect> trash_bounding_boxes;
    for (const auto& contour : filtered_contours) {
      bool is_trash = trash_classifier.detectMultiScale(contour, false, 1.1, 3);
      if (is_trash) {
        trash_bounding_boxes.push_back(cv::boundingRect(contour));
      }
    }

    return trash_bounding_boxes;
  }

private:
  cv::CascadeClassifier trash_classifier;
};

// Main function.
int main() {
  // Load the trash classifier.
  String trash_classifier_path = "trash_classifier.xml";
  TrashDetection trash_detection(trash_classifier_path);

  // Read the image.
  Mat image = imread("image.jpg");

  // Detect trash in the image.
  vector<Rect> trash_bounding_boxes = trash_detection.detect_trash(image);

  // Draw the bounding boxes of the detected trash.
  for (const auto& bounding_box : trash_bounding_boxes) {
    cv::rectangle(image, bounding_box, Scalar(0, 255, 0), 2);
  }

  // Display the image.
  imshow("Trash Detection", image);
  waitKey(0);

  return 0;
}
