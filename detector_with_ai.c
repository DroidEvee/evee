#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// Define the Trash Detection Class.
class TrashDetection {

public:
  TrashDetection(const String& api_key, const String& project_id) {
    google_cloud_vision_api_ = GoogleCloudVisionAPI(api_key, project_id);
  }

  // Detect trash in an image.
  vector<Rect> detect_trash(const Mat& image) {
    // Convert the image to a buffer.
    vector<uchar> image_buffer;
    cv::imencode(".jpg", image, image_buffer);

    // Create a request object.
    GoogleCloudVisionAPIRequest request;
    request.set_image_buffer(image_buffer);

    // Send the request.
    GoogleCloudVisionAPIResponse response = google_cloud_vision_api_.send_request(request);

    // Get the bounding boxes of the detected trash.
    vector<Rect> trash_bounding_boxes;
    for (const auto& object : response.objects()) {
      if (object.name() == "Trash") {
        trash_bounding_boxes.push_back(object.bounding_box());
      }
    }

    return trash_bounding_boxes;
  }

private:
  GoogleCloudVisionAPI google_cloud_vision_api_;
};

// Main function.
int main() {
  // Initialize the camera.
  VideoCapture camera(0);
  if (!camera.isOpened()) {
    cout << "Could not open camera." << endl;
    return -1;
  }

  // Get the API key and project ID from the environment variables.
  String api_key = getenv("GOOGLE_CLOUD_VISION_API_KEY");
  String project_id = getenv("GOOGLE_CLOUD_VISION_PROJECT_ID");

  // Create a TrashDetection object.
  TrashDetection trash_detection(api_key, project_id);

  // Start a loop to read and process frames from the camera.
  while (true) {
    // Read a frame from the camera.
    Mat image;
    camera >> image;

    // Detect trash in the image.
    vector<Rect> trash_bounding_boxes = trash_detection.detect_trash(image);

    // Draw the bounding boxes of the detected trash.
    for (const auto& bounding_box : trash_bounding_boxes) {
      rectangle(image, bounding_box, Scalar(0, 255, 0), 2);
    }

    // Display the image.
    imshow("Trash Detection", image);

    // Wait for a key press.
    int key = waitKey(1);
    if (key == 27) {
      break;
    }
  }

  // Close the camera.
  camera.release();

  return 0;
}
