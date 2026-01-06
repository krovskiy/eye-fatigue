#include <opencv2/opencv.hpp>
#include <iostream>

void initializeCamera(cv::VideoCapture& cam);
bool loadCascades(cv::CascadeClassifier& face, cv::CascadeClassifier& eye);
void detectFaces(cv::Mat& gray, std::vector<cv::Rect>& faces, cv::CascadeClassifier& face);
void detectEyes(cv::Mat& gray, cv::Rect& face_roi, std::vector<cv::Rect>& eyes, cv::CascadeClassifier& eye);
bool analyzeEyeState(cv::Mat& eye_roi);  // returns true if closed
void detectBlinks(bool left_closed, bool right_closed);  // tracks state changes
void calculateMetrics();  // blink rate, PERCLOS, etc.
float computeFatigueScore();
void renderUI(cv::Mat& img, float fatigue_score);

bool loadCascades(cv::CascadeClassifier& face, cv::CascadeClassifier& eye)
{
    char buffer[512]; 
    size_t requiredSize = 0;
    getenv_s(&requiredSize, buffer, sizeof(buffer), "OPENCV_DATA_DIR");
    std::string dir(buffer);
    return face.load(dir + "/haarcascade_frontalface_default.xml") && eye.load(dir + "/haarcascade_eye.xml");
} 

void initializeCamera(cv::VideoCapture& cam) {
    cam.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cam.set(cv::CAP_PROP_FPS, 60);
}


void detectFaces(cv::Mat& gray, std::vector<cv::Rect>& faces, cv::CascadeClassifier& face) {
    face.detectMultiScale(gray, faces);
}

void detectEyes(cv::Mat& gray, cv::Rect& face_roi, std::vector<cv::Rect>& eyes, cv::CascadeClassifier& eye) {
    cv::Rect upper(face_roi.x, face_roi.y, face_roi.width, face_roi.height / 2);
    eye.detectMultiScale(gray(upper), eyes, 1.05, 8);
}


void renderUI(cv::Mat& img, float fatigue_score) {
    std::string text = "Fatigue Score: " + std::to_string(fatigue_score);
    cv::putText(img, text, cv::Point(10, img.rows - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 2);
}

bool analyzeEyeState(cv::Mat& eye_roi) {
	cv::Mat thresh;
    cv::threshold(eye_roi, thresh, 30, 255, cv::THRESH_BINARY);
	double white_ratio = (double)cv::countNonZero(thresh) / (eye_roi.rows * eye_roi.cols);
    bool result = (white_ratio >= 0.2) ? false : true;
    return result;
}

int main() {
    cv::CascadeClassifier face, eye;
	loadCascades(face, eye);

    cv::VideoCapture cam(0);
    cv::Mat img, gray;
    std::vector<cv::Rect> faces, eyes;
	initializeCamera(cam);

    while (true) {
        cam >> img;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        
		detectFaces(gray, faces, face);

        for (auto& f : faces) {
            detectEyes(gray, f, eyes, eye);
			cv::Rect upper(f.x, f.y, f.width, f.height / 2);

            for (size_t i = 0; i < eyes.size() && i < 2; i++) {
                auto& e = eyes[i];
                int x = upper.x + e.x;
                int y = upper.y + e.y;

                cv::rectangle(img, cv::Rect(x, y, e.width, e.height), cv::Scalar(0, 255, 0), 2);

                std::string eye_label = (i == 0) ? "Left Eye: " : "Right Eye: ";

                std::cout << "X=" << x << " Y=" << y << " W=" << e.width << " H=" << e.height << std::endl;

                std::string text = eye_label + "X=" + std::to_string(x) + " Y=" + std::to_string(y) +
                    " W=" + std::to_string(e.width) + " H=" + std::to_string(e.height);
                cv::putText(img, text, cv::Point(0, 30 + i * 30), cv::FONT_HERSHEY_PLAIN, 0.6, cv::Scalar(255, 255, 255), 1);
            }
            
        }

        cv::imshow("eye-fatigue demo", img);
        if (cv::waitKey(30) >= 0) break;
    }

    return 0;
}