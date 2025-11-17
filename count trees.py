# # import argparse
# # import cv2
# # import numpy as np
# # import imutils
# # from utils import load_image, save_image, draw_keypoints

# # def preprocess_gray(img, blur_ksize=5):
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.equalizeHist(gray)
# #     blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
# #     return blur

# # def blob_detection(img, min_area=100, max_area=5000):
# #     proc = preprocess_gray(img)
# #     params = cv2.SimpleBlobDetector_Params()
# #     params.filterByArea = True
# #     params.minArea = min_area
# #     params.maxArea = max_area
# #     detector = cv2.SimpleBlobDetector_create(params)
# #     keypoints = detector.detect(proc)
# #     return keypoints

# # def color_mask_contours(img, min_area=10, max_area=15000):
# #     """
# #     Detects trees, autumn leaves, and underbrush by combining multiple HSV ranges.
# #     """
# #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# #     # Green trees
# #     lower_green = np.array([60,102,77], dtype=np.uint8)
# #     upper_green = np.array([140,230,179], dtype=np.uint8)
# #     mask_green = cv2.inRange(hsv, lower_green, upper_green)

# #     # Autumn leaves (yellow/orange/red)
# #     lower_autumn = np.array([20,153,128], dtype=np.uint8)
# #     upper_autumn = np.array([50,255,230], dtype=np.uint8)
# #     mask_autumn = cv2.inRange(hsv, lower_autumn, upper_autumn)

# #     # Shadows / underbrush
# #     lower_shadow = np.array([60,51,26], dtype=np.uint8)
# #     upper_shadow = np.array([120,128,102], dtype=np.uint8)
# #     mask_shadow = cv2.inRange(hsv, lower_shadow, upper_shadow)

# #     # Combine all masks
# #     mask = cv2.bitwise_or(mask_green, mask_autumn)
# #     mask = cv2.bitwise_or(mask, mask_shadow)

# #     # Morphological closing to remove small holes
# #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # smaller kernel to separate close objects
# #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# #     # Find contours
# #     cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     cnts = imutils.grab_contours(cnts)

# #     centers = []
# #     for c in cnts:
# #         area = cv2.contourArea(c)
# #         if area < min_area or area > max_area:
# #             continue
# #         M = cv2.moments(c)
# #         if M["m00"] != 0:
# #             cx = int(M["m10"]/M["m00"])
# #             cy = int(M["m01"]/M["m00"])
# #             centers.append((cx, cy))
# #     return centers, mask

# # def draw_contours_centers(img, centers, radius=15, color=(0,255,0), thickness=2):
# #     """
# #     Draws circles on the image for each detected center.
# #     """
# #     out = img.copy()
# #     for (x, y) in centers:
# #         cv2.circle(out, (x, y), radius, color, thickness)
# #     return out

# # def run_blob(image_path, out_path):
# #     img = load_image(image_path)
# #     keypoints = blob_detection(img)
# #     out = draw_keypoints(img, keypoints)
# #     save_image(out_path, out)
# #     print(f"Detected {len(keypoints)} trees (blob method). Output saved to {out_path}")
# #     # Display output window
# #     cv2.imshow("Tree Detection (Blob)", out)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

# # def run_contour(image_path, out_path):
# #     img = load_image(image_path)
# #     centers, mask = color_mask_contours(img)
# #     out = draw_contours_centers(img, centers, radius=15, color=(0,255,0), thickness=2)
# #     save_image(out_path, out)
# #     mask_path = out_path.replace(".png", "_mask.png")
# #     cv2.imwrite(mask_path, mask)
# #     print(f"Detected {len(centers)} trees/leaves/underbrush (contour method). Output saved to {out_path}")

# #     # Display windows
# #     cv2.imshow("Tree Detection (Contour)", out)
# #     cv2.imshow("Mask Preview", mask)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--image", "-i", required=True, help="Path to input image")
# #     parser.add_argument("--out", "-o", default="output.png", help="Output image path")
# #     parser.add_argument("--method", "-m", choices=["blob","contour"], default="blob", help="Detection method")
# #     args = parser.parse_args()

# #     if args.method == "blob":
# #         run_blob(args.image, args.out)
# #     else:
# #         run_contour(args.image, args.out)

# import argparse
# import cv2
# import numpy as np
# import imutils
# from utils import load_image, save_image, draw_keypoints

# def preprocess_gray(img, blur_ksize=5):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
#     blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
#     return blur

# def blob_detection(img, min_area=20, max_area=20000):
#     proc = preprocess_gray(img)
#     params = cv2.SimpleBlobDetector_Params()
#     params.filterByArea = True
#     params.minArea = min_area
#     params.maxArea = max_area

#     # Optional filters for better control
#     params.filterByCircularity = True
#     params.minCircularity = 0.3
#     params.filterByConvexity = False
#     params.filterByInertia = False

#     detector = cv2.SimpleBlobDetector_create(params)
#     keypoints = detector.detect(proc)
#     return keypoints

# def color_mask_contours(img, min_area=10, max_area=15000):
#     """
#     Detects trees, autumn leaves, and underbrush by combining multiple HSV ranges.
#     """
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # Green trees
#     lower_green = np.array([60,102,77], dtype=np.uint8)
#     upper_green = np.array([140,230,179], dtype=np.uint8)
#     mask_green = cv2.inRange(hsv, lower_green, upper_green)

#     lower_brown = np.array([60,102,77], dtype=np.uint8)
#     upper_green = np.array([140,230,179], dtype=np.uint8)
#     mask_green = cv2.inRange(hsv, lower_brown, upper_green)
#     # Autumn leaves (yellow/orange/red)
#     lower_autumn = np.array([20,153,128], dtype=np.uint8)
#     upper_autumn = np.array([50,255,230], dtype=np.uint8)
#     mask_autumn = cv2.inRange(hsv, lower_autumn, upper_autumn)

#     # Shadows / underbrush
#     lower_shadow = np.array([60,51,26], dtype=np.uint8)
#     upper_shadow = np.array([120,128,102], dtype=np.uint8)
#     mask_shadow = cv2.inRange(hsv, lower_shadow, upper_shadow)

#     # Combine all masks
#     mask = cv2.bitwise_or(mask_green, mask_autumn)
#     mask = cv2.bitwise_or(mask, mask_shadow)

#     # Morphological operations: open (remove specks), then close (fill holes)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

#     # Find contours
#     cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)

#     centers = []
#     for c in cnts:
#         area = cv2.contourArea(c)
#         if area < min_area or area > max_area:
#             continue
#         M = cv2.moments(c)
#         if M["m00"] != 0:
#             cx = int(M["m10"]/M["m00"])
#             cy = int(M["m01"]/M["m00"])
#             centers.append((cx, cy))
#     return centers, mask

# def draw_contours_centers(img, centers, radius=15, color=(0,255,0), thickness=2):
#     """
#     Draws circles on the image for each detected center.
#     """
#     out = img.copy()
#     for (x, y) in centers:
#         cv2.circle(out, (x, y), radius, color, thickness)
#     return out

# def run_blob(image_path, out_path):
#     img = load_image(image_path)
#     keypoints = blob_detection(img)
#     out = draw_keypoints(img, keypoints)
#     save_image(out_path, out)
#     print(f"Detected {len(keypoints)} trees (blob method). Output saved to {out_path}")
#     cv2.imshow("Tree Detection (Blob)", out)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def run_contour(image_path, out_path):
#     img = load_image(image_path)
#     centers, mask = color_mask_contours(img)
#     out = draw_contours_centers(img, centers, radius=15, color=(0,255,0), thickness=2)
#     save_image(out_path, out)
#     mask_path = out_path.replace(".png", "_mask.png")
#     cv2.imwrite(mask_path, mask)
#     print(f"Detected {len(centers)} trees/leaves/underbrush (contour method). Output saved to {out_path}")
#     cv2.imshow("Tree Detection (Contour)", out)
#     cv2.imshow("Mask Preview", mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image", "-i", required=True, help="Path to input image")
#     parser.add_argument("--out", "-o", default="output.png", help="Output image path")
#     parser.add_argument("--method", "-m", choices=["blob","contour"], default="blob", help="Detection method")
#     args = parser.parse_args()

#     if args.method == "blob":
#         run_blob(args.image, args.out)
#     else:
#         run_contour(args.image, args.out)
# import cv2
# import numpy as np
# import argparse

# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(blurred)
#     return enhanced

# def detect_trees(image_path, output_path, method):
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Image not found.")
#         return

#     processed = preprocess_image(image)

#     if method == "blob":
#         params = cv2.SimpleBlobDetector_Params()
#         params.filterByArea = True
#         params.minArea = 100
#         params.maxArea = 5000
#         params.filterByCircularity = False
#         params.filterByConvexity = False
#         params.filterByInertia = False
#         params.minThreshold = 10
#         params.maxThreshold = 200

#         detector = cv2.SimpleBlobDetector_create(params)
#         keypoints = detector.detect(processed)

#         output = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 255, 0),
#                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#         cv2.imwrite(output_path, output)
#         print(f"Detected {len(keypoints)} trees (blob method). Output saved to {output_path}")
#     else:
#         print("Error: Unsupported method. Use 'blob'.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image", required=True, help="Path to input image")
#     parser.add_argument("--out", default="result.png", help="Path to save output image")
#     parser.add_argument("--method", default="blob", help="Detection method (blob)")

#     args = parser.parse_args()
#     detect_trees(args.image, args.out, args.method)
import cv2
import numpy as np
import argparse

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced

def detect_trees(image_path, output_path, method):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    processed = preprocess_image(image)

    if method == "blob":
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 5000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minThreshold = 10
        params.maxThreshold = 200

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(processed)

        output = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 255, 0),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imwrite(output_path, output)
        print(f"Detected {len(keypoints)} trees (blob method). Output saved to {output_path}")
    else:
        print("Error: Unsupported method. Use 'blob'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="result.png", help="Path to save output image")
    parser.add_argument("--method", default="blob", help="Detection method (blob)")

    args = parser.parse_args()
    detect_trees(args.image, args.out, args.method)