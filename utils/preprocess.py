import cv2
import numpy as np

class Preprocess:
    def __init__(self):
        pass
    
    def hsv_segmentation(self, image):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([30, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        result = cv2.bitwise_and(image, image, mask=mask)
        
        return result
    
    def largest_connected_component(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component_mask = (labels == largest_label).astype("uint8") * 255
        largest_component = cv2.bitwise_and(image, image, mask=largest_component_mask)
        
        return largest_component
    
    def gray_level(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return gray

if __name__ == "__main__":

    test_image_path = r'C:\Users\ouche\Desktop\CV_FP\dataset\label1\image_1.png'  
    image = cv2.imread(test_image_path)
    
    if image is None:
        print(f"無法讀取圖像：{test_image_path}")
        exit()

    processor = Preprocess()
    
    segmented_image = processor.hsv_segmentation(image)
    largest_component_image = processor.largest_connected_component(segmented_image)
    gray_img = processor.gray_level(largest_component_image)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.imshow('Largest Connected Component', largest_component_image)
    cv2.imshow('gray level', gray_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()