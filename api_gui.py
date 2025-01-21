import os
import time
import serial
import requests
import numpy as np
from io import BytesIO
from requests.auth import HTTPBasicAuth
import cv2

# 직렬 포트 설정
ser = serial.Serial("/dev/ttyACM0", 9600)

# API endpoint
api_url = "https://suite-endpoint-api-apne2.superb-ai.com/endpoints/23452697-ed62-4afc-9878-0321625cc9e4/inference"
save_path = '/home/rokey/Desktop/0110/test_file'
text_save_path = '/home/rokey/Desktop/0110/results'
os.makedirs(save_path, exist_ok=True)
os.makedirs(text_save_path, exist_ok=True)
image_counter = 1

# 클래스별 색상 매핑
CLASS_COLORS = {
    'Rasberry PICO': (0, 255, 0),
    'Hole': (255, 0, 0),
    'Chipset': (0, 0, 255),
    'Oscillator': (255, 255, 0),
    'Usb': (255, 0, 255),
    'Bootsel': (0, 255, 255)
}

def get_img():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera Error")
        exit(-1)
    ret, img = cam.read()
    cam.release()
    return img

def crop_img(img, size_dict):
    x, y, w, h = size_dict["x"], size_dict["y"], size_dict["width"], size_dict["height"]
    cropped = img[y : y + h, x : x + w]
    if cropped.size == 0:
        print("Cropping failed: resulting image is empty.")
        return img
    return cropped

def create_blank_image(width, height):
    return np.ones((height, width, 3), dtype=np.uint8) * 255

def resize_image_to_height(image, height):
    h, w, _ = image.shape
    if h != height:
        new_width = int(w * (height / h))
        return cv2.resize(image, (new_width, height))
    return image

def create_overlay_with_boxes(img, objects):
    overlay = img.copy()
    for obj in objects:
        r_box = obj['box']
        r_class = obj['class']
        color = CLASS_COLORS.get(r_class, (255, 255, 255))
        start_point = (r_box[0], r_box[1])
        end_point = (r_box[2], r_box[3])
        cv2.rectangle(overlay, start_point, end_point, color, 2)
    return overlay

def create_overlay_with_status(grouped_objects, detected_counts, required_counts):
    blank_image = create_blank_image(400, 300)
    y_offset = 20
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.6
    font_thickness = 1
    for r_class, required_count in required_counts.items():
        color = CLASS_COLORS.get(r_class, (255, 255, 255))
        detected_count = detected_counts.get(r_class, 0)
        if r_class == "Hole":
            status = "Good" if detected_count == 4 else "Warn"
        else:
            status = "Good" if detected_count == 1 else "Warn"
        status_color = (0, 255, 0) if status == "Good" else (255, 0, 0)
        text = f"{r_class}: {status}"
        text_position = (10, y_offset)
        cv2.putText(blank_image, text, text_position, font, font_scale, status_color, font_thickness, cv2.LINE_AA)
        y_offset += 25
    return blank_image

def save_results_as_text(detected_counts, required_counts, count):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    result_file = os.path.join(text_save_path, f"result_{count}.txt")
    with open(result_file, "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        for r_class, required_count in required_counts.items():
            detected_count = detected_counts.get(r_class, 0)
            if r_class == "Hole":
                status = "Good" if detected_count == 4 else "Warn"
            else:
                status = "Good" if detected_count == 1 else "Warn"
            f.write(f"{r_class}: {status}\n")
    print(f"Results saved to {result_file}")

def find_default(res, img, count):
    original_image = img.copy()
    overlay_with_boxes = create_overlay_with_boxes(img, res['objects'])

    required_counts = {
        'Rasberry PICO': 1,
        'Hole': 4,
        'Chipset': 1,
        'Oscillator': 1,
        'Usb': 1,
        'Bootsel': 1
    }
    detected_counts = {key: 0 for key in required_counts}
    grouped_objects = {}

    for obj in res['objects']:
        r_class = obj['class']
        if r_class in detected_counts:
            detected_counts[r_class] += 1
        if r_class not in grouped_objects:
            grouped_objects[r_class] = []
        grouped_objects[r_class].append(obj)

    overlay_with_status = create_overlay_with_status(grouped_objects, detected_counts, required_counts)

    # 모든 이미지를 동일한 높이로 조정
    target_height = original_image.shape[0]
    overlay_with_boxes = resize_image_to_height(overlay_with_boxes, target_height)
    overlay_with_status = resize_image_to_height(overlay_with_status, target_height)

    # 세 개의 이미지를 가로로 붙이기
    concatenated_image = np.concatenate((original_image, overlay_with_boxes, overlay_with_status), axis=1)

    # 이미지 저장
    original_image_path = os.path.join(save_path, f"original_{count}.jpg")
    boxes_image_path = os.path.join(save_path, f"boxes_{count}.jpg")
    status_image_path = os.path.join(save_path, f"status_{count}.jpg")

    cv2.imwrite(original_image_path, original_image)
    cv2.imwrite(boxes_image_path, overlay_with_boxes)
    cv2.imwrite(status_image_path, overlay_with_status)

    print(f"Images saved: {original_image_path}, {boxes_image_path}, {status_image_path}")

    save_results_as_text(detected_counts, required_counts, count)

def inference_request(img, api_url, count):
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()

    try:
        response = requests.post(
            url=api_url,
            headers={'Content-Type': 'image/jpeg'},
            data=img_bytes,
            auth=HTTPBasicAuth("kdt2025_1-11", "lT6FEplNFo28wwTtPNUgh3d6cC6XHOCn1mr7vV9w")
        )
        if response.status_code == 200:
            print("API Response:", response.json())
            find_default(response.json(), img, count)
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")

count = 1
while True:
    data = ser.read()
    print(data)
    if data == b"0":
        img = get_img()
        crop_info = {"x": 200, "y": 149, "width": 300, "height": 215}

        if crop_info is not None:
            img = crop_img(img, crop_info)
            inference_request(img, api_url, count)

        count += 1
        ser.write(b"1")
    else:
        pass

cv2.destroyAllWindows()
