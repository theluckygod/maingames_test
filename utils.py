
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def detect_circle_avatar(org_img: Image, padding=5, min_radius=None, max_radius=None, param1=50):
    img = np.array(org_img)
    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.bilateralFilter(gray, 11, 30, 30)

    if min_radius is None:
        min_radius = img.shape[0] // 4
    if max_radius is None:
        max_radius = img.shape[0] // 2

    # tune circles size
    detected_circles = cv2.HoughCircles(gray_blurred,
                            cv2.HOUGH_GRADIENT, 1,
                            param1=param1,
                            param2=30,
                            minDist=img.shape[1] // 5,
                            minRadius=min_radius,
                            maxRadius=max_radius)

    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
            if a - r > img.shape[1] // 3:
                continue
            img = img[max(b-r, 0):min(b+r+padding, img.shape[0]), max(a-r, 0):min(a+r+padding, img.shape[1])]
            return Image.fromarray(img), True
    return org_img, False


def detect_rectangle_avatar(org_img: Image, padding=5):
    img = np.array(org_img)

    edges = cv2.Canny(img, 300, 700) 
    
    # Find contours in the edges image 
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    # Iterate over each contour 
    for contour in contours: 
        # Approximate the contour to a polygon 
        polygon = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True) 
    
        # Check if the polygon has 4 sides 
        if len(polygon) == 4: 
            # Draw the rectangle on the image 
            x, y, w, h = cv2.boundingRect(polygon)
            if h < img.shape[0] // 2:
                continue
            img = img[max(y-padding, 0):min(y+h+padding, img.shape[0]), max(x-padding, 0):min(x+w+padding, img.shape[1])]
            return Image.fromarray(img), True

    return org_img, False


def add_corners(im, rad=65, bg=True, bgCol='black', bgPix=2):
    bg_im = Image.new('RGB', tuple(x+(bgPix*2) for x in im.size), bgCol)
    ims = [im if not bg else im, bg_im]
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    for i in ims:
        alpha = Image.new('L', i.size, 'white')
        w, h = i.size
        alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
        alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
        alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
        alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
        i.putalpha(alpha)
    bg_im.paste(im, (bgPix, bgPix), im)
    return im if not bg else bg_im


def process_train_image(img, wsize=128, hsize=128):
    # img = img.filter(ImageFilter.GaussianBlur(1))
    # img = img.filter(ImageFilter.BLUR)
    return post_process(img, wsize, hsize)


def process_test_image(img, wsize=128, hsize=128):
    img, detected = detect_circle_avatar(img, min_radius=20, padding=5)
    # if not detected:
    #     img, detected = detect_rectangle_avatar(img)
    if not detected:
        img, detected = detect_circle_avatar(img, min_radius=15, padding=2)
    if not detected:
        img, detected = detect_circle_avatar(img, min_radius=20, padding=2, param1=70)
    if not detected:
        # Setting the points for cropped image
        left = int(0.08 * img.size[0])
        top = 0
        right = left + min(img.size[0] // 3, img.size[1] * 1.25)
        bottom = img.size[1]

        # Cropped image of above dimension
        # (It will not change original image)
        img = img.crop((left, top, right, bottom))
    
    return post_process(img, wsize, hsize)


def post_process(img, wsize=128, hsize=128):
    # img = img.resize((wsize, hsize), Image.Resampling.LANCZOS)
    # img = add_corners(img)
    img = img.resize((wsize, hsize), Image.Resampling.LANCZOS)
    return img.convert('RGB')