# Referência
#ARZOLLA, V. M. Algoritmo de detecção visual de faixas de trânsito e controle lateral
#para veículo autônomo. GitHub, 2022. Disponível em: <https://github.com/arzolla/
#TCC_VMA_2022>. Acesso em: 05 JUL 2022.

import numpy as np
import cv2
import matplotlib.pyplot as plt

class Lines:
    def __init__(self):
       self.all_lines = 0
       self.left_lines = 0
       self.left_lines_buff = 0
       self.right_lines = 0
       self.right_lines_buff = 0
       self.first_try = 1

    def filter_by_angle(self, sin_max):
        ok_lines = []
        if self.all_lines is not None:
            for line in self.all_lines:
                rho, theta = line[0]
                sin_theta = np.sin(theta)
                if abs(sin_theta) < sin_max :
                    ok_lines.append(np.array(line))
        self.all_lines = np.array(ok_lines)

    def sort_sides_by_roi(self, low_left, high_left, low_right, high_right):
        left = []
        right = []
        if self.all_lines is not None:
            for line in self.all_lines:
                rho, theta = line[0]
                base = rho*(1/np.cos(theta)) - 500*np.sin(theta)*np.cos(theta)
                top = rho*(1/np.cos(theta))
                if base > low_left and base < high_left and top > low_left and top < high_left: 
                    left.append(np.array(line))
                elif base > low_right and base < high_right and top > low_right and top < high_right:
                    right.append(np.array(line))

        self.left_lines = np.array(left)
        self.right_lines = np.array(right)

    def sort_sides_by_angle(self):
        left = []
        right = []
        if self.all_lines is not None:
            for line in self.all_lines:
                rho, theta = line[0]
                if theta < np.pi/2:
                    left.append(np.array(line))
                if theta > np.pi/2:
                    right.append(np.array(line))

        self.left_lines = np.array(left)
        self.right_lines = np.array(right)

    def buffer(self):
        if self.first_try:
            self.left_lines_buff = self.left_lines.copy()
            self.right_lines_buff = self.right_lines.copy()
            self.first_try = 0

        if np.size(self.left_lines) == 0:
            self.left_lines = self.left_lines_buff.copy()
        else:
            self.left_lines_buff = self.left_lines.copy()
    
        if np.size(self.right_lines) == 0:
            self.right_lines = self.right_lines_buff.copy()
        else:
            self.right_lines_buff = self.right_lines.copy()

    def get_average_line(self):
        if self.left_lines is not None and self.right_lines is not None:
            avg_left = [np.mean(self.left_lines, axis=0, dtype=np.float32)]
            avg_right = [np.mean(self.right_lines, axis=0, dtype=np.float32)]
            avg_left = np.reshape(np.array(avg_left), (1,2))
            avg_right = np.reshape(np.array(avg_right), (1,2))
            return avg_left, avg_right
        return np.zeros((1,2)), np.zeros((1,2))
    
class Accumulator:
    def __init__(self, accum_max_size):
        self.accum = []
        self.accum_max_size = accum_max_size

    def accumulate(self, new_item):
        if new_item is not None:
            self.accum.append(new_item)
            if len(self.accum) > self.accum_max_size:
                self.accum.pop(0)
            return self.accum
        return self.accum
    
    def accumulator_average(self):
        res = (sum(self.accum)/len(self.accum))
        return res
        
def skeletize_image(img, rect_size = 3):
    # Step 1: Create an empty skeleton
    skel = np.zeros(img.shape, np.uint8)
    # Get a Rect Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (rect_size,1))
    # If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    while cv2.countNonZero(img) != 0:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
    return skel

def hough_transform(image):
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 40  # minimal of votes
    line_segments =cv2.HoughLines(image, rho, angle, min_threshold, None, 0, 0)
    return line_segments

def normalize_hough(lines):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if rho < 0:
                rho = (-rho)
                theta = (theta - np.pi)
            line[0] = rho, theta
    return lines

def vanish_point(lines):
    A = np.zeros((2,2))
    p = np.zeros((2,1))
    for i in range(2):
        A[i,0] = np.cos(lines[i,1])
        A[i,1] = np.sin(lines[i,1])
        p[i,0] = lines[i,0]
    A_pinv = np.linalg.pinv(A)
    v = np.dot(A_pinv,p)
    return v

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            cv2.line(frame, pt1, pt2, line_color, line_width, cv2.LINE_AA)

