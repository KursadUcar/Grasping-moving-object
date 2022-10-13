"""
author      = "Kürşad UÇAR"
version     = "1.0"
description = "Grasping moving object with a robot arm."
December 2022
"""
# import packages
from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
from time import sleep
import time
import serial
import os
import argparse
from math import atan2, cos, sin, sqrt, pi, degrees
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
# to write/read to Arduiono for robot arm control
def write_read(x):
    arduino.write(bytes(x, 'utf - 8'))
    sleep(1)
    data = arduino.readline()
    return data

# drawing axis for PCA
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
# getting orientation with PCA
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype = np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    return angle, cntr[0], cntr[1]

###########################################################
# to run this code the experimental setup must be same as in our paper, otherwise it is possible to change these variables
# cameras are 640 pixel x 480 pixel
cam_long_edge_p = 640 
cam_short_edge_p = 480
#the cameras see 40 cm x 30 cm area
cam_long_edge_cm = 40
cam_short_edge_cm = 30
# distance between conveyor and cameras ara 65 cm(side cam) and 72 cm(top cam)
top_cam_dis = 72
side_cam_dis = 65
# side camera sees 29 cm at conveyor beginning side
side_cam_see = 29
# template size/2 (pixel)
temp_size = 50 

###########################################################
#SET UPS
arduino = serial.Serial(port = 'COM11', baudrate = 115200, timeout = .1)
sleep(2)
value = write_read("-") # Starting communication with Arduino
ANN_hor = load_model('ANN_hor.h5') # ANN to calculate theta2, theta3, and theta4 for horizontol objects
ANN_ver = load_model('ANN_ver.h5') # ANN to calculate theta2, theta3, and theta4 for vertical objects
cap_top = cv2.VideoCapture(0,cv2.CAP_DSHOW) # starting top camera
cap_side = cv2.VideoCapture(1,cv2.CAP_DSHOW) # starting side camera
cap_top.set(3, cam_short_edge_p)
cap_top.set(4, cam_long_edge_p)
cap_side.set(3, cam_short_edge_p)
cap_side.set(4, cam_long_edge_p)
obj_detected = 0 # for controling object is detected or not
sleep(1) 
meth = 'cv2.TM_SQDIFF_NORMED' # template matching method
method = eval(meth)
ret, frame = cap_top.read()
frame_width = frame.shape[1]
frame_height = frame.shape[0]
ret_side, frame_side = cap_side.read()
frame_width_side = frame_side.shape[1]
frame_height_side = frame_side.shape[0]
frame_width_top = frame.shape[1]
frame_height_top = frame.shape[0]


# labels
labels = ["bottle","j. box"]
# YOLO boxs' color
colors = ["0, 255, 255","255,0,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
ids_list = []
centers_matched = 0 # controller PCA and YOLO box center is matching

while(cap_top.isOpened() and not ids_list):
  start = time.time()  # start to time for speed calculation
  ret, frame = cap_top.read()
  ret_side, frame_side = cap_side.read()
  ret, frame1 = cap_top.read()
  frame3 = frame1

  # image2blob
  frame_blob1 = cv2.dnn.blobFromImage(frame1, 1 / 255, (416, 416), swapRB = True, crop = False)
  model = cv2.dnn.readNetFromDarknet("object_detect.cfg", "hor_obj.weights") 
  #choose output layers
  layers = model.getLayerNames()
  output_layer = [layers[layer[0] - 1] for layer in model.getUnconnectedOutLayers()]
  #run a network with tensor
  model.setInput(frame_blob1)
  detection_layers1 = model.forward(output_layer)
  #non max supprision operation - 1
  ids_list1 = []
  boxes_list1 = []
  confidences_list1 = []
  #end of op.1
#horizontal object detection
# analysis detection layers
  for detection_layer1 in detection_layers1:
    for object_detection1 in detection_layer1: # firts 5 elements are bounding box, remain is estimaion
      scores1 = object_detection1[5:]
      prediction_id1 = np.argmax(scores1)
      confidence1 = scores1[prediction_id1]

      if confidence1 > 0.30: # threshold value for boxes
        label1 = labels[prediction_id1]
        bounding_box1 = object_detection1[0:4]* np.array([frame_width, frame_height, frame_width, frame_height])
        (box_center_x1, box_center_y1, box_width1, box_height1) = bounding_box1.astype("int")
        start_x1 = int(box_center_x1 - (box_width1 / 2))
        start_y1 = int(box_center_y1 - (box_height1 / 2))
        #non max supprision operation - 2
        ids_list1.append(prediction_id1)
        boxes_list1.append([start_x1, start_y1, int(box_width1), int(box_height1)])
        confidences_list1.append(float(confidence1))
      #end of op.2
      #non max supprision operation - 3
  max_ids1 = cv2.dnn.NMSBoxes(boxes_list1, confidences_list1, 0.5, 0.4)

  for max_id1 in max_ids1:
    max_class_id1 = max_id1[0]
    box1 = boxes_list1[max_class_id1]

    start_x1 = box1[0]
    start_y1 = box1[1]
    box_width1 = box1[2]
    box_heighyt1 = box1[3]

    prediction_id1 = ids_list1[max_class_id1]
    label1 = labels[prediction_id1]
    confidence1 = confidences_list1[max_class_id1]
      #end of op.3
    end_x1 = start_x1 + box_width1
    end_y1 = start_y1 + box_height1
##get orientation
    obj_det_im = frame1 # object detected image
    gray1 = cv2.cvtColor(obj_det_im, cv2.COLOR_BGR2GRAY)
    bw1 = cv2.adaptiveThreshold(gray1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,101,2)
    contours1, _ = cv2.findContours(bw1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i1, c1 in enumerate(contours1):
        # Calculate the area of each contour
        area1 = cv2.contourArea(c1)
        # Ignore contours that are too small or too large
        if area1 < 1 * 1e4 or 5 * 1e4 < area1:
            continue
        
        # Draw each contour only for visualisation purposes	
        cv2.drawContours(obj_det_im, contours1, i1, (0, 0, 255), 2)
        # Find the orientation of each shape
        angles1, centerx1, centery1 = getOrientation(c1, obj_det_im)
        center_difference1 = sqrt((centerx1 - box_center_x1) ** 2 + (centery1 - box_center_y1) ** 2) # Difference between YOLO box and PCA center
        if center_difference1 < 50: #  Searching for nearest PCA center to YOLO box (50 threshold)
            cenx1 = centerx1
            ceny1 = centery1
            angle1 = degrees(angles1)
            centers_matched = 1 # controling PCA and YOLO box center is matching
        else:
            centers_matched = 0
    if centers_matched == 1:
        # creating template for speed calculation
        # If the object center is closer than 50px to any edge of the image, a smaller template will be created
        if ceny1 < temp_size:
            if cenx1 < temp_size:
               template = frame3[1:ceny1 + temp_size,1:cenx1 + temp_size,:]
               stat = 0
            elif cenx1 > cam_long_edge_p - temp_size:
               template = frame3[1:ceny1 + temp_size,cenx1 - temp_size:cam_long_edge_p,:]
               stat = cenx1 - temp_size
            else:
                template = frame3[1:ceny1 + temp_size,cenx1 - temp_size:cenx1 + temp_size,:]
                stat = cenx1 - temp_size
        elif ceny1 > cam_short_edge_p - temp_size:
            if cenx1 < temp_size:
               stat = 0
               template = frame3[ceny1 - temp_size:cam_short_edge_p,1:cenx1 + temp_size,:]
            elif cenx1 > cam_long_edge_p - temp_size:
               template = frame3[ceny1 - temp_size:cam_short_edge_p,cenx1 - temp_size:cam_long_edge_p,:]
               stat = cenx1 - temp_size
            else:
                template = frame3[ceny1 - temp_size:cam_short_edge_p,cenx1 - temp_size:cenx1 + temp_size,:]
                stat = cenx1 - temp_size
        else:
            if cenx1 < temp_size:
               stat = 0
               template = frame3[ceny1 - temp_size:ceny1 + temp_size,1:cenx1 + temp_size,:]
            elif cenx1 > cam_long_edge_p - temp_size:
               template = frame3[ceny1 - temp_size:ceny1 + temp_size,cenx1 - temp_size:cam_long_edge_p,:]
               stat = cenx1 - temp_size
            else:
                template = frame3[ceny1 - temp_size:ceny1 + temp_size,cenx1 - temp_size:cenx1 + temp_size,:]
                stat = cenx1 - temp_size
        wt = template.shape[1]
        ht = template.shape[0]
    
        cv2.imshow('temp', template)
        obj_detected = 1

        box_color1 = colors[prediction_id1]
        box_color1 = [int(each) for each in box_color1]
    
        label1 = "{}: {:.2f}%" .format(label1, confidence1 * 100)
        print("predicted object {}" . format(label1))
        cv2.rectangle(frame1, (start_x1, start_y1), (end_x1, end_y1), box_color1, 2)
        cv2.putText(frame1, label1, (start_x1, start_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color1, 2)

  # if horizontal object is not found vertical one will be searched
  
  if not prediction_id1:
      
      ret_side, frame_side1 = cap_side.read()
      frame_side = cv2.flip(frame_side1,1)
      frame_blob_side = cv2.dnn.blobFromImage(frame_side, 1 / 255, (416, 416), swapRB = True, crop = False)
      model_side = cv2.dnn.readNetFromDarknet("object_detect.cfg", "ver_obj_side.weights")
      start_top = time.time()
      ret_top, frame_top = cap_top.read()
      frame_blob_top = cv2.dnn.blobFromImage(frame_top, 1 / 255, (416, 416), swapRB = True, crop = False)
      model_top = cv2.dnn.readNetFromDarknet("object_detect.cfg", "ver_obj_top.weights")
  
      layers_side = model_side.getLayerNames()
      output_layer_side = [layers_side[layer_side[0] - 1] for layer_side in model_side.getUnconnectedOutLayers()]
      layers_top = model_top.getLayerNames()
      output_layer_top = [layers_top[layer_top[0] - 1] for layer_top in model_top.getUnconnectedOutLayers()]
       
      model_side.setInput(frame_blob_side)
      detection_layers_side = model_side.forward(output_layer_side)
      
      model_top.setInput(frame_blob_top)
      detection_layers_top = model_top.forward(output_layer_top)
     
      ids_list_side = []
      boxes_list_side = []
      confidences_list_side =[]
      ids_list_top = []
      boxes_list_top = []
      confidences_list_top = []
      
      for detection_layer_side in detection_layers_side:
        for object_detection_side in detection_layer_side:
            scores_side = object_detection_side[5:]
            prediction_id_side = np.argmax(scores_side)
            confidence_side = scores_side[prediction_id_side]

            if confidence_side > 0.30: 
                label_side = labels[prediction_id_side]
                bounding_box_side = object_detection_side[0:4] *  np.array([frame_width_side, frame_height_side, frame_width_side, frame_height_side])
                (box_center_x_side, box_center_y_side, box_width_side, box_height_side) = bounding_box_side.astype("int")
                start_x_side = int(box_center_x_side - (box_width_side / 2))
                start_y_side = int(box_center_y_side - (box_height_side / 2))
                
                ids_list_side.append(prediction_id_side)
                boxes_list_side.append([start_x_side, start_y_side, int(box_width_side), int(box_height_side)])
                confidences_list_side.append(float(confidence_side))
           
      ver_detected = 0 # vertical object detecter controller
      for detection_layer_top in detection_layers_top:
        for object_detection_top in detection_layer_top: 
            scores_top = object_detection_top[5:]
            prediction_id_top = np.argmax(scores_top)
            confidence_top = scores_top[prediction_id_top]
            
            if confidence_top > 0.30:
                label_top = labels[prediction_id_top]
                bounding_box_top = object_detection_top[0:4] *  np.array([frame_width_top, frame_height_top, frame_width_top, frame_height_top])
                (box_center_x_top, box_center_y_top, box_width_top, box_height_top) = bounding_box_top.astype("int")
                start_x_top = int(box_center_x_top - (box_width_top / 2))
                start_y_top = int(box_center_y_top - (box_height_top / 2))
                
                ids_list_top.append(prediction_id_top)
                boxes_list_top.append([start_x_top, start_y_top, int(box_width_top), int(box_height_top)])
                confidences_list_top.append(float(confidence_top))
               
                ver_detected = 1
      max_ids_side = cv2.dnn.NMSBoxes(boxes_list_side, confidences_list_side, 0.5, 0.4)
      max_ids_top = cv2.dnn.NMSBoxes(boxes_list_top, confidences_list_top, 0.5, 0.4)
      
      for max_id_side in max_ids_side:
        max_class_id_side = max_id_side[0]
        box_side = boxes_list_side[max_class_id_side]

        start_x_side = box_side[0]
        start_y_side = box_side[1]
        box_width_side = box_side[2]
        box_heighyt_side = box_side[3]

        prediction_id_side = ids_list_side[max_class_id_side]
        label_side = labels[prediction_id_side]
        confidence_side = confidences_list_side[max_class_id_side]
             
        end_x_side = start_x_side + box_width_side
        end_y_side = start_y_side + box_height_side

        box_color_side = colors[prediction_id_side]
        box_color_side = [int(each) for each in box_color_side]
    
        label_side = "{}: {:.2f}%" .format(label_side, confidence_side * 100)
        print("predicted object {}" . format(label_side))
        cv2.rectangle(frame_side, (start_x_side, start_y_side), (end_x_side, end_y_side), box_color_side, 2)
        cv2.putText(frame_side, label_side, (start_x_side, start_y_side - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_side, 2)
      cv2.imshow('side', frame_side)
      
      for max_id_top in max_ids_top:
        max_class_id_top = max_id_top[0]
        box_top = boxes_list_top[max_class_id_top]

        start_x_top = box_top[0]
        start_y_top = box_top[1]
        box_width_top = box_top[2]
        box_heighyt_top = box_top[3]

        prediction_id_top = ids_list_top[max_class_id_top]
        label_top = labels[prediction_id_top]
        confidence_top = confidences_list_top[max_class_id_top]
            
        end_x_top = start_x_top + box_width_top
        end_y_top = start_y_top + box_height_top
       
        box_color_top = colors[prediction_id_top]
        box_color_top = [int(each) for each in box_color_top]
    
        label_top = "{}: {:.2f}%" .format(label_top, confidence_top * 100)
        print("predicted object {}" . format(label_top))
        cv2.rectangle(frame_top, (start_x_top, start_y_top), (end_x_top, end_y_top), box_color_top, 2)
        cv2.putText(frame_top, label_top, (start_x_top, start_y_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_top, 2)


      #vertical object speed calculation
       
      if ver_detected == 1:
          ver_detected = 0 
          l_pik = cam_short_edge_p - start_y_side  #object length(pixel)
          d = box_center_y_top    # position of object on top image
          approx_length = l_pik * (side_cam_see * (side_cam_dis + (cam_short_edge_cm * d / cam_short_edge_p))) / (cam_short_edge_p * side_cam_dis)  
          if d < cam_short_edge_p / 2: 
              len_obj_hei_s = (cam_short_edge_cm * (top_cam_dis - approx_length)) / top_cam_dis  #length seen by camera at object height short edge
              real_d = (-((cam_short_edge_p / 2) - d) * 0.4) / cam_short_edge_p + (len_obj_hei_s * d / cam_short_edge_p) + (cam_short_edge_cm - len_obj_hei_s) / 2
          else:
              len_obj_hei_s = (cam_short_edge_cm * (top_cam_dis - approx_length)) / top_cam_dis
              real_d = (d - (cam_short_edge_p / 2) * 0.4) / cam_short_edge_p + cam_short_edge_cm - (len_obj_hei_s * (cam_short_edge_p - d) / cam_short_edge_p) - (cam_short_edge_cm - len_obj_hei_s) / 2
  
          real_length = l_pik * (side_cam_see * (side_cam_dis + real_d)) / (cam_short_edge_p * side_cam_dis)
          len_obj_hei_l = (cam_long_edge_cm * (top_cam_dis - real_length)) / top_cam_dis  #length seen by camera at object height long edge
          band_y = (real_d + 7) * 1.02 # ANN input 2 for vertical obj. 
          band_x = (345 - 3 * band_y) / 10 #ANN input 1 for vertical obj. 

          # template for vertical object
          if box_center_y_top < temp_size:
            if box_center_x_top < temp_size:
               template_top = frame_top[1:box_center_y_top + temp_size,1:box_center_x_top + temp_size,:]
               stat_top = 0
            elif box_center_x_top > cam_long_edge_p - temp_size:
               template_top = frame_top[1:box_center_y_top + temp_size,box_center_x_top - temp_size:cam_long_edge_p,:]
               stat_top = box_center_x_top - temp_size
            else:
                template_top = frame_top[1:box_center_y_top + temp_size,box_center_x_top - temp_size:box_center_x_top + temp_size,:]
                stat_top = box_center_x_top - temp_size
          elif box_center_y_top > cam_short_edge_p - temp_size:
            if box_center_x_top < temp_size:
               stat_top = 0
               template_top = frame_top[box_center_y_top - temp_size:cam_short_edge_p,1:box_center_x_top + temp_size,:]
            elif box_center_x_top > cam_long_edge_p - temp_size:
               template_top = frame_top[box_center_y_top - temp_size:cam_short_edge_p,box_center_x_top - temp_size:cam_long_edge_p,:]
               stat_top = box_center_x_top - temp_size
            else:
                template_top = frame_top[box_center_y_top - temp_size:cam_short_edge_p,box_center_x_top - temp_size:box_center_x_top + temp_size,:]
                stat_top = box_center_x_top - temp_size
          else:
            if box_center_x_top < temp_size:
               stat_top = 0
               template_top = frame_top[box_center_y_top - temp_size:box_center_y_top + temp_size,1:box_center_x_top + temp_size,:]
            elif box_center_x_top > cam_long_edge_p - temp_size:
               template_top = frame_top[box_center_y_top - temp_size:box_center_y_top + temp_size,box_center_x_top - temp_size:cam_long_edge_p,:]
               stat_top = box_center_x_top - temp_size
            else:
                template_top = frame_top[box_center_y_top - temp_size:box_center_y_top + temp_size,box_center_x_top - temp_size:box_center_x_top + temp_size,:]
                stat_top = box_center_x_top - temp_size
          wt_top = template_top.shape[1]
          ht_top = template_top.shape[0]
          
      #capture three more frame and find the template
          sleep(0.2)
          start9_top = time.time()   
          ret9_top, frame9_top = cap_top.read()
          ret9_top, frame9_top = cap_top.read()
          res9_top = cv2.matchTemplate(frame9_top,template_top,method)
          min_val9_top, max_val9_top, min_loc9_top, max_loc9_top = cv2.minMaxLoc(res9_top)
          top_left9_top = min_loc9_top
          stopp9_top = top_left9_top[0]
          bottom_right9_top = (top_left9_top[0] + wt_top, top_left9_top[1] + ht_top)
          found1_top = frame9_top[top_left9_top[1]:top_left9_top[1] + ht_top, top_left9_top[0]:top_left9_top[0] + wt_top]

          sleep(0.2)
          start1_top = time.time()   
          ret2_top, frame2_top = cap_top.read()
          ret2_top, frame2_top = cap_top.read()
          res1_top = cv2.matchTemplate(frame2_top,template_top,method)
          min_val1_top, max_val1_top, min_loc1_top, max_loc1_top = cv2.minMaxLoc(res1_top)
          top_left1_top = min_loc1_top
          stopp1_top = top_left1_top[0]
          bottom_right1_top = (top_left1_top[0] + wt_top, top_left1_top[1] + ht_top)
          found1_top = frame2_top[top_left1_top[1]:top_left1_top[1] + ht_top, top_left1_top[0]:top_left1_top[0] + wt_top]
      #goruntu3
          sleep(0.2)
          start2_top = time.time()
          ret4_top, frame4_top = cap_top.read()
          ret4_top, frame4_top = cap_top.read()
          res2_top = cv2.matchTemplate(frame4_top,template_top,method)
          min_val2_top, max_val2_top, min_loc2_top, max_loc2_top = cv2.minMaxLoc(res2_top)
          top_left2_top = min_loc2_top
          stopp2_top = top_left2_top[0]
          bottom_right2_top = (top_left2_top[0] + wt_top, top_left2_top[1] + ht_top)
          found2_top = frame4_top[top_left2_top[1]:top_left2_top[1] + ht_top, top_left2_top[0]:top_left2_top[0] + wt_top]
      #goruntu4
          sleep(0.2)
          start3_top = time.time()
          ret5_top, frame5_top = cap_top.read()
          ret5_top, frame5_top = cap_top.read()
          res3_top = cv2.matchTemplate(frame5_top,template_top,method)
          min_val3_top, max_val3_top, min_loc3_top, max_loc3_top = cv2.minMaxLoc(res3_top)
          top_left3_top = min_loc3_top
          stopp3_top = top_left3_top[0]
          bottom_right3_top = (top_left3_top[0] + wt_top, top_left3_top[1] + ht_top)
          found3_top = frame5_top[top_left3_top[1]:top_left3_top[1] + ht_top, top_left3_top[0]:top_left3_top[0] + wt_top]

          # speed pixel/second
          speed1_ver = (stopp1_top - stopp9_top) / (start1_top - start9_top) 
          speed2_ver = (stopp2_top - stopp1_top) / (start2_top - start1_top)
          speed3_ver = (stopp3_top - stopp2_top) / (start3_top - start2_top)

          taken_dis1 = len_obj_hei_l * speed1_ver / cam_long_edge_p
          taken_dis2 = len_obj_hei_l * speed2_ver / cam_long_edge_p
          taken_dis3 = len_obj_hei_l * speed3_ver / cam_long_edge_p
          # speed cm/second
          real_speed1_ver = taken_dis1 * cam_long_edge_p / cam_long_edge_cm
          real_speed2_ver = taken_dis2 * cam_long_edge_p / cam_long_edge_cm
          real_speed3_ver = taken_dis3 * cam_long_edge_p / cam_long_edge_cm
          # Checking that there are no errors in speeds
          if real_speed1_ver < 1 and real_speed2_ver < 1 and real_speed3_ver < 1:
              break
          elif real_speed2_ver < 1 and real_speed3_ver < 1:
              real_speed_ver = real_speed1_ver
          
          elif real_speed1_ver < 1 and real_speed3_ver < 1:
              real_speed_ver = real_speed2_ver
           
          elif real_speed1_ver < 1 and real_speed2_ver < 1:
              real_speed_ver = real_speed3_ver
            
          elif real_hız3_ver < 1:
              if real_speed1_ver * 0.85 > real_speed2_ver:
                  real_speed_ver = real_speed1_ver
              elif real_speed2_ver * 0.85 > real_speed1_ver:
                  real_speed_ver = real_speed2_ver
              else:
                  real_speed_ver = (real_speed1_ver + real_speed2_ver) / 2
              
          elif real_speed2_ver < 1:
              if real_speed1_ver * 0.85 > real_speed3_ver:
                  real_speed_ver = real_speed1_ver
              elif real_speed3_ver * 0.85 > real_speed1_ver:
                  real_speed_top = real_speed3_top
              else:
                  real_speed_ver = (real_speed1_ver + real_speed3_ver) / 2
            
          elif real_speed1_ver < 1:
              if real_speed2_ver * 0.85 > real_speed3_ver:
                  real_speed_ver = real_speed2_ver
              elif real_speed3_ver * 0.85 > real_speed2_ver:
                  real_speed_ver = real_speed3_ver
              else:
                  real_speed_ver = (real_speed2_ver + real_speed3_ver) / 2
              
          elif real_speed1_ver > 0 and real_speed2_ver > 0 and real_speed3_ver > 0:
              if real_speed1_ver * 0.85 > real_speed3_ver and real_speed2_ver * 0.85 > real_speed3_ver:
                  real_speed_ver = (real_speed1_ver + real_speed2_ver) / 2
              elif real_speed1_ver * 0.85 > real_speed2_ver and real_speed3_ver * 0.85 > real_speed2_ver:
                  real_speed_ver = (real_speed1_ver + real_speed3_ver) / 2
              elif real_speed2_top * 0.85 > real_speed1_top and real_speed3_top * 0.85 > real_speed1_ver:
                  real_speed_ver = (real_speed3_ver + real_speed2_top) / 2
              else:       
                  real_speed_ver = (real_speed1_ver + real_speed2_ver + real_speed3_ver) / 3

          last_frame_x = ((stopp3_top + temp_size) * (len_obj_hei_l / cam_long_edge_p)) + (cam_long_edge_cm - len_obj_hei_l) / 2# object' s last position 
          dis_to_take = (16 * 37.5) + cam_long_edge_p - (16 * (son_frame_x)) # distance between last position and grasping point
          if real_d < 15:
              ang1_ver = 90 + degrees(atan2((15 - real_d), 35))
              ANN_x = (sqrt((15 - real_d) ** 2 + 35 ** 2))
          else:
              ang1_ver = 90 - degrees(atan2((real_d - 15), 35))
              ANN_x = (sqrt((real_d - 15) ** 2 + 35 ** 2))
          ANN_y = int(-12 + (real_length / 2)) - 2
          
  
          sum_theta = -.5
          time_to_grasp = 1.1 * (100 / real_speed_ver) + dis_to_take / real_speed_ver# time
          test_in_top = np.c_[ANN_x, ANN_y, sum_theta] # x, y, angle
          test_predict_ver = ANN_ver.predict(test_in_top)
          
          # Check ANN result is suitable for robot arm
          while (degrees(test_predict_ver[0,0]) < 0 or  degrees(test_predict_ver[0,1]) < -90 or degrees(test_predict_ver[0,2]) < -90): # robot arm joints limits
              sum_theta = sum_theta - 0.05
              test_in_top = np.c_[ANN_x, ANN_y, sum_theta] 
              test_predict_ver = ANN_ver.predict(test_in_top)
              
          # Angles are calculated twice for grasping horizontal objects, and some joints angles are sent twice.
          # Since the format of sending angles to the controller is the same for horizontal and vertical objects, the same angles are sent twice for vertical objects.
          # theta2, theta3 and theta4 are sent twice.
          theta1_ver1 = int(ang1_ver)
          theta2_ver1 = int(degrees(test_predict_ver[0,0]))
          theta3_ver1 = int(90 + degrees(test_predict_ver[0,1]))
          theta4_ver1 = int(90 + degrees(test_predict_ver[0,2]))
          # the controller needs angles > 0
          if theta2_ver1 == 0:
              theta2_ver1 = theta2_ver1 + 1
          if theta3_ver1 == 0:
              theta3_ver1 = theta3_ver1 + 1
          if theta4_ver1 == 0:
              theta4_ver1 = theta4_ver1 + 1
          theta5_ver1 = int(90)
          # second angles
          theta1_ver2 = (ang1_ver)
          theta2_ver2 = (int(degrees(test_tahminr_top[0,0]))) # onlar bunlar
          theta3_ver2 = (int(90 + degrees(test_tahminr_top[0,1])))
          theta4_ver2 = (int(90 + degrees(test_tahminr_top[0,2])))
          theta5_ver2 = (theta5_ver1)
          if theta2_ver2 == 0:
              theta2_ver2 = theta2_ver2 + 1
          if theta3_ver2 == 0:
              theta3_ver2 = theta3_ver2 + 1
          if theta4_ver2 == 0:
              theta4_ver2 = theta4_ver2 + 1
          theta1_ver1_str = str(theta1_ver1)
          theta2_ver1_str = str(theta2_ver1)
          theta3_ver1_str = str(theta3_ver1)
          theta4_ver1_str = str(theta4_ver1)
          theta5_ver1_str = str(theta5_ver1)

          theta1_ver2_str = str(theta1_ver2)
          theta2_ver2_str = str(theta2_ver2)
          theta3_ver2_str = str(theta3_ver2)
          theta4_ver2_str = str(theta4_ver2)
          #Angles are sent to the control as 3 digits.
          if len(theta1_ver1_str) == 1:
              theta1_ver1_str = "00" + theta1_ver1_str
          elif len(theta1_ver1_str) == 2:
              theta1_ver1_str = "0" + theta1_ver1_str
          if len(theta2_ver1_str) == 1:
              theta2_ver1_str = "00" + theta2_ver1_str
          elif len(theta2_ver1_str) == 2:
              theta2_ver1_str = "0" + theta2_ver1_str
          if len(theta3_ver1_str) == 1:
              theta3_ver1_str = "00" + theta3_ver1_str
          elif len(theta3_ver1_str) == 2:
              theta3_ver1_str = "0" + theta3_ver1_str
          if len(theta4_ver1_str) == 1:
              theta4_ver1_str = "00" + theta4_ver1_str
          elif len(theta4_ver1_str) == 2:
              theta4_ver1_str = "0" + theta4_ver1_str
          if len(theta5_ver1_str) == 1:
              theta5_ver1_str = "00" + theta5_ver1_str
          elif len(theta5_ver1_str) == 2:
              theta5_ver1_str = "0" + "90"
    
          if len(theta2_ver2_str) == 1:
              theta2_ver2_str = "00" + theta2_ver2_str
          elif len(theta2_ver2_str) == 2:
              theta2_ver2_str = "0" + theta2_ver2_str
          if len(theta3_ver2_str) == 1:
              theta3_ver2_str = "00" + theta3_ver2_str
          elif len(theta3_ver2_str) == 2:
              theta3_ver2_str = "0" + theta3_ver2_str
          if len(theta4_ver2_str) == 1:
              theta4_ver2_str = "00" + theta4_ver2_str
          elif len(theta4_ver2_str) == 2:
              theta4_ver2_str = "0" + theta4_ver2_str
          stop_top_time = time.time()
          time_to_grasp_ver = time_to_grasp -(stop_top_time - start3_top)
          # time must be <100 and >1 second
          if time_to_grasp_ver < 10: # convert to time to string for send via serial
              time_to_grasp_ver_str = '0' + str(time_to_grasp_ver * 1000)
          else:
              time_to_grasp_ver_str = str(time_to_grasp_ver * 1000)

          if  theta1_ver1 > 0 and theta2_ver1 > 0 and theta3_ver1 > 0 and theta4_ver1 > 0 and theta2_ver2 > 0 and theta3_ver2 > 0 and theta4_ver2 > 0:
                num_top = theta1_ver1_str + theta2_ver1_str + theta3_ver1_str + theta4_ver1_str + theta5_ver1_str + time_to_grasp_ver_str[0:5] + theta2_ver2_str + theta3_ver2_str + theta4_ver2_str
                value = write_read(num_top)
                print("angles were sent")
                
          sleep(time_to_grasp_ver)
      cv2.imshow('top', frame_top)
  end = time.time()
  
  if centers_matched == 0:
      continue
  centers_matched = 0
  # horizontal object speed calculation
  if obj_detected == 1:

      obj_detected = 0
      sleep(0.2)
      start1 = time.time()
      ret2, frame2 = cap_top.read()
      ret2, frame2 = cap_top.read()
      res1 = cv2.matchTemplate(frame2,template,method)
      min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
      top_left1 = min_loc1
      stopp1 = top_left1[0]
      bottom_right1 = (top_left1[0] + wt, top_left1[1] + ht)
      found1 = frame2[top_left1[1]:top_left1[1] + ht, top_left1[0]:top_left1[0] + wt]

      sleep(0.2)
      start2 = time.time()
      ret4, frame4 = cap_top.read()
      ret4, frame4 = cap_top.read()
      res2 = cv2.matchTemplate(frame4,template,method)
      min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
      top_left2 = min_loc2
      stopp2 = top_left2[0]
      bottom_right2 = (top_left2[0] + wt, top_left2[1] + ht)
      found2 = frame4[top_left2[1]:top_left2[1] + ht, top_left2[0]:top_left2[0] + wt]

      sleep(0.2)
      start3 = time.time()
      ret5, frame5 = cap_top.read()
      ret5, frame5 = cap_top.read()
      res3 = cv2.matchTemplate(frame5,template,method)
      min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
      top_left3 = min_loc3
      stopp3 = top_left3[0]
      bottom_right3 = (top_left3[0] + wt, top_left3[1] + ht)
      found3 = frame5[top_left3[1]:top_left3[1] + ht, top_left3[0]:top_left3[0] + wt]

      speed1_hor = (stopp1 - stat) / (start1 - start)
      speed2_hor = (stopp2 - stat) / (start2 - start)
      speed3_hor = (stopp3 - stat) / (start3 - start)

      if speed1_hor < 0 and speed2_hor < 0 and speed3_hor < 0:
          break
      elif speed2_hor < 0 and speed3_hor < 0:
          speed_hor = speed1_hor
        
      elif speed1_hor < 0 and speed3_hor < 0:
          speed_hor = speed2_hor
        
      elif speed1_hor < 0 and speed2_hor < 0:
          speed_hor = speed3_hor
          
      elif speed3_hor < 0:
          speed_hor = (speed1_hor + speed2_hor) / 2
         
      elif speed2_hor < 0:
          speed_hor = (speed1_hor + speed3_hor) / 2
          
      elif speed1_hor < 0:
          speed1_hor = (speed2_hor + speed3_hor) / 2
         
      elif speed1_hor > 0 and speed2_hor > 0 and speed3_hor > 0:
          speed_hor = (speed1_hor + speed2_hor + speed3_hor) / 3
        
      image_y = ceny1 # pixel
      Robot_x = 10 # cm
      Robot_y = 12 # cm
      ANN_x = (image_y / 16) + Robot_y
      sum_theta = -1
      if ceny1 < 200:
          ANN_y = -6
      else:
          ANN_y = -12
      if prediction_id1 == 1:
          ANN_y = ANN_y
          ANN_x = ANN_x + .5
      test_in = np.c_[ANN_x, ANN_y, sum_theta]
      test_predict_hor = ANN_hor.predict(test_in)
  
      while (degrees(test_predict_hor[0,0]) < 0 or  degrees(test_predict_hor[0,1]) < -90 or degrees(test_predict_hor[0,2]) < -90):
          sum_theta = sum_theta - 0.1
          test_in = np.c_[ANN_x, -10, sum_theta] 
          test_predict_hor = ANN_hor.predict(test_in)
          if sum_theta < -3:
              sum_theta = 1
         
      theta2_hor1 = int(degrees(test_predict_hor[0,0]))
      theta3_hor1 = int(90 + degrees(test_predict_hor[0,1]))
      theta4_hor1 = int(90 + degrees(test_predict_hor[0,2]))
      if angle1 < 0: # PCA angle
          angle1 = -angle1
          theta5_hor1 = int(angle1)
      else:           
          theta5_hor1 = int(180 - angle1)
      if theta2_hor1 == 0:
          theta2_hor1 = theta2_hor1 + 1
      if theta3_hor1 == 0:
          theta3_hor1 = theta3_hor1 + 1
      if theta4_hor1 == 0:
          theta4_hor1 = theta4_hor1 + 1

      theta2_hor1_str = str(theta2_hor1)
      theta3_hor1_str = str(theta3_hor1)
      theta4_hor1_str = str(theta4_hor1)
      theta5_hor1_str = str(theta5_hor1)

      theta1_hor2 = 90
      theta2_hor2 = theta2_hor1 + 25
      theta3_hor2 = theta3_hor1
      theta4_hor2 = theta4_hor1 - 12
      
      theta1_hor2_str = "0" + str(theta1_hor2)
      theta2_hor2_str = str(theta2_hor2)
      theta3_hor2_str = str(theta3_hor2)
      theta4_hor2_str = str(theta4_hor2)
      if len(theta2_hor1_str) == 1:
          theta2_hor1_str = "00" + theta2_hor1_str
      elif len(theta2_hor1_str) == 2:
          theta2_hor1_str = "0" + theta2_hor1_str
      if len(theta3_hor1_str) == 1:
          theta3_hor1_str = "00" + theta3_hor1_str
      elif len(theta3_hor1_str) == 2:
          theta3_hor1_str = "0" + theta3_hor1_str
      if len(theta4_hor1_str) == 1:
          theta4_hor1_str = "00" + theta4_hor1_str
      elif len(theta4_hor1_str) == 2:
          theta4_hor1_str = "0" + theta4_hor1_str
      if len(theta5_hor1_str) == 1:
          theta5_hor1_str = "00" + theta5_hor1_str
      elif len(theta5_hor1_str) == 2:
          theta5_hor1_str = "0" + theta5_hor1_str
    
      if len(theta2_hor2_str) == 1:
          theta2_hor2_str = "00" + theta2_hor2_str
      elif len(theta2_hor2_str) == 2:
          theta2_hor2_str = "0" + theta2_hor2_str
      if len(theta3_hor1_str) == 1:
          theta3_hor1_str = "00" + theta3_hor1_str
      elif len(theta3_hor1_str) == 2:
          theta3_hor1_str = "0" + theta3_hor1_str
      if len(theta4_hor1_str) == 1:
          theta4_hor1_str = "00" + theta4_hor1_str
      elif len(theta4_hor1_str) == 2:
          theta4_hor1_str = "0" + theta4_hor1_str

      end1 = time.time()
      time_to_grasp = ((1200 - cenx1) / speed_hor) - (end1 - start1) - (end - start) + (1 / (speed_hor / 100))
      if time_to_grasp < 10:
          waitting = '0' + str(time_to_grasp * 1000)
      else:
          waitting = str(time_to_grasp * 1000)
      sleep(time_to_grasp)
      if  theta2_hor1 > 0 and theta3_hor1 > 0 and theta4_hor1 > 0 and theta2_hor2 > 0 and theta3_hor2 > 0 and theta4_hor2 > 0 and theta2_hor2 < 150 and theta3_hor2 < 150 and theta4_hor2 < 150:
       
                num = theta1_hor2 + theta2_hor1 + theta3_hor1 + theta4_hor1 + theta5_hor1 + waitting[0:5] + theta2_hor2 + theta3_hor2 + theta4_hor2
                value = write_read(num)
              
      ret, frame = cap_top.read()
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
cap_top.release()
cv2.destroyAllWindows()
