import torch

# Implementation for top left and bottom right corner is given
# To do for midepoint w,h format
def intersection_over_union(box_values, actual_box):
  # box shape (N,4) 4-> (x1,y1), (x2, y2)
  # 1) calculate intersection box coords

