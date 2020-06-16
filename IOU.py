import torch

# Implementation for top left and bottom right corner is given
# To do for midepoint w,h format
def intersection_over_union(box_values, actual_box):
  # box shape (N,4) 4-> (x1,y1), (x2, y2)
  # 1) calculate intersection box coords
  inter_box_x1 = torch.max(box_values[...,0:1], actual_box[...,0:1])
  inter_box_y1 = torch.max(box_values[...,1:2], actual_box[...,1:2])
  inter_box_x2 = torch.min(box_values[...,2:3], actual_box[...,2:3])
  inter_box_y2 = torch.min(box_values[...,3:4], actual_box[...,3:4])

  # when no intersection substraction will be negitive so clamp -ve values to 0.
  inter_box_area = (inter_box_x2 - inter_box_x1).clamp(0) * (inter_box_y2 - inter_box_y1).clamp(0)

