from PIL import Image, ImageDraw
from einops import rearrange, reduce, repeat
import math
import time
from pathlib import Path

import numpy as np



def get_coords(x, y, angle, vel):
    length = vel
    endx2 = x + length * math.cos(math.radians(angle))
    endy2 = y + length * math.sin(math.radians(angle))

    return x, y, endx2, endy2        


def viz_tokens_bev(input, pred, gt, brake_pred, brake_gt, epoch, train=False):
    batch_size = input.shape[0]
    
    size = 1024
    origin = (size//2, size//2)
    
    color = [(0, 255, 0), (255, 0, 0), (30, 144, 255)]
   
    
    for i in range(batch_size):
        # create black image
        image = Image.new('RGB', (size, size))
        draw = ImageDraw.Draw(image)
        
        for ix, sequence in enumerate([input, pred, gt]):
            features = sequence[i][1:]
            
            # we have 5 features per vehicle and want to filter EOF tokens out
            EOF_tokens = features.shape[0]%4
            features = sequence[i][1:-EOF_tokens]
            
            features = rearrange(features, '(vehicle features) -> vehicle features', features=4)
            
            for ixx, vehicle in enumerate(features):
                # draw vehicle
                y = -vehicle[0].item()*5 + origin[0]
                x = -vehicle[1].item()*5 + origin[1]
                yaw = vehicle[2].item()
                vel = vehicle[3].item()
                origin_v = (x, y)
                
                if ixx == 0 and ix == 0:
                    draw.regular_polygon((origin_v, 8), n_sides=4, rotation=-yaw-90, outline=color[ix], fill='green') #ego vehicle
                else:
                    draw.regular_polygon((origin_v, 8), n_sides=4, rotation=-yaw-90, outline=color[ix])
                
                endx1, endy1, endx2, endy2 = get_coords(x, y, yaw-90, vel)
                draw.line((endx1, endy1, endx2, endy2), fill=color[ix], width=2)
                
            
        if train:
            suffix = 'train'
        else:
            suffix = 'val'
        
        Path(f'bev_viz_{suffix}/epoch{epoch}').mkdir(parents=True, exist_ok=True)
        image.save(f'bev_viz_{suffix}/epoch{epoch}/brakepred_{brake_pred[i].item()}_gt_{brake_gt[i].item()}_{time.time()}.png')
    
    
def create_BEV(vehicles, route, input_extent, LBC, pix_per_m=5, use_speed_for_whole_vehicle=True, remove_speed=False, inp='input', cnt=0, visualize=False):
    
    size = 192
    # TODO support for AIMBEV:
    # origin = (size+14, size//2)

    if LBC and pix_per_m == 5:
        origin = (size, size//2)
        PIXELS_PER_METER = 5
    elif LBC and pix_per_m == 3:
        max_d = 30
        size = int(max_d*pix_per_m*2)
        origin = (size, size//2)
        PIXELS_PER_METER = pix_per_m
    else:
        max_d = 30
        size = int(max_d*pix_per_m*2)
        origin = (size//2, size//2)
        PIXELS_PER_METER = pix_per_m

    
    # color = [(255, 0, 0), (0, 0, 255)]
    color = [(255), (255)]
   
    
    # create black image
    image_0 = Image.new('L', (size, size))
    image_1 = Image.new('L', (size, size))
    image_2 = Image.new('L', (size, size))
    vel_array = np.zeros((size, size))
    draw0 = ImageDraw.Draw(image_0)
    draw1 = ImageDraw.Draw(image_1)
    draw2 = ImageDraw.Draw(image_2)

    draws = [draw0, draw1, draw2]
    imgs = [image_0, image_1, image_2]
    
    for ix, sequence in enumerate([vehicles, route]):
               
        # features = rearrange(features, '(vehicle features) -> vehicle features', features=4)
        for ixx, vehicle in enumerate(sequence):
            # draw vehicle
            x = -vehicle[2]*PIXELS_PER_METER + origin[1]
            y = -vehicle[1]*PIXELS_PER_METER + origin[0]
            yaw = vehicle[3]
            extent_x = vehicle[5]*PIXELS_PER_METER/2
            extent_y = vehicle[6]*PIXELS_PER_METER/2
            origin_v = (x, y)
            vel = vehicle[4]/3.6 # in m/s
            
            if input_extent:
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
                draws[ix].polygon((p1, p2, p3, p4), outline=color[ix], fill=color[ix])
                if ix == 0 and use_speed_for_whole_vehicle:
                    image_2 = Image.new('L', (size, size))
                    draw2 = ImageDraw.Draw(image_2)
                    draw2.polygon((p1, p2, p3, p4), outline=(255), fill=(255))
                    mask_img = np.asarray(image_2)/255
                    if remove_speed:
                        vel = 0
                    vel_array = vel_array + mask_img*vel
            else:
                draws[ix].regular_polygon((origin_v, 3), n_sides=4, rotation=-yaw-90, outline=color[ix], fill=color[ix])


    if use_speed_for_whole_vehicle == False:
        images = [np.asarray(img) for img in imgs]
        image = np.stack([images[0], images[2], images[1]], axis=-1)
        BEV = image/255
        for vehicle in vehicles:
            x = -vehicle[2]*PIXELS_PER_METER + origin[1]
            y = -vehicle[1]*PIXELS_PER_METER + origin[0]
            vel = vehicle[4]/3.6 # in m/s

            if x > size or y > size:
                continue

            if remove_speed:
                vel = 0
            BEV[int(y),int(x),1] = vel
    else:
        images = [np.asarray(img) for img in imgs]
        image = np.stack([images[0]/255, vel_array, images[1]/255], axis=-1)
        BEV = image

    if cnt%20 == 0 and visualize:
        Path(f'bev_viz').mkdir(parents=True, exist_ok=True)
        Image.fromarray((BEV * 255).astype(np.uint8)).save(f'bev_viz/new2_{cnt}_{inp}_{time.time()}.png')


    return BEV


def get_coords_BB(x, y, angle, extent_x, extent_y):
    endx1 = x - extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy1 = y + extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx2 = x + extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy2 = y - extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx3 = x + extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy3 = y - extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    endx4 = x - extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy4 = y + extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)




