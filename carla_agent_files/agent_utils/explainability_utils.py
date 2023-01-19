import carla
from PIL import Image
import numpy as np
import cv2


def get_masked_viz_3rd_person(save_path_org, save_path_mask, step, input_data):
    # PIXELS_PER_METER = 5.689
    im3 = Image.fromarray(cv2.cvtColor(input_data['rgb_back'][1][:, :, :3], cv2.COLOR_BGR2RGB))
    sem = input_data['sem_back'][1][:, :, 2]
    mask = (sem == 10) | (sem==0)
    mask_neg = (mask==0)

    img_raw = input_data['rgb_back'][1][:, :, :3].copy()
    img_raw[mask_neg] = [255,255,255]
    foreground = np.concatenate((img_raw, np.expand_dims(mask*255,-1).astype(np.uint8)), axis=2)
    masked_img = Image.fromarray(cv2.cvtColor(foreground, cv2.COLOR_BGRA2RGBA))
    background = Image.fromarray(cv2.cvtColor(input_data['rgb_back'][1][:, :, :3].copy(), cv2.COLOR_BGR2RGB).astype(np.uint8)).convert('RGBA')
    background.putalpha(130)
    white_img = Image.new("RGBA", (mask.shape[1],mask.shape[0]), (255, 255, 255, 255))
    white_img.paste(background, (0,0), background)
    white_img.paste(masked_img, (0,0), masked_img)
    white_img2 = Image.new("RGB", (mask.shape[1],mask.shape[0]), (255, 255, 255))
    white_img2.paste(white_img, (0,0), white_img.split()[3])

    img = Image.fromarray(np.asarray(white_img2).copy())
    
    im3.save(f'{save_path_org}/{step}.png')
    img.save(f'{save_path_mask}/{step}.png')


def draw_attention_bb_in_carla(_world, keep_vehicle_ids, keep_vehicle_attn, frame_rate_sim):
    actors = _world.get_actors()
    all_vehicles = actors.filter('*vehicle*')
    
    for vehicle in all_vehicles:
        # print(vehicle.id)
        if vehicle.id in keep_vehicle_ids:
            index = keep_vehicle_ids.index(vehicle.id)
            # cmap = plt.get_cmap('YlOrRd')
            # c = cmap(object[1])
            # color = carla.Color(*[int(i*255) for i in c])
            c = get_color(keep_vehicle_attn[index])
            color = carla.Color(r=int(c[0]), g=int(c[1]), b=int(c[2]))        
            loc = vehicle.get_location()
            loc.z = vehicle.bounding_box.extent.z/2
            bb = carla.BoundingBox(loc, vehicle.bounding_box.extent)
            bb.extent.z=0.2
            bb.extent.x+=0.2      
            bb.extent.y+=0.05                  
            # bb = carla.BoundingBox(vehicle.get_transform().location, vehicle.bounding_box.extent)
            _world.debug.draw_box(box=bb, rotation= vehicle.get_transform().rotation, thickness=0.4, color=color, life_time=(1.0 / frame_rate_sim)) #(1.0/self.frame_rate_sim))



def get_attn_norm_vehicles(attention_score, data_car, attn_map):
    if attention_score == 'AllLayer':
        # attention score for CLS token, sum of all heads
        attn_vector = [np.sum(attn_map[i][0,:,0,1:].numpy(), axis=0) for i in range(len(attn_map))]
    else:
        print(f"Attention score {attention_score} not implemented! Please use 'AllLayer'!")
        raise NotImplementedError
        
    attn_vector = np.array(attn_vector)
    offset = 0
    # if no vehicle is in the detection range we add a dummy vehicle
    if len(data_car) == 0:
        attn_vector = np.asarray([[0.0]])
        offset = 1

    # sum over layers
    attn_vector = np.sum(attn_vector, axis=0)
    
    # remove route elements
    attn_vector = attn_vector[:len(data_car)+offset]+0.00001

    # get max attention score for normalization
    # normalization is only for visualization purposes
    max_attn = np.max(attn_vector)
    attn_vector = attn_vector / max_attn
    attn_vector = np.clip(attn_vector, None, 1)
    
    return attn_vector

def get_vehicleID_from_attn_scores(data, data_car, topk, attn_vector):
    # get ids of all vehicles in detection range
    data_car_ids = [
        float(x['id'])
        for x in data if x['class'] == 'Car']

    # get topk indices of attn_vector
    if topk > len(attn_vector):
        topk = len(attn_vector)
    else:
        topk = topk
    
    # get topk vehicles indices
    attn_indices = np.argpartition(attn_vector, -topk)[-topk:]
    
    # get carla vehicles ids of topk vehicles
    keep_vehicle_ids = []
    keep_vehicle_attn = []
    for indice in attn_indices:
        if indice < len(data_car_ids):
            keep_vehicle_ids.append(data_car_ids[indice])
            keep_vehicle_attn.append(attn_vector[indice])
    
    # if we don't have any detected vehicle we should not have any ids here
    # otherwise we want #topk vehicles
    if len(data_car) > 0:
        assert len(keep_vehicle_ids) == topk
    else:
        assert len(keep_vehicle_ids) == 0
        
    return keep_vehicle_ids, attn_indices, keep_vehicle_attn
    

def get_color(attention):
    colors = [
        (255, 255, 255, 255),
        # (220, 228, 180, 255),
        # (190, 225, 150, 255),
        (240, 240, 210, 255),
        # (190, 219, 96, 255),
        (240, 220, 150, 255),
        # (170, 213, 79, 255),
        (240, 210, 110, 255),
        # (155, 206, 62, 255),
        (240, 200, 70, 255),
        # (162, 199, 44, 255),
        (240, 190, 30, 255),
        # (170, 192, 20, 255),
        (240, 185, 0, 255),
        # (177, 185, 0, 255),
        (240, 181, 0, 255),
        # (184, 177, 0, 255),
        (240, 173, 0, 255),
        # (191, 169, 0, 255),
        (240, 165, 0, 255),
        # (198, 160, 0, 255),
        (240, 156, 0, 255),
        # (205, 151, 0, 255),
        (240, 147, 0, 255),
        # (212, 142, 0, 255),
        (240, 137, 0, 255),
        # (218, 131, 0, 255),
        (240, 126, 0, 255),
        # (224, 120, 0, 255),
        (240, 114, 0, 255),
        # (230, 108, 0, 255),
        (240, 102, 0, 255),
        # (235, 95, 0, 255),
        (240, 88, 0, 255),
        # (240, 80, 0, 255),
        (242, 71, 0, 255),
        # (244, 61, 0, 255),
        (246, 49, 0, 255),
        # (247, 34, 0, 255),
        (248, 15, 0, 255),
        (249, 6, 6, 255),
    ]
    
    ix = int(attention * (len(colors) - 1))
    return colors[ix]
    