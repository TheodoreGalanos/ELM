from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from ladybug.epw import EPW

domains = [(0, 0,   2.5, [120, 120, 255], 'sitting long' ),
           (1, 2.5, 4.0, [120, 255, 255], 'sitting short'),
           (2, 4.0, 6.0, [120, 255, 120], 'walking leisurely' ),
           (3, 6.0, 8.0, [255, 255, 120], 'walking fast' ),
           (4, 8.0, 15,  [255, 197, 110], 'uncomfortable'),
           (5, 15,  100, [255, 120, 120], 'unsafe'    )]

def wind_factors(pil_img):
    min_hue, max_hue = 170, 255
    if pil_img.mode != 'HSV':
        pil_img = pil_img.convert('HSV')
    pil_img = pil_img.filter(ImageFilter.BLUR)
    hue_npa = np.array(pil_img).astype(float)[:, :, 0]
    return 1 - hue_npa / min_hue

def snapshot(pil_img, input_height_map, wind_speed):
    law_npa = wind_speed * wind_factors(pil_img)
    map_npa = np.full((*law_npa.shape, 3), 255)
    lawson_values = np.full((*law_npa.shape, 1), 0)
    mask_npa = np.array(input_height_map.convert('L'))
    mask_npa[mask_npa < 255] = 0

    mask_copy = mask_npa.copy()
    mask_copy[mask_copy == 255] = 1
    total_area = mask_copy.sum()
    for dom in domains:
        cond = (law_npa >= dom[1]) & (law_npa <= dom[2]) & (mask_npa == 255)
        map_npa[..., :][cond] = dom[3]
        lawson_values[..., :][cond] = dom[0]
    return Image.fromarray(np.uint8(map_npa), mode="RGB"), lawson_values, total_area
