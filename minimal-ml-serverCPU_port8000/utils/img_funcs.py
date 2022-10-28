from PIL import Image
import numpy as np

def rotate_input(pil_img, degrees, interval=512):
    """Method to rotate image by `degrees` in a COUNTER-CLOCKWISE direction.
    As some rotations cause the corners of the original image to be cropped,
    the `interval` argument allows the image to expand in size.
    """
    def next_interval(current):
        c = int(current)
        if c % interval == 0:
            return c
        else:
            return interval * ((c // interval) + 1)

    def paste_top_left_coords(rot_width, rot_height, exp_width, exp_height):
        calc = lambda r, e: int((e - r) / 2)
        return calc(rot_width, exp_width), calc(rot_height, exp_height)

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    degrees = degrees % 360
    if degrees % 90 != 0:
        rot_img = pil_img.rotate(
            angle=degrees,
            resample=Image.BICUBIC,
            expand=1,
            fillcolor=(255, 255, 255)
        )
        min_width, min_height = rot_img.size
        exp_width  = next_interval(min_width)
        exp_height = next_interval(min_height)
        pil_img = Image.new('RGB', (exp_width, exp_height), (255, 255, 255))
        paste_coords = paste_top_left_coords(min_width, min_height,
                                             exp_width, exp_height)
        pil_img.paste(rot_img, paste_coords)
    else:
        pil_img = pil_img.rotate(
            angle=degrees,
            resample=Image.BICUBIC,
            fillcolor=(255, 255, 255)
        )
    return pil_img

def rotate_to_origin(pil_img, original_height, original_width, degrees):
    rot_img = pil_img.rotate(
        angle=degrees,
        resample=Image.BICUBIC,
        expand=1,
        fillcolor=(255, 255, 255)
    )
    rot_width, rot_height = rot_img.size
    return rot_img.crop((
        (rot_width  - original_width)  / 2,
        (rot_height - original_height) / 2,
        (rot_width  - original_width)  / 2 + original_width,
        (rot_height - original_height) / 2 + original_height
    ))

def remap_wind(pil_img):
    pil_img = pil_img.convert('HSV')
    h, s, v = pil_img.split()
    remap = lambda h: int(170 * ((-(h - 170) / 85) + 1))
    hue = [remap(max(170, h_)) for h_ in h.getdata()]
    h.putdata(hue)
    s.putdata([170] * len(hue))
    v.putdata([255] * len(hue))
    return Image.merge('HSV', (h, s, v)).convert('RGB')

def remove_buildings(pil_img, input_height_map):
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    mask = np.array(input_height_map.convert('L'))
    mask[mask < 255] = 0
    pil_img = Image.fromarray(np.dstack((np.array(pil_img), mask)), mode='RGBA')
    return pil_img