from transformers import TrOCRProcessor
from PIL import Image
from transformers import VisionEncoderDecoderModel
import cv2
import scipy.ndimage as ndimage
import difflib
import os
import numpy as np

vocabs = {'Phone', 'Router', 'Switch', 'Firewall', 'Laptop', 'phone', 'router', 'switch', 'firewall', 'laptop'}
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
ROOT = '.'
def read_img(img_path):
    image = cv2.imread(img_path) # os.path.join(ROOT, 'network1.jpg'))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.dilate(thresh, kernel, iterations=6)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    boxes = []
    for i, c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
        pixel_values = processor(ROI, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if generated_text.replace(' ', '').isalpha():
            path = os.path.join(ROOT, 'output/ROI')
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(ROOT, 'output/ROI/ROI_{}.png'.format(i)), ROI)
            boxes.append((x,y,w,h, generated_text))
    return boxes


def merge_boxes(boxes):
    avg_word_wid = 0
    avg_word_hei = 0
    for b in boxes:
        x,y,w,h,text = b
        avg_word_wid += w
        avg_word_hei += h
    avg_word_wid /= len(boxes)
    avg_word_hei /= len(boxes)

    new_boxes = []    
    skip_set = set()
    for i in range(len(boxes)-1):
        x1,y1,w1,h1,text1 = boxes[i]
        skip = 0
        for j in range(i+1, len(boxes)):
            x2,y2,w2,h2,text2 = boxes[j]
            if x1+w1+avg_word_wid/2 >= x2 and x1+w1 < x2 and abs(y1-y2) < avg_word_hei/2 and abs(h1-h2) < avg_word_hei/2:
                w1 += abs(x2-x1)-w1+w2
                h1 = max(h1,h2)
                text1 += ' '+text2
                skip_set.add((x2,y2,w2,h2,text2))
                skip = 1
        if (x1,y1,w1,h1,text1) not in skip_set:
            new_boxes.append((x1,y1,w1,h1,text1))
    if boxes[-1] not in skip_set:
        new_boxes.append(boxes[-1])
    return new_boxes

def text_process(img_path):
    boxes = read_img(img_path)
    new_boxes = merge_boxes(boxes)
    masked_img = cv2.imread(img_path)
    node_dicts = []
    H, W, _ = masked_img.shape
    mask = np.zeros((H, W))
    for i, box in enumerate(new_boxes):
        x,y,w,h, text = box
        ROI = masked_img[y:y+h, x:x+w]
        new_mask = np.zeros((H, W))
        new_mask[y:y+h, x:x+w] = 255
        mask = np.logical_or(new_mask, mask)
        border = [x,y,w,h]
        words = text.split(' ')
        if len(words) < 2:
            continue
        words[0] = difflib.get_close_matches(words[0], vocabs, n=1)[0]
        fixedname = ' '.join(words)
        node_dicts.append({
        "id":i,
        "pred_text": fixedname,
        "border": border
        })
        path = os.path.join(ROOT, 'output/names')
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(ROOT, 'output/names/name_{}.png'.format(fixedname)), ROI)

    mask = mask.astype(float)# transform mask array to an image
    mask = cv2.resize(mask, dsize=(masked_img.shape[1], masked_img.shape[0]), interpolation=cv2.INTER_CUBIC)# resize mask: shrink to masked_img's size use area average
    mask = mask.astype(bool)# back to boolean

    # dilate the mask
    mask = ndimage.binary_dilation(mask, [[True, True, True], 
                    [True, True, True], 
                    [True, True, True]])

    masked_img[mask] = 255.0
    cv2.imwrite(os.path.join(ROOT, 'output/img.png'), masked_img)
    return masked_img, node_dicts