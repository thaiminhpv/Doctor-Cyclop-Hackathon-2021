import matplotlib.cm as cm
import cv2
import torch
import base64
import requests

API_URL = 'http://18.140.53.169/api/uploadMaskImage'
cmap = cm.get_cmap('viridis')
    
def push_mask_list(caseIds, imgs, masks):
    for i in range(len(caseIds)):
        print(caseIds[i])
        push_mask(caseIds[i], imgs[i], masks[i][0]);

def push_mask(caseId, img, mask):
    # img = cv2.resize(img, (1024, 1024))
    img = img.cpu().numpy().transpose(1,2,0) * 255
    for i in range(1024):
        for j in range(1024):
            value = mask[i][j].item()
            if value >= 0.15:
                r, b, g, a = cmap(value)
                img[i][j] = torch.FloatTensor([r * 255, b * 255, g * 255])

    retval, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer)

    # Send to server
    body = {
        'caseId': caseId,
        'base64': img_base64
    }
    requests.post(API_URL, data=body)
