import cv2
import yaml

config = yaml.load(open('src/settings.yaml'), yaml.FullLoader)
print(config)
cap = cv2.VideoCapture(config['id_camera_1'])
ret, frame = cap.read()
r = cv2.selectROI('cc', frame)
y1 = int(r[1])
y2 = int(r[1]+r[3])
x1 = int(r[0])
x2 = int(r[0]+r[2])

roi = [x1, y1, x2, y2]
config.update({'roi_camera_1': roi})

with open('src/settings.yaml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)