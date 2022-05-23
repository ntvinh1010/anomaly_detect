import cv2
import numpy as np
from imutils import paths

root_path = "./UCSDped1/Test/"
#root_path = "./UCSDped2/Test/"

#Videos which has anomaly 
# For Ped 1
image_paths = ["Test003", "Test004", "Test014", "Test018", "Test019", "Test021", "Test022", "Test023", "Test024", "Test032"]

# For Ped 2
#image_paths = ["Test001", "Test002", "Test003", "Test004", "Test005", "Test006", "Test007", "Test008", "Test009", "Test010", "Test011", "Test012"]

box_paths = [s + "_gt" for s in image_paths]

for image_path, box_path in zip(image_paths, box_paths):
    imgs = sorted(list(paths.list_images(root_path + image_path)))
    boxes = sorted(list(paths.list_images(root_path + box_path)))

    print(imgs)

    #Resolution for Ped 1
    # (h, w) = (158, 238)

    #Resolution for Ped 2
    (h, w) = (240, 360)

    video = cv2.VideoWriter('./' + image_path + '.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h), True)


    for img_path, box_path in zip(imgs, boxes):
        img = cv2.imread(img_path, 1)
        box = cv2.imread(box_path, 0)

        cv2.imshow('Images', img)
        cv2.imshow('Boxes', box)

        b,g,r = cv2.split(img)
        r = np.where(box > 0, 255, r)
        img = cv2.merge((b,g,r))

        cv2.imshow('Image Box', img)
        cv2.waitKey(1)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()