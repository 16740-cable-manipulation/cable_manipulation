from rs_driver import Realsense
import cv2
import numpy as np
import time
import os

if __name__ == "__main__":
    image_cnt = 0
    realsense = Realsense()
    root_path = "/home/student/cable_ws/src/cable_manipulation/cableImages/rs_cable_imgs"
    while True:
        depth, bgr = realsense.getFrameSet()  # actually bgr, not rgb
        cv2.imshow("img", bgr)
        cmd = cv2.waitKey(32)
        # save image if pressed 's' or space key
        if cmd == ord("s") or cmd == 32:
            print("Saved image no. %03d!" % image_cnt)
            file_path = os.path.join(
                root_path, "img" + ("%03d" % image_cnt) + ".png"
            )
            cv2.imwrite(file_path, bgr)
            image_cnt += 1
        elif cmd == ord("q") or cmd == 27:
            print("Exit im_saver")
            break
    cv2.destroyAllWindows()
    realsense.close()
