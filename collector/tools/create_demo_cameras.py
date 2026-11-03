#!/usr/bin/env python3
"""Create local camera demo feeds (video-only). No fake human labels."""
from pathlib import Path
import cv2
import numpy as np

out=Path(__file__).resolve().parent/'demo_feeds';out.mkdir(exist_ok=True)
for cid,bg,theta in [('cam01',(39,46,80),0),('cam02',(57,71,42),24)]:
    dest=out/f'{cid}.mp4'
    w=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),20,(960,540))
    if not w.isOpened():raise RuntimeError('mp4v encoder not available')
    for i in range(500):
        img=np.zeros((540,960,3),dtype=np.uint8)
        img[:]=bg
        shift=int(10*np.sin(i/22+theta))
        # Draw table & a stylized silhouette. These are camera-test feeds,
        # not real pose landmarks for model training.
        cv2.rectangle(img,(240,320),(735,410),(102,108,129),-1)
        cv2.circle(img,(480+shift,175),50,(185,180,193),-1)
        cv2.line(img,(480+shift,225),(480+shift,330),(196,192,209),20)
        cv2.line(img,(480+shift,260),(370+shift,323),(190,190,211),18)
        cv2.line(img,(480+shift,260),(570+shift,320),(190,190,211),18)
        cv2.putText(img,f'{cid.upper()}  -  LOCAL DEMO  -  {i/20:.2f}s', (28,46),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(232,237,249),2,cv2.LINE_AA)
        cv2.putText(img,'NOT FOR TRAINING / NO REAL SUBJECT', (26,505),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(195,211,245),2,cv2.LINE_AA)
        w.write(img)
    w.release()
    print(dest)
