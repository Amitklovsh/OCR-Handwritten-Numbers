import cv2
import numpy as np

img = np.full((600, 800, 3), 255, np.uint8)
rows, cols = 6, 8
h, w = img.shape[:2]
for r in range(rows + 1):
    y = int(r * h / rows)
    cv2.line(img, (0, y), (w - 1, y), (0, 0, 0), 2)
for c in range(cols + 1):
    x = int(c * w / cols)
    cv2.line(img, (x, 0), (x, h - 1), (0, 0, 0), 2)
for r in range(rows):
    for c in range(cols):
        y = int((r + 0.6) * h / rows)
        x = int((c + 0.3) * w / cols)
        val = (r * cols + c) % 10
        cv2.putText(img, str(val), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
cv2.imwrite('/workspace/synth_table.png', img)
print('saved /workspace/synth_table.png')