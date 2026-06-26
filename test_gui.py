# import tkinter as tk

# root = tk.Tk()

# root.title("Image Processing")
# root.geometry("500x400")

# label = tk.Label(root, text="Hello Image Processing")
# label.pack(pady=20)

# root.mainloop()
import cv2
import numpy as np
import matplotlib.pyplot as plt

def translate_image(img, tx, ty):
    rows, cols = img.shape[:2]
    M = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])
    return cv2.warpAffine(img, M, (cols, rows))

# --- الجزء الجديد لتشغيل الكود ورؤية النتيجة ---

# 1. اقرأ صورة من جهازك (حط مسار صورة عندك مكان 'your_image.jpg')
image = cv2.imread('cir.png') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # تحويل الألوان لتعرض بشكل صحيح في matplotlib

# 2. استدعاء الدالة (إزاحة 50 بكسل يمين، و 30 بكسل لتحت)
shifted_image = translate_image(image, 50, 30)

# 3. عرض الصورة الأصلية والمعدلة
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Translated Image")
plt.imshow(shifted_image)

plt.show()