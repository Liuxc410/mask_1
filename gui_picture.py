import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch



# 创建主窗口
root = tk.Tk()

root.title("口罩佩戴检测(图片模式)")
root.geometry("1100x600")
root.configure(bg="white")
# 创建画布2
canvas2 = tk.Canvas(root, width=500, height=500)
canvas2.place(x=540, y=70)

model = torch.load('./model/mask_50_2.pth')
# 转为Blob格式函数
def imgBlob(img):
    # 转为Blob
    img_blob = cv2.dnn.blobFromImage(img,1,(100,100),(104,177,123),swapRB=True)
    # 维度压缩
    img_squeeze = np.squeeze(img_blob).T
    # 旋转
    img_rotate = cv2.rotate(img_squeeze,cv2.ROTATE_90_CLOCKWISE)
    # 镜像
    img_flip =  cv2.flip(img_rotate,1)
    # 去除负数，并归一化
    img_blob = np.maximum(img_flip,0) / img_flip.max()
    return img_blob
# 全局变量，存储选择的图片路径
selected_image_path = ""
# 创建画布1
canvas1 = tk.Canvas(root, width=500, height=500)
canvas1.place(x=20, y=70)
# 定义按钮A的函数
def show_image():
    global selected_image_path
    # 打开文件对话框，让用户选择图片文件
    file_path = filedialog.askopenfilename(initialdir="./test_data/",
                                           title="选择图片", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("all files", "*.*")))
    if file_path:
        # 使用PIL库打开图片文件，并将其显示在画布1上
        img = Image.open(file_path)
#        img = img.resize((500, 500), Image.ANTIALIAS)
        img = img.resize((500, 500), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(img)
        print(img_tk)
        canvas1.create_image(0, 0, anchor="nw", image=img_tk)
        canvas1.image = img_tk
        # 存储选择的图片路径到全局变量中
    selected_image_path = file_path
    #print(selected_image_path)
# 定义函数，获取canvas1的图片，将其人脸加上方框，并在canvas2中显示
def process_image():
    # 设置文本样式
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    color = (0, 0, 255)
    thickness = 2
    global selected_image_path
    # 使用OpenCV库读取选择的图片
    img = cv2.imread(selected_image_path)
    # 使用deploy.prototxt.txt和res10_300x300_ssd_iter_140000.caffemodel文件来加载人脸检测模型
    model = cv2.dnn.readNetFromCaffe('./model/deploy.prototxt.txt',
                                   './model/res10_300x300_ssd_iter_140000.caffemodel')
    # 将图片转换为blob格式
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # 将blob输入模型中进行人脸检测
    model.setInput(blob)
    detections = model.forward()
    # 遍历检测到的人脸
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # 判断置信度是否足够高
        if confidence > 0.5:
            # 获取人脸方框的坐标
            box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            # 提取人脸ROI
            face_roi = img[startY:endY, startX:endX]

            if face_roi is None:
                cv2.putText(img, "no_person", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                            2)
                # cv2.putText(frame, "no person", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            else:
                img2 = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                # 缩放图像并转换为数组
                img2 = cv2.resize(img2, (128, 128))
                img_array = np.array(img2)
                # 增加一维，以符合模型输入形状
                input_image = np.expand_dims(img_array, axis=0)
                # 进行预测并输出结果
                #predictions = model.predict(input_image)
                resized_image = cv2.resize(input_image, (300, 300))
                model.setInput(cv2.dnn.blobFromImage(resized_image, scalefactor=1.0/255, size=(300, 300), swapRB=True, crop=False))
                predictions = model.forward()
                # print(predictions)
                predicted_class_index = np.argmax(predictions[0])
                print(predicted_class_index)
                a = predicted_class_index
                file_path = "./txt/my_dict3.txt"
                my_dict = {}
                with open(file_path, 'r') as f:
                    for line in f:
                        key, value = line.strip().split(":")
                        my_dict[key.strip()] = int(value.strip())

                print(my_dict)
                # 将字典中的键值对翻转，变为以值为键，以键为值的形式
                inverted_dict = {v: k for k, v in my_dict.items()}
                # 输出预测结果对应的分类名称
                predicted_class_name = inverted_dict.get(a)
                print(predicted_class_name)

                label = str(predicted_class_name)
                print(label)
                color = (0, 255, 0)
                cv2.putText(img, str(label), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

    # 将处理后的图片显示在canvas2中
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((500, 500), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    canvas2.create_image(0, 0, anchor="nw", image=img_tk)
    canvas2.image = img_tk
# 定义函数，关闭GUI界面
def close_window():
    root.destroy()
# 定义函数，删除canvas1和canvas2中的图片
def delete_images():
    canvas1.delete("all")
    canvas1.image = None
    canvas2.delete("all")

def main():
    # 创建按钮A
    btn_a = tk.Button(root, text="打开", command=show_image)
    btn_a.place(x=20, y=20, width=100, height=30)
    # 创建按钮B
    btn_b = tk.Button(root, text="测试", command=process_image)
    btn_b.place(x=150, y=20, width=100, height=30)
    # 创建按钮C
    btn_c = tk.Button(root, text="删除",command=delete_images)
    btn_c.place(x=280, y=20, width=100, height=30)
    # 创建按钮D
    btn_d = tk.Button(root, text="关闭",command=close_window)
    btn_d.place(x=410, y=20, width=100, height=30)
    # 进入主循环
    root.mainloop()

if __name__ == '__main__':
    main()
