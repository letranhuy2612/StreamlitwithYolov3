import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def obj_detection(my_img):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    column1, column2 = st.columns(2)

    column1.subheader("Input image")
    st.text("")
    plt.figure(figsize = (16,16))
    plt.imshow(my_img)
    column1.pyplot(use_column_width=True)

    # YOLO model
    net = cv2.dnn.readNet("D:\\Workspace\\Yolov3\\yolov3.weights","D:\\Workspace\\Yolov3\\yolov3.cfg")

    labels = []# Khởi tạo mảng để lưu trữ các nhãn đầu ra 
    with open("D:\\Workspace\\Yolov3\\coco.names", "r") as f:
        #labels = f.read().rstrip('\n').split('\n')
        labels = [line.strip() for line in f.readlines()]#xóa các khoảng trắng ở đầu và cuối khỏi các chuỗi nhãn
    #Lưu trữ tên các lớp của mô hình thu được
    names_of_layer = net.getLayerNames()
    output_layers = [names_of_layer[i-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(len(labels), 3))#Các giá trị #RGB được chọn ngẫu nhiên từ 0 đến 255    


    # Tải ảnh
    newImage = np.array(my_img.convert('RGB'))
    img = cv2.cvtColor(newImage,1)
    height,width,channels = img.shape


    # Objects detection 
    #Chuyển đổi hình ảnh thành các đốm màu
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   #(image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    classID = []#Array để lưu trữ các nhãn đầu ra
    confidences = []#Array để lưu trữ các hộp connfidences score
    boxes =[]#Array để lưu trữ các kích thước hộp giới hạn

    #Hiển thị thông tin của lớp đầu ra
    for op in outputs:
        for detection in op:
            scores = detection[5:]
            class_id = np.argmax(scores)  
            confidence = scores[class_id] 
            if confidence > 0.5:   
                # OBJECT DETECTED
                #lấy ra các center,width,height của đối tượng
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  #width is the original width of image
                h = int(detection[3] * height) #height is the original height of the image

                # Tính toán tọa độ của hộp giới hạn
                x = int(center_x - w /2)   #Top-Left x
                y = int(center_y - h/2)   #Top-left y

                #Sắp xếp các đối tượng được phát hiện trong một mảng
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                classID.append(class_id)

    #Điều chỉnh ngưỡng tin cậy và ngưỡng NMS (Không triệt tiêu tối đa)
    score_threshold = st.sidebar.slider("Confidence_threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)      
    print(indexes)

   # font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            #To get the name of object
            label = str.upper((labels[classID[i]]))   
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
            cv2.putText(img,f'{labels[classID[i]].upper()} {int(confidences[i]*100)}%',
                    (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,color,2)     
            items.append(label)


    st.text("")
    column2.subheader("Output image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    column2.pyplot(use_column_width=True)

    if len(indexes)>1:
        st.success("Found {} Objects - {}".format(len(indexes),[item for item in set(items)]))
    else:
        st.success("Found {} Object - {}".format(len(indexes),[item for item in set(items)]))


def main():
    
    st.title("Streamlit app")
    st.write("You can view real-time object detection done using YOLO model here. Select one of the following options to proceed:")

    choice = st.radio("", ("See an illustration", "Choose an image of your choice"))
    #st.write()

    if choice == "Choose an image of your choice":
        #st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Upload", type=['jpg','png','jpeg'])

        if image_file is not None:
            my_img = Image.open(image_file)  
            obj_detection(my_img)

    elif choice == "See an illustration":
        my_img = Image.open("D:\\Workspace\\Yolov3\\hanhyojoo.jpg")
        obj_detection(my_img)

if __name__ == '__main__':
    main()
