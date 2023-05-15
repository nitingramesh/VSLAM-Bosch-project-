import cv2 

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
i = 0
flag = False
while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        
        if key==ord('s'):
            flag = True
        
        if flag:
            img_resized = cv2.imwrite(f'{i}.jpg', img=frame)
            cv2.waitKey(1650)
            img_new = cv2.imread(f'{i}.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            print("Processing image...")
            img_ = cv2.imread(f'{i}.jpg', cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")
            print("Resizing image to 28x28 scale...")
            img_ = cv2.resize(gray,(28,28))
            print("Resized...")
            img_resized = cv2.imwrite(f'{i}_final.jpg', img=img_)
            print("Image saved!")
            cv2.waitKey(1650)
            i+=1
        
        
        if key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
    