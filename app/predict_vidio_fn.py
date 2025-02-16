# C:\Users\ASUS\Downloads\archive\Dataset\Test
import cv2
import os
import numpy as np



X=[]
Y=[]
def read_img(fullpath):
        data_list = []
        file_path = os.path.join(fullpath)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
        data_list.append(res)
        # label = dirPath.split('/')[-1]

        # label_list.remove("./training")
        return (np.asarray(data_list, dtype=np.float32))


num_classes =2
batch_size = 100
epochs = 10
#------------------------------

import os, cv2, keras
import numpy as np
from keras.models import Sequential

def predictvideofn(fname):


        vidObj = cv2.VideoCapture(fname)

        # Used as counter variable
        count = 0

        # checks whether frames were extracted
        success = 1
        resultlist=[]
        tlist=[]
        flist=[]
        while success:
                # vidObj object calls read
                # function extract frames
                success, image = vidObj.read()
                if not success:
                        break
                # Saves the frames with frame-count
                cv2.imwrite(r"C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\media\frame\frame%d.jpg" % count, image)


                if count%25==0:
                        import tensorflow as tf

                        # Clear the current session
                        tf.keras.backend.clear_session()
                        from keras.engine.saving import load_model
                        # manipulate with numpy,load with panda
                        import numpy as np

                        model = load_model(r"C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\app\model_video_new.h5")

                        X=read_img(r"C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\media\frame\frame%d.jpg" % count)



                        X = np.array(X, 'float32')


                        X /= 255  # normalize inputs between [0, 1]


                        x_train = X.reshape(X.shape[0], 48,48, 1)

                        from sklearn.metrics import confusion_matrix
                        yp=model.predict(x_train,verbose=0)
                        print(yp[0])
                        print(yp[0][0]>yp[0][1])
                        ypindex=np.argmax(yp[0])
                        resultlist.append(ypindex)
                        truncated_num = yp[0][ypindex]
                        if ypindex == 0:
                                flist.append(truncated_num)
                        else:
                                tlist.append(truncated_num)
                        s = len(resultlist) - sum(resultlist)
                        if s>10:
                                break

                        print(ypindex)
                count += 1
        s=len(resultlist)-sum(resultlist)
        p=s/len(resultlist)
        if p>0.2 or s>10:
                ypindex=0
                truncated_num=sum(flist)/len(flist)

        else:
                ypindex = 1
                truncated_num = sum(tlist) / len(tlist)
        truncated_num = math.trunc(truncated_num * 100*100) / 100
        response = {"prediction": int(ypindex),"per":truncated_num}
        print(response)
        import os

        # Specify the file path
        file_path = r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\example.txt'

        # Open the file in write mode ('w'), this will clear the file if it exists
        with open(file_path, 'w') as file:
                file.write(str(ypindex) + "#" + str(int(truncated_num)))

                return response
        # return ypindex,truncated_num
import math
# #
# # num = 3.149
# # truncated_num = math.trunc(num * 100) / 100
# # print(truncated_num)  # Output: 3.14
r=input()
predictvideofn(r)