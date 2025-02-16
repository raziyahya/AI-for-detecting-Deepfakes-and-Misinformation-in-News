# C:\Users\ASUS\Downloads\archive\Dataset\Test
import cv2
import os
import numpy as np
import math



X=[]
Y=[]

#------------------------------

import os, cv2, keras
import numpy as np


def read_img(path):
        data_list=[]
        file_path = os.path.join(path)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
        data_list.append(res)
        # label = dirPath.split('/')[-1]

        # label_list.remove("./training")
        return (np.asarray(data_list, dtype=np.float32))


def predictfn(fname):
        # C:\Users\ASUS\Downloads\archive\Dataset\Test
        import cv2
        import os
        import numpy as np

        X = []
        Y = []



        # ------------------------------

        import os, cv2, keras
        import numpy as np
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D
        from keras.engine.saving import load_model
        # manipulate with numpy,load with panda
        import numpy as np
        import tensorflow as tf

        # Reset the default graph and session
        tf.keras.backend.clear_session()
        model = load_model(r"D:\germany\Deep Fake\Deepfake\app\model_img_new.h5")

        X = read_img(fname)

        X = np.array(X, 'float32')

        X /= 255  # normalize inputs between [0, 1]

        x_train = X.reshape(X.shape[0], 48,48, 1)

        # load weights
        from sklearn.metrics import confusion_matrix
        yp1 = model.predict(x_train)
        print(yp1)
        yp = model.predict_classes(x_train, verbose=0)
        print(yp)
        truncated_num = math.trunc(yp1[0][yp[-1]] * 100) / 100
        return yp[-1], truncated_num
def predictfn(fname):
        # from keras.layers import Dense, Dropout, Flatten
        # from keras.layers import Conv2D, MaxPooling2D
        # from keras.engine.saving import load_model
        # # manipulate with numpy,load with panda
        # import numpy as np
        # import tensorflow as tf
        #
        # # Reset the default graph and session
        # tf.keras.backend.clear_session()
        # model = load_model(r"D:\germany\Deep Fake\Deepfake\app\model11.h5")

        from tensorflow.keras.models import load_model

        # Load the model with custom_objects
        model = load_model(r"model_img_new.h5")

        X=read_img(fname)


        X = np.array(X, 'float32')


        X /= 255  # normalize inputs between [0, 1]


        x_train = X.reshape(X.shape[0], 48,48,1)





         # load weights
        from sklearn.metrics import confusion_matrix
        yp=model.predict(x_train)
        print("y===",yp)
        ypindex=np.argmax(yp[0])
        truncated_num = math.trunc(yp[0][ypindex] * 100*100) / 100
        print(yp)
        return ypindex,truncated_num

