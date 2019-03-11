import tensorflow as tf
import parameters as par
import cv2
import numpy as np
from PIL import ImageOps, Image

saver = tf.train.import_meta_graph(par.saved_path + str('501.meta'))
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./Saved/'))

    # Get Operations to restore
    graph = sess.graph

    # Get Input Graph
    X = graph.get_tensor_by_name('Input:0')
    #Y = graph.get_tensor_by_name('Target:0')
    # keep_prob = tf.placeholder(tf.float32)
    keep_prob = graph.get_tensor_by_name('Placeholder:0')

    # Get Ops
    prediction = graph.get_tensor_by_name('prediction:0')
    logits = graph.get_tensor_by_name('logits:0')
    accuracy = graph.get_tensor_by_name('accuracy:0')

    # Get the image
    count = 0
    while 1:
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        if ret:
            cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
            crop_img = img[100:300, 100:300]
            grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            value = (35, 35)
            blurred = cv2.GaussianBlur(grey, value, 0)
            _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            img = cv2.resize(img, dsize=None, fx=1.5, fy=1.5)
            img = cv2.flip(img, 1) # To flip image
            #cv2.imshow('title',thresh1)
            cv2.imshow('frame',img)
            thresh1 = (thresh1 * 1.0) / 255
            thresh1 = Image.fromarray(thresh1)
            thresh1 = ImageOps.fit(thresh1, [par.image_size, par.image_size])
            if par.threshold:
                testImage = np.reshape(thresh1, [-1, par.image_size, par.image_size, 1])
            else:
                testImage = np.reshape(thresh1, [-1, par.image_size, par.image_size, 3])
            testImage = testImage.astype(np.float32)
            if count == 0: # First iteration
                testY = sess.run(prediction, feed_dict={X: testImage, keep_prob: 1.0})
                testY_previous = testY
            else: 
                testY_previous = testY
                testY = sess.run(prediction, feed_dict={X: testImage, keep_prob: 1.0})
            count += 1
            print(testY)
            # Print predicted letter, only if it has changed since the last prediction
            for i in range(len(testY[0])):
                if testY[0][i] != testY_previous[0][i]:
                    if testY[0][0] == [1]:
                        print("A")
                    elif testY[0][1] == [1]: 
                        print("B")
                    elif testY[0][2] == [1]:
                        print("C")
                    elif testY[0][3] == [1]:
                        print("D")
                    elif testY[0][4] == [1]:
                        print("G")
                    elif testY[0][5] == [1]:
                        print("I")
                    elif testY[0][6] == [1]:
                        print("L")
                    elif testY[0][7] == [1]:
                        print("V")
                    elif testY[0][8] == [1]:
                        print("Y")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            continue
    
    cap.release()
    cv2.destroyAllWindows()
