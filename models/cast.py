import tensorflow as tf
from tensorflow import keras
import h5py

def top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true=y_true, y_pred=y_pred, k=2)

model=tf.keras.models.load_model('./models/sc_best_model.h5', custom_objects={'top_2': top_2})
print("sc_best_model model loaded")

#img= Path("sc_test/benign/729.jpg")
#650
#100,101,102,103,144
img_height, img_width, img_channels = 224,224,3
batch_size = 64
nb_classes = 2

test_images=[]
test_labels=[]

img = imread(img)
img = cv2.resize(img, (img_height, img_width))
test_images.append(img)


test_images = np.array(test_images, dtype=np.float32)
test_images = mobilenet_v2.preprocess_input(test_images)
test_labels = np.array(test_labels)
test_labels_cat = to_categorical(test_labels, num_classes=2)
print("="*50)
print("\n", test_images.shape, test_labels.shape, test_labels_cat.shape)
print("="*50)

preds = model.predict(test_images)
preds = np.argmax(preds, axis=-1)
print("="*50)
print("prediction is ",preds)
print("="*50)

print(preds[0])