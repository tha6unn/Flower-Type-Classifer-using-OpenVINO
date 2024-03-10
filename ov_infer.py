import openvino as ov
import cv2
import numpy as np

core = ov.Core()

model = core.read_model('flower.xml')

compiled_model = core.compile_model(model=model) 
output_key = compiled_model.output(0)

op='rose1.jpg'
img_op=cv2.imread(op)
img_op=cv2.resize(img_op,(180,180))
img_op=np.expand_dims(img_op,axis=0)

op = compiled_model(img_op)[output_key]
op=np.argmax(op)

class_labels=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
print(class_labels[op])