import os
import json
import urllib

import h5py
import numpy as np
import pickle as pk

from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file

#Loading models
VGG16_model = VGG16(weights='imagenet')
print ("VGG16_model loaded")
DamgeDetection_model = load_model('static/models/car_damage-model3.h5')
print ("DamgeDetection_model loaded")
location_model = load_model('static/models/carLocationWithfrontAndBack3400Images.h5')
print ("Location model loaded")
severity_model = load_model('static/models/car_damage_severity.h5')
print ("Severity model loaded")
#checked
def prepare_img_224(img_path):
	img = load_img(img_path, target_size=(224, 224))
	x = img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

#checked
def prepare_img_128(img_path):
	img = load_img(img_path, target_size=(128 , 128))
	x = img_to_array(img)
	x = x.reshape((1,) + x.shape)/64
#    x = np.expand_dims(x, axis=0)
#    x = preprocess_input(x)
	return x

#checked
#validating provided image is car
def car_categories_gate(img_224, model):
    print ("Validating that this is a picture of your car...")
    out = model.predict(img_224)
    carList =['sports_car', 'minivan', 'grille', 'convertible', 'racer', 'beach_wagon', 'jeep', 'pickup', 'minibus', 
          'limousine', 'golfcart', 'cab', 'car_wheel', 'ambulance', 'police_van', 'Model_T', 
          'tow_truck', 'trailer_truck', 'coffeepot', 'car_mirror', 'snowmobile','chain_saw', 'snowplow',
          'fire_engine', 'recreational_vehicle', 'waffle_iron', 'harmonica', 'backpack', 
          'steam_locomotive', 'ashcan', 'espresso_maker', 'buckle', 'water_jug', 'projector',
          'seat_belt', 'motor_scooter',  'mailbox', 'tractor', 'moving_van', 'amphibian','half_track','garbage_truck','tank']
    label = decode_predictions(out)
    #print(label)
    predicted_List = label[0]
#    print(predicted_List)
    setFlagForImageDetection = False
    for tupleX in predicted_List:
        if setFlagForImageDetection == True:
            return True 
        else:
            for classNames in carList:
#                print(classNames)
                if classNames in tupleX[1]:
                    setFlagForImageDetection  = True              
                    return True
    
    if setFlagForImageDetection == False:
        return False
    
#checked
def prepare_img_256(img_path):
	img = load_img(img_path, target_size=(256, 256)) # this is a PIL image 
	x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
	x = x.reshape((1,) + x.shape)/255
	return x

#damges or undamaged
#checked
def car_damage_gate(img_256, model):
	print ("Validating that damage exists...")
	pred = model.predict(img_256)
	if pred[0][0] <=.65:
		return True # print "Validation complete - proceed to location and severity determination"
	else:
		return False
		# print "Are you sure that your car is damaged? Please submit another picture of the damage."
		# print "Hint: Try zooming in/out, using a different angle or different lighting"

#checked
#location of damaged

def location_assessment(img_128, model):
    print ("Determining location of damage...")
#    pred = model.predict(img_128)
    prediction = model.predict(img_128)
#    print(prediction[0])
    if prediction[0][0] <=.5:
#        print("Front Part Of car")
        return "Front Part Of car"
    else:
        return"Rear Part Of vehicle"
#       return "Rear Part Of vehicle"
#	pred_label = np.argmax(pred, axis=1)
#	d = {0: 'Front', 1: 'Rear', 2: 'Side'}
#	for key in d.iterkeys():
#		if pred_label[0] == key:
#			return d[key]
        
	# 		print "Assessment: {} damage to vehicle".format(d[key])
	# print "Location assessment complete."

def severity_assessment(img_256, model):
    print("Determining severity of damage...")
    pred_severe = model.predict(img_256)
    pred_label = np.argmax(pred_severe, axis=1)
    d = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}
    for key in d:
        if pred_label[0] == key:
            return "{} ".format(d[key])
			
	# 		print "Assessment: {} damage to vehicle".format(d[key])
	# print "Severity assessment complete."

# load models
def engine(img_path):
    # Load models and support
    #####VGG16_model = VGG16(weights='imagenet')
    #print ("First gate loaded")

    img_224 = prepare_img_224(img_path)
    g1 = car_categories_gate(img_224, VGG16_model)
    print(g1)
    if g1 is False:
        result = {'gate1': 'Car validation check: ', 
		'gate1_result': 0, 
		'gate1_message': {0: 'Are you sure this is a picture of your car? Please retry your submission.', 
		1: 'Hint: Try zooming in/out, using a different angle or different lighting'},
		'gate2': None,
		'gate2_result': None,
		'gate2_message': {0: None, 1: None},
		'location': None,
		'severity': None,
		'final': 'Damage assessment unsuccessful!'}
        return result
    
    #DamgeDetection_model = load_model('static/models/car_damage-model3.h5')
    #print ("Second gate loaded")
    	
    img_256 = prepare_img_256(img_path)
    g2 = car_damage_gate(img_256, DamgeDetection_model)

    if g2 is False:
        result = {'gate1': 'Car validation check: ', 
		'gate1_result': 1, 
		'gate1_message': {0: None, 1: None},
		'gate2': 'Damage presence check: ',
		'gate2_result': 0,
		'gate2_message': {0: 'Are you sure that your car is damaged? Please retry your submission.'},
		'location': None,
		'severity': None,
		'final': 'Damage assessment unsuccessful!'}
        return result
    #location_model = load_model('static/models/carLocationWithfrontAndBack3400Images.h5')
    #print ("Location model loaded")
    #severity_model = load_model('static/models/car_damage_severity.h5')
    #print ("Severity model loaded")

    img_128 = prepare_img_128(img_path)
    x = location_assessment(img_128, location_model)
    y = severity_assessment(img_256, severity_model)
	
    result = {'gate1': 'Car validation check: ', 
	'gate1_result': 1, 
	'gate1_message': {0: None, 1: None},
	'gate2': 'Damage presence check: ',
	'gate2_result': 1,
	'gate2_message': {0: None, 1: None},
	'location': x,
	'severity': y,
	'final': 'Damage assessment complete!'}
    return result
