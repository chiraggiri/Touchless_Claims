import os
import json
#import urllib
#import h5py
import numpy as np
import pickle as pk
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import  img_to_array, load_img
from keras.models import load_model
from keras.utils.data_utils import get_file
from os.path import join, dirname, realpath
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash
from werkzeug.utils import secure_filename

# A <form> tag is marked with enctype=multipart/form-data and an <input type=file> is placed in that form.
# The application accesses the file from the files dictionary on the request object.
# use the save() method of the file to save the file permanently somewhere on the filesystem.

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/') # where uploaded files are stored
ALLOWED_EXTENSIONS = set(['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF']) # models support png and gif as well
#Loading models
first_gate = VGG16(weights='imagenet')
print ("First gate loaded")
second_gate = load_model('static/models/car_damage-model3.h5')
print ("Second gate loaded")
location_model = load_model('static/models/carLocationWithfrontAndBack3400Images.h5')
print ("Location model loaded")
severity_model = load_model('static/models/car_damage_severity.h5')
print ("Severity model loaded")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # max upload - 10MB
app.secret_key = 'secret'

with open('static/models/vgg16_cat_list.pk', 'rb') as f:
        cat_list = pk.load(f)
        print ("Cat list loaded")
# check if an extension is valid and that uploads the file and redirects the user to the URL for the uploaded file
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



# from Keras GitHub  
CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

def get_predictions(preds, top=5):
	global CLASS_INDEX
	if len(preds.shape) != 2 or preds.shape[1] != 1000:
		raise ValueError('`decode_predictions` expects '
						 'a batch of predictions '
						 '(i.e. a 2D array of shape (samples, 1000)). '
						 'Found array with shape: ' + str(preds.shape))
	if CLASS_INDEX is None:
		fpath = get_file('imagenet_class_index.json',
						 CLASS_INDEX_PATH,
						 cache_subdir='models')
		CLASS_INDEX = json.load(open(fpath))
	l = []
	for pred in preds:
		top_indices = pred.argsort()[-top:][::-1]
		indexes = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
		indexes.sort(key=lambda x: x[2], reverse=True)
		l.append(indexes)
	return l

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
	top = get_predictions(out, top=5)
	for j in top[0]:
		if j[0:2] in cat_list:
			# print j[0:2]
			return True 
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
	print(pred[0][0])
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
    #####first_gate = VGG16(weights='imagenet')
    print ("First gate loaded")

    img_224 = prepare_img_224(img_path)
    g1 = car_categories_gate(img_224, first_gate)

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
    
    #second_gate = load_model('static/models/car_damage-model3.h5')
    #print ("Second gate loaded")
    	
    img_256 = prepare_img_256(img_path)
    g2 = car_damage_gate(img_256, second_gate)

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

@app.route('/')
def home():
	return render_template('index.html', result=None)

@app.route('/<a>')
def available(a):
	flash('{} coming soon!'.format(a))
	return render_template('index.html', result=None, scroll='third')

@app.route('/assessment')
def assess():
	return render_template('index.html', result=None, scroll='third')


@app.route('/assessment', methods=['GET', 'POST'])
def upload_and_classify():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(url_for('assess'))
		
		file = request.files['file']

		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(url_for('assess'))

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename) # used to secure a filename before storing it directly on the filesystem
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			# return redirect(url_for('uploaded_file',
			#                         filename=filename))
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			model_results =engine(filepath)

			return render_template('results.html', result=model_results, scroll='third', filename=filename)
	
	flash('Invalid file format - please try your upload again.')
	return redirect(url_for('assess'))

# @app.route('/show/<filename>')
# def uploaded_file(filename):
#     return render_template('template.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Now one last thing is missing: the serving of the uploaded files. 
# In the upload_file() we redirect the user to url_for('uploaded_file', filename=filename), 
# that is, /uploads/filename. So we write the uploaded_file() function to return the file of that name. 

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],
							   filename)

#http://127.0.0.1:5000/
if __name__ == '__main__': 
	app.run(host='127.0.0.1', port=8080, debug=True, use_reloader=False) # remember to set back to False	