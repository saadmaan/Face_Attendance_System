from flask import Response
from flask import Flask, jsonify, send_file, send_from_directory
from flask import render_template
from Trainer_Recognition import train
from collect_images import collect
import threading
import argparse
import imutils
import time
import cv2
import numpy as np
import torch, torchvision
print(torchvision.__version__)
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.autograd import Variable
import pickle
import face_recognition
import os, os.path
from os import listdir
from os.path import isfile, join, exists
import glob
import random
import urllib
import json
import queue
import sqlite3
import concurrent.futures
from datetime import datetime, timedelta, date
from sklearn import neighbors
from facenet_pytorch import MTCNN
from csv import reader
from flask_cors import CORS
from enum import unique
from operator import itemgetter
from settings import *
mtcnn = MTCNN(image_size=160,keep_all=True,select_largest=False, post_process=False,device='cuda:0')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
global outputFrame, vs, flag, knn_Trained,q
lock = threading.Lock()
outputFrame = None
serial = 0
q=queue.Queue()
cdict = {}
CORS(app)
db = SQLAlchemy(app)
v =0
vs = cv2.VideoCapture("rtsp://admin:Genex@321@103.85.159.78:554/Streaming/Channels/201")
trdict = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'anti_spoofres.pth'
model_f = models.resnet18()
class model_ftt(nn.Module):
    def __init__(self, my_pretrained_model):
        super(model_ftt, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(nn.Linear(1000, 100),
                                           nn.ReLU(),
                                           nn.Linear(100, 2))  
    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        return x
model = model_ftt(my_pretrained_model=model_f)
# model.add_module(module=nn.Linear(list(model.children())[-1][-1].in_features, 2), name='fc')
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(PATH))
model.eval()
class_names = ['real', 'spoof']

#Users for registration and login access
class Users(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.Integer)
    name = db.Column(db.String(50))
    password = db.Column(db.String(50))
    admin = db.Column(db.Boolean)

    def json(self):
        return {'id': self.id, 'public_id':self.public_id, 'name': self.name, 'password':self.password, 'admin':self.admin}
        
    def get_all_users():
        '''function to get all users in our database'''
        return [Users.json(us) for us in Users.query.all()]

class MyEnum(str, enum.Enum):
    male = 'male'
    female = 'female'
    others = 'others'

#Employees who will be recognized and tracked
class Employee(db.Model):
    __tablename__ = 'employees'  # creating a table name
    employee_id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    designation = db.Column(db.String(80), nullable=False)
    department = db.Column(db.String(80), nullable=True)
    photo = db.Column(db.String(120), nullable=False)
    email_official = db.Column(db.String(120), nullable=False,unique=True)
    email_personal = db.Column(db.String(120), nullable=False, unique=True)
    phone_official = db.Column(db.String(16), nullable=False, unique=True)
    phone_personal = db.Column(db.String(16), nullable=False, unique=True)
    emergency_phone = db.Column(db.String(16), nullable=False,unique=True)
    joining_date = db.Column(db.DateTime(), nullable=False)
    date_of_birth = db.Column(db.DateTime(), nullable=False)
    gender = db.Column(db.Enum(MyEnum))
    attendance = db.relationship('Attendance', backref='employees', lazy=True)

    def json(self):
        return {'employee_id': self.employee_id, 'first_name': self.first_name, 'last_name': self.last_name, 'designation': self.designation, 'department': self.department, 'photo': self.photo, 'email_official': self.email_official, 'email_personal': self.email_personal, 'phone_official': self.phone_official, 'phone_personal': self.phone_personal, 'emergency_phone': self.emergency_phone, 'joining_date': self.joining_date, 'date_of_birth': self.date_of_birth, 'gender': self.gender}
            # this method we are defining will convert our output to json
        
    def add_emp(_employee_id, _first_name, _last_name, _designation, _department, _photo, _email_official, _email_personal, _phone_official, _phone_personal, _emergency_phone, _joining_date, _date_of_birth, _gender):
        # creating an instance of our emp constructor
        new_emp = Employee(employee_id=_employee_id, first_name=_first_name, last_name=_last_name, designation=_designation, department=_department, photo=_photo, email_official=_email_official, email_personal=_email_personal, phone_official=_phone_official, phone_personal=_phone_personal, emergency_phone=_emergency_phone, joining_date=_joining_date, date_of_birth=_date_of_birth, gender=_gender)
        db.session.add(new_emp)
        db.session.commit()
        
    def get_all_emps():
        '''function to get all emps in our database'''
        return [Employee.json(emp) for emp in Employee.query.all()]
    
    def get_emp(_employee_id):
        '''function to get emp using the id of the movie as parameter'''
        return [Employee.json(Employee.query.filter_by(id=_employee_id).first())]
    
    def update_emp(_employee_id, _first_name, _last_name, _designation, _department, _photo, _email_official, _email_personal, _phone_official, _phone_personal, _emergency_phone, _joining_date, _date_of_birth, _gender):
        user_to_update = Employee.query.filter_by(id=_employee_id).first()
        user_to_update.employee_id = _employee_id
        user_to_update.first_name = _first_name
        user_to_update.last_name = _last_name
        user_to_update.designation = _designation
        user_to_update.department = _department
        user_to_update.photo = _photo
        user_to_update.email_official = _email_official
        user_to_update.email_personal = _email_personal
        user_to_update.phone_official = _phone_official
        user_to_update.phone_personal = _phone_personal
        user_to_update.emergency_phone = _emergency_phone
        user_to_update.joining_date = _joining_date
        user_to_update.date_of_birth = _date_of_birth
        user_to_update.gender = _gender
        db.session.commit()
        
    def delete_emp(_employee_id):
        Employee.query.filter_by(id=_employee_id).delete()
        # filter emp by id and delete
        db.session.commit()

#Attendance history store
class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.employee_id"), nullable=False,unique=False)
    date = db.Column(db.String(50), nullable=False)
    time = db.Column(db.String(50), nullable=False)

    def json(self):
        return {'id': self.id, 'name': self.name, 'employee_id':self.employee_id, 'date':self.date, 'time':self.time}
        
    def get_all_attendance():
        '''function to get all attendance in our database'''
        return [Attendance.json(att) for att in Attendance.query.all()]
    
    def get_attendance(_employee_id):
        '''function to get attendance using the id of the movie as parameter'''
        return [Attendance.json(Attendance.query.filter_by(id=_employee_id).first())]

    def delete_attendance(_employee_id):
        Attendance.query.filter_by(id=_employee_id).delete()
        # filter attendance by id and delete
        db.session.commit()

#For Authentication purpose    
def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):

        token = None

        if 'x-access-tokens' in request.headers:
            token = request.headers['x-access-tokens']

        if not token:
            return jsonify({'message': 'a valid token is missing'})

        try:
            print('TOKEN', token)
            data = jwt.decode(token, options={"verify_signature": False})
            print(data)
            current_user = Users.query.filter_by(public_id=data['public_id']).first()
        except:
            return jsonify({'message': 'token is invalid'})

        return f(current_user, *args, **kwargs)
    return decorator

#Collect images for new person and re-train the recognition model
def knn_training(e_id, name):
    global vs, knn_Trained
    ename = e_id+'_'+name
    if not (os.path.isfile(f'./train_dirNew/{ename}')):
        os.mkdir(f'./train_dirNew/{ename}')
    dir = f'./train_dirNew/{ename}'
    collect(dir)
    train('train_dirNew')
    knn_Trained = pickle.load(open('knnSaved', 'rb'))

#Serve multiple purpose: Take daily attendance, take snapshot whenever a person enters, 
def markAttendance(namesFound, frame, faces):
    global cdict, trdict, v, serial
    nameList = []
    dateList = []
    with open('Attendance1.csv','r+') as f:
        myDataList = f.readlines()
        if len(myDataList)>1:
            i=1
            if myDataList[i]=='\n':
                while myDataList[i]=='\n':
                   i +=1 
            myDataList = myDataList[i:]
            for line in myDataList:
                namentry = line.split(',')[0]
                datentry = line.split(',')[2]
                if namentry is not '\n':
                    nameList.append(namentry)
                    dateList.append(datentry)
           
        for nameid,faceid in zip(namesFound,faces):
            if nameid is not 'unknown':  
                name = nameid.split('_')[1]
            else:
                name = 'unknown'
            cdate = date.today().strftime('%Y-%m-%d')
            if name not in nameList:
                if name in cdict.keys():
                    if (cdict[name][0] >23) and ((time.perf_counter()-cdict[name][1])<=2): 
                            t0 = time.perf_counter()
                            now = datetime.now() + timedelta(hours = 6)
                            dtString = now.strftime('%H:%M:%S')
                            if name == 'unknown':
                                nname = 'unknown'+str(v)
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                img = frame[faceid[0]-50:faceid[2]+50, faceid[3]-60:faceid[1]+60]
                                with lock:
                                    serial +=1
                                    ser = str(serial)
                                    cv2.imwrite(f'detected_persons1/{ser}_{cdate}_{dtString}_{nname}.jpeg', img)
                                    trdict[nname] = t0
                                    v=v+1
                            else:
                                e_id = nameid.split('_')[0]
                                # oname = name
                                with lock:
                                    f.writelines(f'\n{name},{e_id},{cdate},{dtString},{t0}')
                                    new_entry = Attendance(name = name, employee_id = e_id, date = cdate, time = dtString)
                                    db.session.add(new_entry)
                                    db.session.commit()
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                img = frame[faceid[0]-50:faceid[2]+50, faceid[3]-60:faceid[1]+60]
                                with lock:
                                    serial +=1
                                    ser = str(serial)
                                    cv2.imwrite(f'detected_persons1/{ser}_{cdate}_{dtString}_{name}.jpeg', img)
                                    trdict[name]=t0
                            with lock:
                                if name in cdict.keys():
                                    del cdict[name]
                    elif (cdict[name][0] <12) and ((time.perf_counter()-cdict[name][1])>2):
                        with lock:
                            if name in cdict.keys():
                                del cdict[name]
                    else:
                        tt = time.perf_counter()
                        if (tt-cdict[name][1])>3:
                            with lock:
                                if name in cdict.keys():
                                    del cdict[name]
                        else:
                            with lock:
                                cdict[name][0] = cdict[name][0]+1
                else:
                    ti = time.perf_counter()
                    with lock:
                        cdict[name] = [1, ti] 
                        
            else:
                if name in nameList:
                    if name in cdict.keys():     
                        if (cdict[name][0] >23) and ((time.perf_counter()-cdict[name][1])<=2):
                            t = time.perf_counter()
                            if name in trdict.keys():
                                dates = []
                                for n,d in zip(nameList,dateList):
                                    if n == name:
                                        dates.append(d)
                                now = datetime.now() + timedelta(hours = 6)
                                dtString = now.strftime('%H:%M:%S')
                                with lock:
                                    if cdate not in dates:
                                        e_id = nameid.split('_')[0]
                                        oname = name
                                        f.writelines(f'\n{oname},{e_id},{cdate},{dtString},{t}')
                                        new_entry = Attendance(name = oname, employee_id = e_id, date = cdate, time = dtString)
                                        db.session.add(new_entry)
                                        db.session.commit()
                                if t - trdict[name] >10:
                                    with lock:
                                        trdict[name] = t
                                    if name == 'unknown':
                                        nname = 'unknown'+str(v)
                                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        img = frame[faceid[0]-50:faceid[2]+50, faceid[3]-60:faceid[1]+60]
                                        with lock:
                                            serial +=1
                                            ser = str(serial)
                                            cv2.imwrite(f'detected_persons1/{ser}_{cdate}_{dtString}_{nname}.jpeg', img)
                                        v=v+1
                                    else:
                                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        img = frame[faceid[0]-50:faceid[2]+50, faceid[3]-60:faceid[1]+60]
                                        with lock:
                                            serial +=1
                                            ser = str(serial)
                                            cv2.imwrite(f'detected_persons1/{ser}_{cdate}_{dtString}_{name}.jpeg', img)
                                    with lock:
                                        if name in cdict.keys():
                                            del cdict[name]
                        elif (cdict[name][0] <12) and ((time.perf_counter()-cdict[name][1])>2):
                            with lock:
                                if name in cdict.keys():
                                    del cdict[name]
                        else:
                            tt = time.perf_counter()
                            if (tt-cdict[name][1])>3:
                                with lock:
                                    if name in cdict.keys():
                                        del cdict[name]
                            else:
                                with lock:
                                    cdict[name][0] = cdict[name][0]+1
                    else:
                        ti = time.perf_counter()
                        with lock:
                            cdict[name] = [1, ti] 

#Read each frame from IP camera and put that in Queue
def receive():
    global vs, outputFrame, q
    start_frame_number = 0
    # distance_threshold =0.6
    # FPS = 1/20
    # FPS_MS = int(FPS * 1000)
    while True:
            ret,frame = vs.read()
            if ret:
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              q.put(frame)

#process each frame, detect faces, recognize them and check liveliness
def recognize(frame):   
    global vs, outputFrame, q,v,cdict
    distance_threshold = 0.5
    names = []
    faces =[]
    dltfaces = []
    result = []
    while True:
        if q.empty() !=True:
          frame=q.get()
        fframe = frame.copy()
        wframe = frame.copy()
        width = int(wframe.shape[1] * .25)
        height = int(wframe.shape[0] * .25)
        dim = (width, height)
        wframe = cv2.resize(wframe, dim, interpolation = cv2.INTER_AREA)
        allfaces = mtcnn.detect(wframe)
        faces = allfaces[0]
        conf = allfaces[1]
        if faces is None or max(conf)<.99:
            with lock:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                outputFrame = frame.copy()
                continue
                # return
        else:
            faces = faces.astype(int)
            faces = faces.tolist()
            for i in range(len(faces)):
                faces[i][3], faces[i][0], faces[i][1], faces[i][2] = faces[i]
            faces =[tuple(faces[i]) for i in range(len(faces))]
            for i in range(len(faces)):
                faces[i] = faces[i][0]*4,faces[i][1]*4,faces[i][2]*4,faces[i][3]*4 

            #### FOR LIVELINESS DETECTION ########
            for (y1,x2,y2,x1) in faces:  
                    ratio = ((y2-y1)/frame.shape[0])*100
#                     if ratio >20:
                    if (height-y2>30 or y1>30):
                        face = frame[y1-0:y2+0,x1-0:x2+0]
                        face = cv2.resize(face,(224,224), cv2.INTER_AREA)
                        face = np.expand_dims(face, axis=0)
                        face = np.rollaxis(face, 3, 1)
                        face = face.astype("float64") / 255.0
                        face = Variable(torch.Tensor(face))
                        outputs = model(face)
                        _, preds = torch.max(outputs, 1)
                        label = class_names[preds[0]]
                        if label == 'real':
                            cv2.putText(frame, label, (x1,y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                            cv2.rectangle(frame, (x1, y1), (x2,y2),
                                (0, 0, 255), 2)
                        elif label == 'spoof':
                            cv2.putText(frame, label, (x1,y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                            cv2.rectangle(frame, (x1, y1), (x2,y2),
                            (0, 255, 0), 2)
            faces_encodings = face_recognition.face_encodings(frame,faces, num_jitters=1)   
            if len(faces)== 0 and len(faces_encodings)==0:
                with lock:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    outputFrame = frame.copy()
                    continue
                    # return
            for face in faces:
                sframe = frame[face[0]-10:face[2]+10, face[3]-10:face[1]+10]
                sframe = cv2.cvtColor(sframe, cv2.COLOR_RGB2BGR)
                width = int(wframe.shape[1] * .25)
                height = int(wframe.shape[0] * .25)
                dim = (width, height)
                sframe = cv2.resize(sframe, dim, interpolation = cv2.INTER_AREA)

            closest_distances = knn_Trained.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(faces))]

            match = [(name, loc, score) if rec else ("unknown", loc, score) for name, loc, score, rec in zip(knn_Trained.predict(faces_encodings), faces, conf, are_matches)]
            if len(match) == 0:
                with lock:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    outputFrame = frame.copy()
                    continue
                    # return
            name = [a[0] for a in match]
            loc = [a[1] for a in match]
            score = [a[2] for a in match]
            for n,l,s in zip(name,loc,score):
                y1,x2,y2,x1 = l
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.rectangle(frame,(int(x1),int(y2-35)),(int(x2),int(y2)),(0,255,0),cv2.FILLED)
                cv2.putText(frame,n,(int(x1+6),int(y2-6)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name, fframe, faces)
            
            with lock:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                outputFrame = frame.copy()
                # return

#Get processed frames one by one from 'recognize' function           
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock, vs
    # loop over frames from the output stream
    while True:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/register', methods=['GET', 'POST'])
def signup_user():  
    data = request.get_json()

    hashed_password = generate_password_hash(data['password'], method='sha256')

    new_user = Users(public_id=str(uuid.uuid4()), name=data['name'], password=hashed_password, admin=False) 
    db.session.add(new_user)  
    db.session.commit()    

    return jsonify({'message': 'registered successfully'})


@app.route('/login', methods=['GET', 'POST'])  
def login_user(): 
    auth = request.authorization

    if not auth or not auth.username or not auth.password:
        return make_response('could not verify 1', 401, {'WWW.Authentication': 'Basic realm: "login required"'})    

    user = Users.query.filter_by(name=auth.username).first()   
     
    if check_password_hash(user.password, auth.password):

        token = jwt.encode({'public_id': user.public_id, 'exp' : datetime.utcnow() + timedelta(minutes=30)}, app.config['SECRET_KEY'])  
        return jsonify({'token' : token})

    return make_response('could not verify 2',  401, {'WWW.Authentication': 'Basic realm: "login required"'})

@app.route('/users', methods=['GET'])
@token_required
def get_users(current_user):
    '''Function to get all the users in the database'''
    return jsonify({'Users': Users.get_all_users()})

@app.route('/employees', methods=['GET'])
@token_required
def get_emps(current_user):
    '''Function to get all the users in the database'''
    return jsonify({'Employees': Employee.get_all_emps()})

# route to get user by id
@app.route('/employees/<int:employee_id>', methods=['GET'])
@token_required
def get_emp_by_id(current_user,employee_id):
    return_value = Employee.get_emp(employee_id)
    return jsonify(return_value)

# route to add new user
@app.route('/employees', methods=['POST'])
@token_required
def add_emp(current_user):
    '''Function to add new emp to our database'''
    request_data = request.get_json()  # getting data from client
    request_data["date_of_birth"] = datetime.strptime(request_data["date_of_birth"], '%d/%m/%y')
    request_data["joining_date"] = datetime.strptime(request_data["joining_date"], '%d/%m/%y')
    Employee.add_emp(request_data["employee_id"], request_data["first_name"], request_data["last_name"],
                     request_data["designation"], request_data["department"], request_data["photo"],request_data["email_official"], request_data["email_personal"], request_data["phone_official"], request_data["phone_personal"], request_data["emergency_phone"], request_data["joining_date"], request_data["date_of_birth"], request_data["gender"])
    response = Response("Employee added", status = 201, mimetype='application/json')
    return response

# route to update user with PUT method
@app.route('/employees/<int:employee_id>', methods=['PUT'])
@token_required
def update_emp(current_user,employee_id):
    '''Function to edit user in our database using movie id'''
    request_data = request.get_json()
    request_data["date_of_birth"] = datetime.strptime(request_data["date_of_birth"], '%d/%m/%y')
    request_data["joining_date"] = datetime.strptime(request_data["joining_date"], '%d/%m/%y %H:%M:%S')
    Employee.update_emp(employee_id, request_data["first_name"], request_data["last_name"],
                 request_data["designation"], request_data["department"], request_data["photo"],request_data["email_official"], request_data["email_personal"], request_data["phone_official"], request_data["phone_personal"], request_data["emergency_phone"], request_data["joining_date"], request_data["date_of_birth"], request_data["gender"])
    response = Response("Employee Updated", status=200, mimetype='application/json')
    return response

# route to delete employees using the DELETE method
@app.route('/employees/<int:employee_id>', methods=['DELETE'])
@token_required
def remove_emp(current_user,employee_id):
    '''Function to delete movie from our database'''
    Employee.delete_emp(employee_id)
    response = Response("Employee Deleted", status=200, mimetype='application/json')
    return response

# showing all the attendance listed
@app.route("/attendance")
# @token_required
def show_data(current_user):
    testData = []
    with open('Attendance1.csv','r') as f:
        csv_reader = reader(f)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            for row in csv_reader:
                if len(row)>2:
                    dict = {'name': row[0], 'employee-id': row[1], 'date':row[2], 'time': row[3]}
                    testData.append(dict)
        return jsonify(testData)
    
@app.route("/tracking")
def tracking():
    snapData = []
    global serial
    leng = len(os.listdir('detected_persons1'))
    if leng >0:
        for file in os.listdir('detected_persons1'):
            name = file.split('_')[3].split('.')[0]
            date = file.split('_')[1]
            time = file.split('_')[2]
            seria = int(file.split('_')[0])
            dict = {'serial': seria, 'imageURL': f'http://103.85.159.70:8005/images/{file}', 'name': name,'date': date, 'time': time}
            snapData.append(dict)
    snapData = sorted(snapData, key=itemgetter('serial'),reverse=True)
    return jsonify(snapData)

@app.route("/images/<path:path>")
def static_dir(path):
    return send_from_directory("detected_persons1", path)

@app.route('/training', methods=['POST'])
# @token_required
def add_training():
    request_data = request.get_json()
    e_id = request_data['employee_id']
    name = request_data['name']
    knn_training(e_id, name)
    response = Response("Training is done with new data", status=200, mimetype='application/json')
    return response  

if __name__ == '__main__':
    knn_Trained = pickle.load(open('knnSaved', 'rb'))
    
    with open('Attendance1.csv','r+') as f:
        
        nameList = []
        timeList = []     
        myDataList = f.readlines()
        if len(myDataList)>0:
            i=1
            if myDataList[i]=='\n':
                while myDataList[i]=='\n':
                   i +=1 
            myDataList = myDataList[i:]
            for line in myDataList:
                namentry = line.split(',')[0]
                timentry = line.split(',')[4]
                if namentry is not '\n':
                    nameList.append(namentry)
                    timeList.append(timentry)
        for name,time in zip(nameList,timeList):
            trdict[name]=float(time)
    leng = len(os.listdir('detected_persons1'))
    slist = []
    if leng >0:
        for file in os.listdir('detected_persons1'):
            seri = file.split('_')[0]
            seri = int(seri)
            slist.append(seri)
        serial = max(slist)
    else:
        serial = 1
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    t0 = threading.Thread(target=receive)
    t0.daemon = True
    t0.start()  
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.submit(receive)
    #     while True:
    #         if q.empty() !=True:
    #             frame = q.get()
                
    #             executor.submit(recognize, frame = frame)
    #             app.run(host=args["ip"], port=args["port"], debug=True,
    #     threaded=True, use_reloader=False)
    #             print("INSIDE THHHRRRH")
    # print("OUTSIDE THFREEEE")

    # start a thread that will perform face recognition
      
    t1 = threading.Thread(target=recognize)
    t1.daemon = True
    t1.start()
    t2 = threading.Thread(target=recognize)
    t2.daemon = True
    t2.start()
    t3 = threading.Thread(target=recognize)
    t3.daemon = True
    t3.start()
    t4 = threading.Thread(target=recognize)
    t4.daemon = True
    t4.start()
    t5 = threading.Thread(target=recognize)
    t5.daemon = True
    t5.start()
    t6 = threading.Thread(target=recognize)
    t6.daemon = True
    t6.start()
    # t7 = threading.Thread(target=recognize)
    # t7.daemon = True
    # t7.start()
    # t8 = threading.Thread(target=recognize)
    # t8.daemon = True
    # t8.start()

    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
    
# release the video stream pointer
vs.release()

