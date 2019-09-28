#importing libraries
import datetime
import tempfile
from os import path

from model import CSRNet
import torch
from image import *
import PIL.Image as Image
from flask import jsonify, render_template, send_from_directory
from flask import request
from flask import Flask,Response
from flask import Flask
import logging
import sqlite3
from sqlite3 import Error
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
from VideoGet import VideoGet
import tensorflow as tf
from torchvision import datasets, transforms
import time
import yaml
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

model = CSRNet()
if torch.cuda.is_available():
   model.cuda()
from flask import g
graph = tf.get_default_graph()

DATABASE = 'peopleCount.db'
table_name = 'peopleCount'

fn_yaml = "cam1.yml"
last_pos=0
with open(fn_yaml, 'r') as stream:
    observ_points = yaml.load(stream)

contours=[]
bounding_rects=[]
sec_to_wait = 5
allpoints=[]
if observ_points != None:
    for square in observ_points:
        points = np.array(square['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        #points_shifted[:,0] = points[:,0] - rect[0] # shift contour to region of interest
        #points_shifted[:,1] = points[:,1] - rect[1]
        contours.append(points)
        bounding_rects.append(rect)
        allpoints.append(points)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html')
def Motion():
    return render_template('index.html')


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

@app.route('/images/<path:path>')
def send_gfx(path):
    return send_from_directory('images', path)

@app.route('/js/<path:path>')
def send_t(path):
    return send_from_directory('js', path)

@app.route('/sass/<path:path>')
def send_sass(path):
    return send_from_directory('sass', path)

@app.route('/<path:path>')
def get_file(path):
    return send_from_directory('', path)


@app.route('/get_attend/<path:path>')
def get_attend(path):
    return app.send_static_file('../attendance',path)


@app.route('/getattend')
def getattend():
    with app.app_context():
        c = get_db().cursor()
        c.execute("SELECT * FROM " + table_name)
        rows = c.fetchall()
        return jsonify(rows)

@app.route('/getlastattend')
def getlastattend():
    camid = request.args.get("camid")
    locid = request.args.get("locid")
    if camid is not None  and locid is not None:
        with app.app_context():
            imgname = 'croped1_'+locid+'.jpg'
            if path.exists(imgname):
                if torch.cuda.is_available():
                    # img = transform(Image.fromarray(croped_frame).convert('RGB')).cuda()
                    img = transform(Image.open(imgname).convert('RGB')).cuda()
                else:
                    # img = transform(Image.fromarray(croped_frame).convert('RGB'))
                    img = transform(Image.open(imgname).convert('RGB'))
                output = model(img.unsqueeze(0))
                x = int(output.detach().cpu().sum().numpy())
                timess = datetime.datetime.now()
                c = get_db().cursor()
                c.execute("INSERT INTO " + table_name + " VALUES (?,?,?,?)", (x, timess, 1, locid))
                get_db().commit()
                c.execute("SELECT peoplecnt FROM " + table_name + " where camno = "+camid+" and locid = "+locid+" order by logtime desc limit 1 ")
                row = c.fetchone()
                return jsonify(row)
    return jsonify("")


def get_db():
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(DATABASE)
        return db


@app.teardown_appcontext
def close_connection(exception):
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "not ok", "message": "this server could not understand your request"})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "not found", "message": "route not found"})


@app.errorhandler(500)
def not_found(e):
    return jsonify({"status": "internal error", "message": "internal error occurred in server"})


@app.route('/detect', methods=['POST'])
def postimage():
    file = request.files.get('upload')
    filename, ext = os.path.splitext(file.filename)
    if ext not in ('.png', '.jpg', '.jpeg'):
         return 'File extension not allowed.'
        # loading the trained weights

    tmp = tempfile.TemporaryDirectory()
    temp_storage = path.join(tmp.name, file.filename)
    file.save(temp_storage)
    img = transform(Image.open(temp_storage).convert('RGB')).cuda()
    output = model(img.unsqueeze(0))
    timess = str(datetime.datetime.now())
    x=int(output.detach().cpu().sum().numpy())
    print("Predicted Count : ", int(output.detach().cpu().sum().numpy()+10))
    print('time: ',timess)
    with app.app_context():
        c = get_db().cursor()
        c.execute("INSERT INTO " + table_name + " VALUES (" + str(x) + ", '" + timess + "')")
        get_db().commit()
    return jsonify(int(output.detach().cpu().sum().numpy()+10),timess)

def gen1():
   global graph
   global last_pos
   with graph.as_default():
       capture = cv2.VideoCapture("My Video.mp4")
       while capture.isOpened():
        grabbed,frame = capture.read()
        if grabbed:
            frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
            video_cur_pos = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if video_cur_pos - last_pos > sec_to_wait : #or last_pos==0
                last_pos = video_cur_pos
                for ind, seq in enumerate(observ_points):
                   points = np.array(seq['points'])
                   rect = bounding_rects[ind]
                   croped_frame = frame[rect[1]:(rect[1] + rect[3]),
                                  rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
                   croped_frame = cv2.cvtColor(croped_frame, cv2.COLOR_BGR2RGB)
                   imgname = 'croped{}_{}.jpg'.format(1,ind)
                   cv2.imwrite(imgname, croped_frame)

            cv2.drawContours(frame, allpoints, contourIdx=-1,
                           color=(0, 255, 0), thickness=2, lineType=cv2.LINE_8)
            frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if jpeg is not None:
               yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
               print("frame is none")
            #time.sleep(1)
        else:
            break

def gen2():
   global graph
   with graph.as_default():
       capture = cv2.VideoCapture("My Video2.mp4")
       while capture.isOpened():
        grabbed,frame = capture.read()
        if grabbed:
           scale_percent = 45  # percent of original size
           width = int(frame.shape[1] * scale_percent / 100)
           height = int(frame.shape[0] * scale_percent / 100)
           dim = (width, height)
           # resize image
           frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
           if torch.cuda.is_available():
             img = transform(Image.fromarray(frame).convert('RGB')).cuda()
           else:
               img = transform(Image.fromarray(frame).convert('RGB'))
           output = model(img.unsqueeze(0))
           imess = str(datetime.datetime.now())
           x = int(output.detach().cpu().sum().numpy())+10
           timess = datetime.datetime.now()
           with app.app_context():
                c = get_db().cursor()
                c.execute("INSERT INTO " + table_name + " VALUES (?,?,?)", (x, timess,2))
                get_db().commit()

           ret, jpeg = cv2.imencode('.jpg', frame)
           if jpeg is not None:
               yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
           else:
               print("frame is none")
           time.sleep(1)
        else:
            break


@app.route('/video_feed1')
def video_feed1():
    return Response(gen1(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    with app.app_context():
        if torch.cuda.is_available():
          checkpoint = torch.load('PartAmodel_best.pth.tar')
        else:
            checkpoint = torch.load('PartAmodel_best.pth.tar',map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        log = logging.getLogger('werkzeug')
        source = "My Video.mp4"
        #video_getter = VideoGet(source).start()
        log.setLevel(logging.ERROR)
        c = get_db().cursor()
        sql = 'create table if not exists ' + table_name + ' (peoplecnt integer , logtime text , camno integer , locid integer)'
        c.execute(sql)
        get_db().commit()
        c.close()
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        print("Starting server on http://localhost:2000")
        print("Serving ...",  app.run(host='0.0.0.0', port=2000))
        print("Finished !")
        print("Done !")