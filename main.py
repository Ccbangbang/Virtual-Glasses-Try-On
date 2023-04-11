import os, sys, tempfile, shutil, contextlib
import flask
import urllib
import urllib.parse
import urllib.request
import hashlib
import collections
import mimetypes
import cv2
import re
import lxml
from PIL import  Image
from lxml.html import HTMLParser, document_fromstring


app = flask.Flask(__name__)

@app.route('/')
def root_page():
    return flask.render_template('root.html')

@app.route('/view/')
def view_page():
    url = flask.request.args.get("url")
    o = urllib.parse.urlparse(url)
    if o.scheme == '':
        print("Invalid url: Scheme error")
        exit(2)
    if o.netloc == '':
        print("Invalid url: Scheme error")
        exit(2)

    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'PurdueUniversityClassProject/1.0 (wang1927@purdue.edu https://goo.gl/dk8u5S)')

    with urllib.request.urlopen(req) as response:
        htm = response.read()
        html = htm.decode("UTF-8")
    # Credit: Adapted from example in Python 3.4 Documentation, urllib.request
    #         License: PSFL https://www.python.org/download/releases/3.4.1/license/
    #                  https://docs.python.org/3.4/library/urllib.request.html


    #root = lxml.html.parse(html)
    #print(root)
    parser = HTMLParser(encoding="UTF-8")
    root = document_fromstring(html, parser=parser, base_url=url)
    #print(root)
    #root.make_links_absolute(url, resolve_base_href=True)
    #print(html)
    for node in root.iter():
        if node.tag == 'head':
            newstr = "<base href = {}>".format(url)
            html = html.replace('<head>','<head>' + '\n' + newstr)
        if node.tag == 'HEAD':
            newstr = "<BASE HREF = {}>".format(url)
            html = html.replace('<TITLE>','<TITLE>' + '\n' + newstr)
    path = copy_profile_photo_to_static(root)
    static_url = flask.url_for('static',filename = os.path.basename(path), _external = True)
    #print(static_url)

    expr = r"(/home/ecegridfs/a/ee364a13/hpo/static/)([\w]+)(\.[\w]+)"
    o = re.match(expr,path)
    photo  = o.group(2)
    for node in root.iter():
        if node.tag == "img":
            url = node.get("src")
            with urllib.request.urlopen(url) as response:
                type = response.info().get('Content-Type')
                extension = mimetypes.guess_extension(type)
                filename = make_filename(url,extension)
                col = filename.split(".")
                name = col[0]
                match_name = name

                if  match_name == photo:
                    #print("Found")
                    #print(node.attrib['src'])
                    html = html.replace(node.attrib['src'],static_url)


    return html

def make_filename(url,extension):
    filename = hashlib.sha1(url.encode('UTF-8')).hexdigest()+extension
    return filename

@contextlib.contextmanager
def pushd_temp_dir(base_dir=None, prefix="tmp.hpo."):
    '''
    Create a temporary directory starting with {prefix} within {base_dir}
    and cd to it.
    This is a context manager.  That means it can---and must---be called using
    the with statement like this:
        with pushd_temp_dir():
            ....   # We are now in the temp directory
        # Back to original directory.  Temp directory has been deleted.
    After the with statement, the temp directory and its contents are deleted.
    Putting the @contextlib.contextmanager decorator just above a function
    makes it a context manager.  It must be a generator function with one yield.
    - base_dir --- the new temp directory will be created inside {base_dir}.
                   This defaults to {main_dir}/data ... where {main_dir} is
                   the directory containing whatever .py file started the
                   application (e.g., main.py).
    - prefix ----- prefix for the temp directory name.  In case something
                   happens that prevents
    '''
    if base_dir is None:
        proj_dir = sys.path[0]
        # e.g., "/home/ecegridfs/a/ee364z15/hpo"

        main_dir = os.path.join(proj_dir, "data")
        # e.g., "/home/ecegridfs/a/ee364z15/hpo/data"

    # Create temp directory
    temp_dir_path = tempfile.mkdtemp(prefix=prefix, dir=base_dir)

    try:
        start_dir = os.getcwd()  # get current working directory
        os.chdir(temp_dir_path)  # change to the new temp directory

        try:
            yield
        finally:
            # No matter what, change back to where you started.
            os.chdir(start_dir)
    finally:
        # No matter what, remove temp dir and contents.
        shutil.rmtree(temp_dir_path, ignore_errors=True)




@contextlib.contextmanager
def fetch_images(etree):
    etree.make_links_absolute()
    with pushd_temp_dir():
        filename_to_node = collections.OrderedDict()
        for node in etree.iter():
            if node.tag == "img":
                url = node.get("src")
                #req = urllib.request.Request(url)
                with urllib.request.urlopen(url) as response:
                    type = response.info().get('Content-Type')
                    extension = mimetypes.guess_extension(type)
                    filename = make_filename(url,extension)
                    filename_to_node[filename] = node
                    with open(filename,"wb") as file:
                        file.write(response.read())
        yield filename_to_node

def face_size(face):
    size = face[2]*face[3]
    return size

def get_image_info(filename):
    FACE_DATA_PATH = '/home/ecegridfs/a/ee364/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(FACE_DATA_PATH)
    image_info = dict()
    faces = list()
    img = cv2.imread(filename)   #Use the function cv2.imread() to read an image. The image should be in the working directory or a full path of image should be given.
    height, width = img.shape[:2]  #Shape of image is accessed by img.shape. It returns a tuple of number of rows, columns and channels (if image is color):
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img_grayscale, 1.3, 5)
    face = sorted(face, key=face_size, reverse=True)
    for f in face:
        ff = dict()
        ff = dict()
        ff["w"] = f[2]
        ff["h"] = f[3]
        ff["x"] = f[0]
        ff["y"] = f[1]
        faces.append(ff)
    image_info["w"] = width
    image_info["h"] = height
    image_info["faces"] = faces
    return image_info

def find_profile_photo_filename(filename_to_etree):
    max_size = 0
    for filename in filename_to_etree:   #loop all the keys from dict
        col = filename.split(".")
        ext = col[1]
        if ext == 'jpg' or ext == 'jpeg' or ext == 'jpe':
            image_info = get_image_info(filename)
            if image_info["faces"] != None:           #Image should has face
                if len(image_info["faces"]) == 1:    #Image only has one face
                    current_size = image_info["faces"][0]["h"]*image_info["faces"][0]["w"]
                    if current_size > max_size:        #Choose the biggest face
                        max_size = current_size
                        photo = filename
    return photo

def copy_profile_photo_to_static(etree):
    with fetch_images(etree) as filename_to_node:
        photo = find_profile_photo_filename(filename_to_node)
        image_info = get_image_info(photo)
        #print(image_info)
        face_info = image_info["faces"]
        final_name = add_glasses(photo,face_info)
        im = Image.open(final_name)

    proj_dir = sys.path[0]
    static_dir = os.path.join(proj_dir,"static")
    path = os.path.join(static_dir,final_name)
    im.save(path)

    return path

def add_glasses(filename, face_info):
    EYE_DATA = "/home/ecegridfs/a/ee364/site-packages/cv2/data/haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(EYE_DATA)
    w = face_info[0].get("w")
    h = face_info[0].get("h")
    x = face_info[0].get("x")
    y = face_info[0].get("y")
    #print(face_info)
    face_img = cv2.imread(filename)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    eye_coordinate = list()
    for (ex,ey,ew,eh) in eyes:
        #eye_coordinate.append(ex)
        #eye_coordinate.append(ey)
        #eye_coordinate.append(ew)
        #eye_coordinate.append(eh)
        eye_coordinate.append(((x+int(ex+0.5*ew)),(y+int(ey+0.5*eh)),ew,eh))
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #print(eye_coordinate)
    #print(type(eye_coordinate))
    if len(eye_coordinate) == 0:
        print("No eyes detecded")
        tuple1 = (int(x+2*w/8),int(y+2*h/8),int(w/4),int(w/4))
        cv2.rectangle(face_img,(tuple1[0],tuple1[1]),(tuple1[0]+tuple1[2],tuple1[1]+tuple1[3]),(0,255,0),2)
        tuple2 = (int(x+5*w/8),int(y+2*h/8),int(w/4),int(w/4))
        cv2.rectangle(face_img,(tuple2[0],tuple2[1]),(tuple2[0]+tuple2[2],tuple2[1]+tuple2[3]),(0,255,0),2)
        eye_coordinate.append(tuple1)
        eye_coordinate.append(tuple2)
        #print(eye_coordinate)

    if len(eye_coordinate) == 1:
        print("Only one eye detecded")
        new_list = tuple()
        if eye_coordinate[0][0] >= int(x+w/2):
            new_list = (eye_coordinate[0][0]-25,eye_coordinate[0][1],eye_coordinate[0][2],eye_coordinate[0][2])
            cv2.rectangle(roi_color,(ex-20,ey),(ex-25+ew,ey+eh),(0,255,0),2)
        if eye_coordinate[0][0] <= int(x+w/2):
            new_list = (eye_coordinate[0][0]+25,eye_coordinate[0][1],eye_coordinate[0][2],eye_coordinate[0][2])
            cv2.rectangle(roi_color,(ex+20,ey),(ex+25+ew,ey+eh),(0,255,0),2)
        eye_coordinate.append(new_list)
        #print(eye_coordinate)


    #print(eye_coordinate[0][0])
    if eye_coordinate[0][0] >= int(x+w/2):
        print("Right eye")
        cv2.line(face_img,(eye_coordinate[0][0]-int(0.5*eye_coordinate[0][2]),eye_coordinate[0][1]),(eye_coordinate[1][0]+int(0.5*eye_coordinate[1][2]),eye_coordinate[1][1]),(255,0,0),2)
        cv2.line(face_img,(eye_coordinate[0][0]+int(0.5*eye_coordinate[0][2]),eye_coordinate[0][1]),(eye_coordinate[0][0]+int(0.5*eye_coordinate[0][2]+10),eye_coordinate[0][1]),(0,0,255),2)
        cv2.line(face_img,(eye_coordinate[1][0]-int(0.5*eye_coordinate[1][2]),eye_coordinate[1][1]),(eye_coordinate[1][0]-int(0.5*eye_coordinate[1][2]+10),eye_coordinate[1][1]),(0,0,255),2)
    if eye_coordinate[0][0] <= int(x+w/2):
        print("Left eye")
        cv2.line(face_img,(eye_coordinate[0][0]+int(0.5*eye_coordinate[0][2]),eye_coordinate[0][1]),(eye_coordinate[1][0]-int(0.5*eye_coordinate[1][2]),eye_coordinate[1][1]),(255,0,0),2)
        cv2.line(face_img,(eye_coordinate[0][0]-int(0.5*eye_coordinate[0][2]),eye_coordinate[0][1]),(eye_coordinate[0][0]-int(0.5*eye_coordinate[0][2]+10),eye_coordinate[0][1]),(0,0,255),2)
        cv2.line(face_img,(eye_coordinate[1][0]+int(0.5*eye_coordinate[1][2]),eye_coordinate[1][1]),(eye_coordinate[1][0]+int(0.5*eye_coordinate[1][2]+10),eye_coordinate[1][1]),(0,0,255),2)
    col = filename.split(".")
    name = col[0]
    final_name = name+".jpg"
    cv2.imwrite(final_name,face_img)


    #cv2.imshow('image',face_img)
    ##cv2.waitKey()
    #cv2.destroyAllWindows()
    return final_name

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=os.environ.get("ECE364_HTTP_PORT", 10013), use_reloader = True, use_evalex = False, debug = True, use_debugger = False)
    #value = view_page()
    #print(value)