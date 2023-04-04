from flask import Flask, render_template, Response, request,redirect,url_for, make_response
from necklace_camera import NecklaceVideoCamera
from tshirt_camera import TshirtVideoCamera
from makeup_camera import MakeupVideoCamera
from flask_cors import CORS,cross_origin

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
_red = 133
_green = 21
_blue = 21
_jewellery_name = ''
_tshirt_name = ''

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route("/post_rgb", methods=["GET","POST"], strict_slashes=False)
@cross_origin()
def post_rgb():
    global _red,_green,_blue
    if request.method == "POST":
        red = int(request.json.get('r'))
        green = int(request.json.get('g'))
        blue = int(request.json.get('b'))

        if red == -1 and blue == -1 and green == -1:
            pass
        if red and blue and green:
            _red = red
            _blue = blue
            _green = green
    return render_template(url_for('index'))

@app.route("/post_jewellery", methods=["GET","POST"], strict_slashes=False)
@cross_origin()
def post_jewellery():
    global _jewellery_name
    if request.method == "POST":
        jewellery_name = request.json.get('r')
        print(jewellery_name)
        _jewellery_name = jewellery_name
        
        
    return render_template(url_for('index'))

@app.route("/post_tshirt", methods=["GET","POST"], strict_slashes=False)
@cross_origin()
def post_tshirt():
    global _tshirt_name
    if request.method == "POST":
        tshirt_name = request.json.get('r')
        print(_tshirt_name)
        _tshirt_name = tshirt_name
        
        
    return render_template(url_for('index'))

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
def makeup_gen(camera):
    while True:
        frame = camera.get_frame(_red,_blue,_green,0.5)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
def jewellery_gen(camera):
    while True:
        frame = camera.get_frame(_jewellery_name)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
def tshirt_gen(camera):
    while True:
        frame = camera.get_frame(_tshirt_name)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed_makeup')
@cross_origin()
def video_feed_makeup():
    return Response(makeup_gen(MakeupVideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_necklace')
@cross_origin()
def video_feed_necklace():
    return Response(jewellery_gen(NecklaceVideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_tshirt')
@cross_origin()
def video_feed_tshirt():
    return Response(tshirt_gen(TshirtVideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080, debug=True)
