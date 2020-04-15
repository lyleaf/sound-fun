from flask import Flask, request, render_template, url_for
import base64
from voice_transfer import VoiceTransfer
from google.cloud import storage

app = Flask(__name__)


@app.route("/start")
def start():
    app.logger.debug('hello start')
    return render_template('start.html')

@app.route("/")
@app.route("/intro")
def intro():
    return render_template('intro.html')

@app.route("/home")
def home():
    app.logger.debug('hello home')
    return render_template('home.html')

@app.route("/exit")
def exit():
    app.logger.debug('hello exit')
    return render_template('exit.html')

@app.route("/processed")
def processed():
    return render_template('processed.html')

@app.route("/processing")
def processing():
    return render_template('processing.html')

@app.route("/about")
def about():
    return "<h1>About Page</h1>"

@app.route("/process", methods=["GET","POST"])
def process():
    if request.method == 'POST':
        url = str(request.data)
        fn = url.split('=')[-1][:-1] + '.webm'
        print('process function')
        print(fn)
        vt = VoiceTransfer('./sound/%s' % fn)
        vt.extract_feature()
        vt.load_model('Flute')
        output_file, output_mix_file, original_file = vt.transfer(0, 0.4, -5)
        print(output_file, output_mix_file, original_file)
        
        storage_client = storage.Client()
        bucket = storage_client.bucket("sound-of-india")
        for file in [output_file, output_mix_file, original_file]:
            blob_name = file.split('/')[-1]
            print('blob name is %s' % blob_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file)
            blob.make_public()
            print(blob.media_link)
        return "Proccessed"

@app.route("/savesound", methods=["GET","POST"])
def savesound():
    if request.method == 'POST':
        file = request.files['audio_data']
        fn = request.files['audio_data'].filename
        file.save('./sound/%s' % fn) 
        return "saved!"

if __name__ == '__main__':
    app.run(debug=True, port="5443")