let playing = false;

let mic, recorder, soundFile;
let record_button, upload_button;
let vid;
let soundBlob;

function create_UUID(){
    var dt = new Date().getTime();
    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = (dt + Math.random()*16)%16 | 0;
        dt = Math.floor(dt/16);
        return (c=='x' ? r :(r&0x3|0x8)).toString(16);
    });
    return uuid;
}

function playVidAndRecord()  {
  getAudioContext().resume()
  if (playing) { // when playing, clicking on button will stop video, return to beginning, discard soundFile
    vid.stop();
    playing = false;
    recorder.stop();
    record_button.html('Play & Record');
  } else { // when not playing, clicking on button will start video, record sound
    vid.play();
    recorder.record(soundFile, 62);
    record_button.html('recording now! click to restart');
    playing = true;
  }
}

function replay() {
  soundFile.play();
}

function upload() {
  
  // soundBlob = soundFile.getBlob();
  // let serverUrl = '/upload_to_gcs';
  let fileName = create_UUID();
  saveSound(soundFile, '%s.wav' % fileName);
  // console.log(fileName);
  // let httpRequestOptions = {
  //   method: 'POST',
  //   body: new FormData().append('audio_data', soundBlob, fileName),
  //   headers: new Headers({
  //     'Content-Type': 'multipart/form-data'
  //   })
  // };
  // httpDo(serverUrl, httpRequestOptions);
}

function preload() {
  vid = createVideo(['static/background_video.webm']);
}

function preload() {
  vid = createVideo(['static/background_video.webm']);
}

function setup() {
  createCanvas(0, 0);
  mic = new p5.AudioIn();
  mic.start();
  recorder = new p5.SoundRecorder();
  recorder.setInput(mic);
  soundFile = new p5.SoundFile();
  record_button = createButton('record');
  record_button.mousePressed(playVidAndRecord); 
  upload_button = createButton('upload');
  upload_button.mousePressed(upload); 
}
