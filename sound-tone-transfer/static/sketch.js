let playing = false;

let mic, recorder, soundFile;
let record_button, upload_button;
let vid;
let soundBlob;
let recordTimeLength = 62;


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
    recorder.record(soundFile, recordTimeLength, recordFinish);
    record_button.html('recording now! click to restart');
    playing = true;
  }
}

function replay() {
  let blobUrl = URL.createObjectURL(soundBlob);
  ele = createAudio(blobUrl);
  ele.autoplay(true);
}

function recordFinish() {
  soundBlob = soundFile.getBlob();
  upload_button.show(); 
  playback_button.show();
}

function upload() {
  let fileName = `donate_${create_UUID()}.wav`;
  console.log(fileName);
  saveSound(soundFile, fileName);

  let serverUrl = '/upload_to_gcs';
  soundBlob = soundFile.getBlob();

  var oReq = new XMLHttpRequest();
  oReq.open("POST", serverUrl, true);
  var fd = new FormData();
  fd.append("audio_data", soundBlob, fileName)
  oReq.send(fd);
}

function preload() {
  

  //vid.hide(); 
}

function vidLoad() {
  vid.center();
  vid.volume(0);
}

function setup() {
  var cnv = createCanvas(windowWidth, windowHeight);
  cnv.style('background-color', color('#FCB017'));
  cnv.style('z-index', -1)

  vid = createVideo(
    ['static/background_video.webm'],
    vidLoad
  );

  vid.size(400, 400);


  mic = new p5.AudioIn();
  mic.start();
  recorder = new p5.SoundRecorder();
  recorder.setInput(mic);
  soundFile = new p5.SoundFile();

  let col = color(25, 23, 200, 50);

  record_button = createButton('Play Video And Record');
  record_button.mousePressed(playVidAndRecord); 
  record_button.position(windowWidth * 0.4, windowHeight * 0.7);
  record_button.style('background-color', col);
  record_button.style('font-size', '30px');

  playback_button = createButton('Playback');
  playback_button.mousePressed(replay); 
  playback_button.position(windowWidth * 0.4, windowHeight * 0.8);
  playback_button.style('background-color', col);
  playback_button.style('font-size', '30px');
  playback_button.hide();

  upload_button = createButton('upload');
  upload_button.mousePressed(upload); 
  upload_button.position(windowWidth * 0.4, windowHeight * 0.9);
  upload_button.style('background-color', col);
  upload_button.style('font-size', '30px');
  upload_button.hide();


}