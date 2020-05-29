let playing = false;
let hasGoodRecording = false;
let playback = true;
let blobUrl; 
let playbackEle;

let mic, recorder, soundFile;
let record_button, upload_button, playback_button;
let vid;
let soundBlob;
let recordTimeLength = 66;
let img;
let col;
let font;

let buttonWidth = 0.25
let buttonHeight = 0.04
let buttonFontSize = '2vh'
let borderRadius = '25px'

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
    hasGoodRecording = false;
    record_button.html('Play Video & Record Voice');
  } else { // when not playing, clicking on button will start video, record sound
    vid.play();
    hasGoodRecording = true;
    upload_button.hide();
    playback_button.hide();
    recorder.record(soundFile, recordTimeLength, recordFinish);
    record_button.html('Click to start over');
    playing = true;
  }
}

function replay() {
  if (playback) {
    playbackEle.loop();
    playback_button.html('Pause')
  }
  else {
    playbackEle.pause();
    playback_button.html('Playback Your Recording')
  }
  playback = !playback;
}

function recordFinish() {
  soundBlob = soundFile.getBlob();
  playing = false;
  if (hasGoodRecording) {
    record_button.html("Play Video & Record Voice");
    upload_button.show(); 
    playback_button.show();
    blobUrl = URL.createObjectURL(soundBlob);
    playbackEle = createAudio(blobUrl);
  }
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
  
  upload_button.html("Uploaded, thank you!")
}

function vidLoad() {
  vid.center();
}

function preload() {
  img = loadImage('static/guy.png');
  imageMode(CENTER);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

function setup() {
  var cnv = createCanvas(windowWidth, windowHeight);
  cnv.style('background-color', color('#FCB017'));
  cnv.style('z-index', -1);
   

  col = color('white');



  vid = createVideo(
    ['static/janadonate.mp4'],
    vidLoad
  );

  vid.size(windowWidth * 0.4, AUTO);


  mic = new p5.AudioIn();
  mic.start();
  recorder = new p5.SoundRecorder();
  recorder.setInput(mic);
  soundFile = new p5.SoundFile();

  record_button = createButton('Play Video & Record Voice');
  record_button.mousePressed(playVidAndRecord); 
  record_button.style('background-color', col);

  playback_button = createButton('Playback Your Recording');
  playback_button.mousePressed(replay); 
  playback_button.hide();

  upload_button = createButton(`I'm Happy! Submit`);
  upload_button.mousePressed(upload); 
  upload_button.hide();
  noLoop();
}

function draw() {
  background('#FCB017');

  vid.center();

  record_button.position(windowWidth * 0.5 - windowWidth * buttonWidth / 2.0, windowHeight * 0.81);
  record_button.style('font-size', buttonFontSize);
  record_button.style('border', 0);
  record_button.style('border-radius', borderRadius);
  record_button.size(windowWidth * buttonWidth, windowHeight * buttonHeight);

  playback_button.position(windowWidth * 0.5 - windowWidth * buttonWidth / 2.0, windowHeight * 0.86);
  playback_button.style('background-color', col);
  playback_button.style('font-size', buttonFontSize);
  playback_button.style('border', 0);
  playback_button.style('border-radius', borderRadius);
  playback_button.size(windowWidth * buttonWidth, windowHeight * buttonHeight);

  upload_button.position(windowWidth * 0.5 - windowWidth * buttonWidth / 2.0, windowHeight * 0.91);
  upload_button.style('background-color', col);
  upload_button.style('font-size', buttonFontSize);
  upload_button.style('border', 0);
  upload_button.style('border-radius', borderRadius);
  upload_button.size(windowWidth * buttonWidth, windowHeight * buttonHeight);


  image(img, windowWidth * 0.48, windowHeight * 0.02, windowHeight * 0.1, windowHeight * 0.13);
  textSize(windowHeight * 0.02);
  textAlign(CENTER);
  font = loadFont('static/font.ttf');
  textFont(font);
  text(`Sing along with the video below. Please sing the entire anthem at one go.`, 
    windowWidth * 0.5 - windowWidth * 0.4 / 2, windowHeight * 0.2,
    windowWidth * 0.4, windowHeight * 0.3);

  textSize(windowHeight * 0.02);
  textAlign(CENTER);
  text(`Recording will automatically start when the video plays and 
    stop  when the video finishes (~60s).`, 
    windowWidth * 0.5 - windowWidth * 0.3 / 2, windowHeight * 0.715,
    windowWidth * 0.3, windowHeight * 0.3);

}
