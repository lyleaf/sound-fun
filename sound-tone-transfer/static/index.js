function create_UUID(){
    var dt = new Date().getTime();
    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = (dt + Math.random()*16)%16 | 0;
        dt = Math.floor(dt/16);
        return (c=='x' ? r :(r&0x3|0x8)).toString(16);
    });
    return uuid;
}

const recordAudio = () =>
  new Promise(async resolve => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks = [];

    mediaRecorder.addEventListener("dataavailable", event => {
      audioChunks.push(event.data);
    });

    const start = () => mediaRecorder.start();

    const stop = () =>
      new Promise(resolve => {
        mediaRecorder.addEventListener("stop", () => {
          const audioBlob = new Blob(audioChunks);
          const audioUrl = URL.createObjectURL(audioBlob);
          console.log(audioUrl)
          const audio = new Audio(audioUrl);
          const play = () => audio.play();
          resolve({ audioBlob, audioUrl, play });
        });

        mediaRecorder.stop();
      });

    resolve({ start, stop });
  });

const sleep = time => new Promise(resolve => setTimeout(resolve, time));




const recordButtonAction = async () => {
  // Hit record button will record voice to a blob, and jump to the next page.
  document.getElementById("action").innerHTML = "Listening..."
  const recorder = await recordAudio();
  const actionButton = document.getElementById('action');
  actionButton.disabled = true;
  recorder.start();
  await sleep(10000);
  const audio = await recorder.stop();
  document.getElementById("action").innerHTML = "Playing back..."
  audio.play();
  await sleep(10000);
  var blob = audio.audioBlob;

  var oReq = new XMLHttpRequest();
  oReq.open("POST", '/savesound', true);

  var fd = new FormData();
  const fileName = create_UUID();
  oReq.onreadystatechange = function (oEvent) {
    console.log(`processing?filename=${fileName}`)
    window.location.href = `processing?filename=${fileName}`;
  };
  fd.append("audio_data", blob, fileName+'.webm')
  oReq.send(fd);
}

const processPageLoad = async () => {
  var oReq = new XMLHttpRequest();
  var link = window.location.href;
  console.log('process page load')
  console.log(link)
  console.log(`processed${window.location.search}`)
  oReq.open("POST", '/process', true);
  oReq.onreadystatechange = function (oEvent) {
    window.location.href = `processed${window.location.search}`;
  };
  oReq.send(link);
  // actionButton.disabled = false;
}