import crepe
import ddsp
import ddsp.training
import os
import gin
import tensorflow.compat.v2 as tf
import librosa
import numpy as np
import time

# input: wav file
# output: wav file

# vt = VoiceTransfer(file)
# vt.transfer()

class VoiceTransfer(object):
    """docstring for VoiceTransfer"""
    def __init__(self, input_file):
        super(VoiceTransfer, self).__init__()
        self.input_file = input_file

    def shift_ld(self, audio_features, ld_shift=0.0):
      """Shift loudness by a number of ocatves."""
      audio_features['loudness_db'] += ld_shift
      return audio_features


    def shift_f0(self, audio_features, f0_octave_shift=0.0):
      """Shift f0 by a number of ocatves."""
      audio_features['f0_hz'] *= 2.0 ** (f0_octave_shift)
      audio_features['f0_hz'] = np.clip(audio_features['f0_hz'], 
                                        0.0, 
                                        librosa.midi_to_hz(110.0))
      return audio_features


    def mask_by_confidence(self, audio_features, confidence_level=0.1):
      """For the violin model, the masking causes fast dips in loudness. 
      This quick transient is interpreted by the model as the "plunk" sound.
      """
      mask_idx = audio_features['f0_confidence'] < confidence_level
      audio_features['f0_hz'][mask_idx] = 0.0
      # audio_features['loudness_db'][mask_idx] = -ddsp.spectral_ops.LD_RANGE
      return audio_features


    def smooth_loudness(self, audio_features, filter_size=3):
      """Smooth loudness with a box filter."""
      smoothing_filter = np.ones([filter_size]) / float(filter_size)
      audio_features['loudness_db'] = np.convolve(audio_features['loudness_db'], 
                                               smoothing_filter, 
                                               mode='same')
      return audio_features

    def extract_feature(self):
        audio, self.sr = librosa.load(self.input_file)
        self.audio = audio[np.newaxis, :]

        # Compute features.
        start_time = time.time()
        self.audio_features = ddsp.training.eval_util.compute_audio_features(self.audio)
        self.audio_features['loudness_db'] = self.audio_features['loudness_db'].astype(np.float32)
        self.audio_features_mod = None
        print('Audio features took %.1f seconds' % (time.time() - start_time))

    def load_model(self, model):
        #model = 'Flute2' #@param ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone','Upload your own (checkpoint folder as .zip)']
        MODEL = model
        self.model_name = model
        if model in ('Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone'):
          # Pretrained models.
          PRETRAINED_DIR = 'pretrained'
          model_dir = PRETRAINED_DIR
          gin_file = os.path.join(PRETRAINED_DIR, 'operative_config-0.gin')


        # Parse gin config,
        with gin.unlock_config():
          gin.parse_config_file(gin_file, skip_unknown=True)

        # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
        ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
        ckpt_name = ckpt_files[0].split('.')[0]
        ckpt = os.path.join(model_dir, ckpt_name)

        # Ensure dimensions and sampling rates are equal
        time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
        n_samples_train = gin.query_parameter('Additive.n_samples')
        hop_size = int(n_samples_train / time_steps_train)
        print(self.audio.shape[1])
        time_steps = int(self.audio.shape[1] / hop_size)
        print(time_steps)
        n_samples = time_steps * hop_size
        print(n_samples)
        gin_params = [
            'Additive.n_samples = {}'.format(n_samples),
            'FilteredNoise.n_samples = {}'.format(n_samples),
            'DefaultPreprocessor.time_steps = {}'.format(time_steps),
        ]

        with gin.unlock_config():
          gin.parse_config(gin_params)


        # Trim all input vectors to correct lengths 
        for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
          print(type(self.audio_features[key]))
          self.audio_features[key] = self.audio_features[key][:time_steps]

        print(self.audio_features['audio'].shape)
        print(n_samples)
        self.audio_features['audio'] = self.audio_features['audio'][:, :n_samples]


        # Set up the model just to predict audio given new conditioning
        self.model = ddsp.training.models.Autoencoder()
        self.model.restore(ckpt)

        # Build model by running a batch through it.
        start_time = time.time()
        _ = self.model(self.audio_features, training=False)
        print('Restoring model took %.1f seconds' % (time.time() - start_time))

    def transfer(self, f0_octave_shift=0, f0_confidence_threshold=0, loudness_db_shift=0):      
        auto_adjust = True #@param{type:"boolean"}

        #@markdown You can also make additional manual adjustments:
        #@markdown * Shift the fundmental frequency to a more natural register.
        #@markdown * Silence audio below a threshold on f0_confidence.
        #@markdown * Adjsut the overall loudness level.
        # f0_octave_shift =  0 #@param {type:"slider", min:-2, max:2, step:1}
        # f0_confidence_threshold =  0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
        # loudness_db_shift = 0 #@param {type:"slider", min:-20, max:20, step:1}

        #@markdown You might get more realistic sounds by shifting a few dB, or try going extreme and see what weird sounds you can make...

        self.audio_features_mod = {k: v.copy() for k, v in self.audio_features.items()}
        MODEL = self.model_name


        if auto_adjust:    
          if MODEL in ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Saxophone', 'Tenor_Saxophone']:
            print('\nEnabling auto-adjust.')
            # Adjust the peak loudness.
            l = self.audio_features['loudness_db']
            model_ld_avg_max = {
                'Violin': -34.0,
                'Flute': -45.0,
                'Flute2': -44.0,
                'Trumpet': -52.3,
                'Tenor_Saxophone': -31.2
            }[MODEL]
            ld_max = np.max(self.audio_features['loudness_db'])
            ld_diff_max = model_ld_avg_max - ld_max
            self.audio_features_mod = self.shift_ld(self.audio_features_mod, ld_diff_max)

            # Further adjust the average loudness above a threshold.
            l = self.audio_features_mod['loudness_db']
            model_ld_mean = {
                'Violin': -44.0,
                'Flute': -51.0,
                'Flute2': -53.0,
                'Trumpet': -69.2,
                'Tenor_Saxophone': -50.8
            }[MODEL]
            ld_thresh = -70.0
            ld_mean = np.mean(l[l > ld_thresh])
            ld_diff_mean = model_ld_mean - ld_mean
            self.audio_features_mod = self.shift_ld(self.audio_features_mod, ld_diff_mean)

            # Shift the pitch register.
            model_p_mean = {
                'Violin': 73.0,
                'Flute': 81.0,
                'Flute2': 74.0,
                'Trumpet': 65.8,
                'Tenor_Saxophone': 57.8
            }[MODEL]
            p = librosa.hz_to_midi(self.audio_features['f0_hz'])
            p[p == -np.inf] = 0.0
            p_mean = p[l > ld_thresh].mean()
            p_diff = model_p_mean - p_mean
            p_diff_octave = p_diff / 12.0
            round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
            p_diff_octave = round_fn(p_diff_octave)
            self.audio_features_mod = self.shift_f0(self.audio_features_mod, p_diff_octave)

          else:
            print('\nUser uploaded model: disabling auto-adjust.')


          
        self.audio_features_mod = self.shift_ld(self.audio_features_mod, loudness_db_shift)
        self.audio_features_mod = self.shift_f0(self.audio_features_mod, f0_octave_shift)
        self.audio_features_mod = self.mask_by_confidence(self.audio_features_mod, f0_confidence_threshold)



        # Resynthesize audio

        af = self.audio_features if self.audio_features_mod is None else self.audio_features_mod

        # Run a batch of predictions.
        start_time = time.time()
        audio_gen = self.model(af, training=False)
        print('Prediction took %.1f seconds' % (time.time() - start_time))


        print(self.input_file)
        sound_id = self.input_file.split('.')[1].split('/')[-1]
        audio_gen = audio_gen.numpy().flatten()
        self.output_file = './sound/%s_%s_%d_%1.2f_%d_voice_transfer.wav' % (
          sound_id,
          self.model_name,
          f0_octave_shift,
          f0_confidence_threshold,
          loudness_db_shift)
        librosa.output.write_wav(self.output_file, audio_gen, self.sr, True)

        audio_np = self.audio.flatten()[:len(audio_gen)]
        self.output_mixed_file = './sound/%s_%s_%d_%1.2f_%d_voice_transfer_mixed.wav' % (
          sound_id,
          self.model_name,
          f0_octave_shift,
          f0_confidence_threshold,
          loudness_db_shift)
        #audio_mixed = np.vstack((audio_gen_np,audio_np))
        librosa.output.write_wav(self.output_mixed_file, 
          np.asfortranarray(np.vstack((audio_gen,audio_np))), self.sr, True)

        self.original_file = './sound/%s.wav' % sound_id
        librosa.output.write_wav(self.original_file, audio_np, self.sr, True)
        print('Saved to %s, %s, %s' % (self.output_file, self.output_mixed_file, self.original_file))

        return self.output_file, self.output_mixed_file, self.original_file

if __name__ == '__main__':
    vt = VoiceTransfer('./sound/Adi.wav')
    vt.extract_feature()
    vt.load_model('Flute2')
    for f0_octave_shift in np.arange(0, 1, 1):
      for f0_confidence_threshold in np.arange(0.2, 0.5, 0.2):
        for loudness_db_shift in np.arange(-20, 21, 5):
          output_file = vt.transfer(f0_octave_shift, f0_confidence_threshold, loudness_db_shift)