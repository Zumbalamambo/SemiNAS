from scipy.io.wavfile import read
import numpy as np
import audio
import soundfile as sf

def _process_utterance(wav_path, hparams):
    try:
        # Load the audio as numpy array
        sr, wav = read(wav_path)
        wav = wav.astype(np.float32)
        MAX_WAV_VALUE = 32768.0
        wav = wav / MAX_WAV_VALUE
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
        wav_path))
        return None
    except Exception as e:
        wav, sr = sf.read(wav_path)

    D = audio._stft(wav, hparams)

    mel_spectrogram = audio._linear_to_mel(np.abs(D), hparams)

    if hparams.vocoder == 'waveglow':
        mel_spectrogram = audio.dynamic_range_compression(mel_spectrogram)
    else:
        mel_spectrogram = audio.amp_to_db(mel_spectrogram)
        mel_spectrogram = audio.normalize(mel_spectrogram, hparams)
        mel_spectrogram = (mel_spectrogram * 8.) - 4.
    mel_spectrogram = mel_spectrogram.astype(np.float32)

    mel_frames = mel_spectrogram.shape[1]

    constant_values = 0.
    if hparams.use_lws:
        # Ensure time resolution adjustement between audio and mel-spectrogram
        fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
        l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

        # Zero pad audio signal
        out = np.pad(wav, (l, r), mode='constant', constant_values=constant_values)
    else:
        # Ensure time resolution adjustement between audio and mel-spectrogram
        l_pad, r_pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams), 1)

        # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=constant_values)

    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0

    return out, mel_spectrogram
