# helper functions for the notebook
import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates

RGB_YIQ_MAT = np.array([[0.299, 0.587, 0.114],
                        [0.596, -0.275, -0.321],
                        [0.212, -0.523, 0.311]])


def rgb2yiq(imRGB):
    """
    Converting an RGB image to YIQ
    :param imRGB: RGB image matrix
    :return: YIQ image
    """
    yiq_trans = RGB_YIQ_MAT.T  # Transpose of the matrix
    res = np.dot(imRGB, yiq_trans)
    return res


def yiq2rgb(imYIQ):
    """
    Converting an YIQ image to RGB
    :param imYIQ: YIQ image matrix
    :return: RGB image
    """
    yiq_rgb_mat = np.linalg.inv(RGB_YIQ_MAT)  # Invert transform matrix
    res = np.dot(imYIQ, yiq_rgb_mat.T)
    return res


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]
    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)
    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real
    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(
        np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]
    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # store
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # map to -pi:pi
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
