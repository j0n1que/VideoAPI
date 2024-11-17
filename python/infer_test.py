import ctypes
import cv2 as cv
import numpy as np
from ctypes import CDLL, POINTER, Structure, cast, c_int, c_float, c_char_p, c_void_p
import tempfile
import sys
from moviepy.editor import VideoFileClip, AudioClip

class BBox(Structure):
    _fields_ = [('x', c_float), ('y', c_float), ('width', c_float), ('height', c_float)]


class InferResults(Structure):
    _fields_ = [('box', BBox), ('score', c_float)]


class InferContext:
    class NativeContext(Structure):
        pass

    c_int_p = POINTER(c_int)
    c_float_p = POINTER(c_float)
    c_ctx_p = POINTER(NativeContext)
    c_results_p = POINTER(InferResults)
    c_results_pp = POINTER(c_results_p)

    def __init__(self, lib: CDLL):
        lib.eva_init.restype = self.c_ctx_p
        lib.eva_infer.argtypes = [self.c_ctx_p, c_int, c_int, c_char_p, c_int, c_void_p]
        lib.eva_get_results.argtypes = [self.c_ctx_p, self.c_results_pp, self.c_int_p]

        self.lib = lib
        self.ctx = self.lib.eva_init()

        if not self.ctx:
            raise Exception('Could not init model inference.')

    def __del__(self):
        self.lib.eva_free(self.ctx)

    def process_image(self, image: np.ndarray):
        if image.shape[2] != 3 or image.dtype != 'uint8':
            raise Exception('Invalid image shape.')

        h, w, ch = image.shape

        if self.lib.eva_infer(self.ctx, c_int(w), c_int(h), c_char_p('bgr'.encode()), c_int(3),
                              cast(image.flatten().tobytes(), c_void_p)) > 0:
            results = self.c_results_p()
            count = c_int()

            self.lib.eva_get_results(self.ctx, self.c_results_pp(results), self.c_int_p(count))

            font = cv.FONT_HERSHEY_PLAIN
            font_scale = 1.3
            title_thickness = 1

            for i in range(count.value):
                box = results[i].box
                conf_title = f'{results[i].score:.2f}'

                title_size = cv.getTextSize(conf_title, font, font_scale, title_thickness)
                cv.rectangle(image, (int(box.x), int(box.y)), (int(box.x + box.width), int(box.y + box.height)),
                             (0, 255, 0), 2)
                cv.rectangle(image, (int(box.x), int(box.y)),
                             (int(box.x) + title_size[0][0], int(box.y - title_size[0][1])), (32, 32, 32), -1)
                cv.putText(image, conf_title, (int(box.x), int(box.y)), font, font_scale,
                           (255, 255, 255), title_thickness)


def write_video_to_tempfile(video_bytes) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    with open(temp_file.name, 'wb') as f:
        f.write(video_bytes)
    return temp_file.name

def configure_video(filePath) -> (cv.VideoWriter, cv.VideoCapture):
    video_bytes = sys.stdin.buffer.read()
    temp_filename = write_video_to_tempfile(video_bytes)
    cap = cv.VideoCapture(temp_filename)
    if not cap.isOpened():
        raise Exception('Could not open capture device')
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    return cv.VideoWriter(filePath + 'video.mp4', fourcc, fps, (width, height)), cap

def add_audio_to_video(video_file,  output_file, duration=None):
    video = VideoFileClip(video_file)
    if duration is None:
        duration = video.duration
    silence = AudioClip(lambda t: 0, duration=duration)
    silence = silence.set_fps(44100)
    video = video.set_audio(silence)
    video.write_videofile(output_file, codec="libx264", audio_codec="aac")

infer = InferContext(CDLL('./libinfer.so'))
filePath = '../../../python/build/'
video, cap = configure_video(filePath)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    infer.process_image(frame)
    video.write(frame)
cap.release()
video.release()
add_audio_to_video(filePath + 'video.mp4', filePath + 'video_with_audio.mp4')
cv.destroyAllWindows()
