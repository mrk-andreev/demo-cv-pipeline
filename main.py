import time
from multiprocessing import Lock
from multiprocessing import Process
from multiprocessing.managers import SharedMemoryManager

import cv2
import numpy as np


class PipeOut:
    def __init__(self, shm, shape, dtype, lock):
        self._shm = shm
        self._shape = shape
        self._dtype = dtype
        self._lock = lock

    def read(self):
        self._lock.acquire()
        try:
            arr = np.frombuffer(self._shm.buf, dtype=self._dtype)
            return arr.reshape(self._shape)
        finally:
            self._lock.release()


class PipeIn:
    def __init__(self, shm, shape, dtype, lock):
        arr = np.frombuffer(shm.buf, dtype=dtype)
        arr.reshape(shape)
        self._arr = arr
        self._lock = lock

    def write(self, frame):
        self._lock.acquire()
        try:
            self._arr[:] = frame.flatten()
        finally:
            self._lock.release()


def create_pipe(shared_memory_manager: SharedMemoryManager, shape, dtype, itemsize):
    d_size = itemsize * np.prod(shape)
    shm = shared_memory_manager.SharedMemory(size=d_size)

    lock = Lock()
    return PipeIn(shm, shape, dtype, lock), PipeOut(shm, shape, dtype, lock)


def model_loop(pipe_out: PipeOut, shape):
    height, width = shape[:2]
    video_out = cv2.VideoWriter(f'out/model_{int(time.time())}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1.0,
                                (width, height))
    prev_time = time.time()

    try:
        while True:
            while time.time() - prev_time < 1:
                # throttle to 1fps
                continue

            try:
                frame = pipe_out.read()
                video_out.write(frame)
            except Exception as e:
                print(e)
            finally:
                prev_time = time.time()
    finally:
        video_out.release()


def video_loop(shared_memory_manager: SharedMemoryManager):
    cap = cv2.VideoCapture(0)

    video_out = None
    model_process = None
    pipe_in, pipe_out, destructor = None, None, None
    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret and not video_out:
                height, width = frame.shape[:2]
                video_out = cv2.VideoWriter(f'out/main_{int(time.time())}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0,
                                            (width, height))
            if not pipe_in and not pipe_out and not destructor:
                pipe_in, pipe_out = create_pipe(shared_memory_manager, frame.shape, frame.dtype,
                                                frame.itemsize)
                model_process = Process(target=model_loop, args=(pipe_out, frame.shape))
                model_process.start()

            if time.time() - prev_time >= 0.5:  # skip frames
                pipe_in.write(frame)
                prev_time = time.time()

            video_out.write(frame)
    finally:
        cap.release()
        if video_out:
            video_out.release()
        if model_process:
            model_process.terminate()


def main():
    with SharedMemoryManager() as shared_memory_manager:
        video_loop(shared_memory_manager)


if __name__ == '__main__':
    main()
