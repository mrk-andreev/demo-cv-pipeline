import time
from multiprocessing import Lock
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np


def model_loop(shared_mem, dtype, shape, lock: Lock):
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
                lock.acquire()
                try:
                    arr = np.frombuffer(shared_mem.buf, dtype=dtype)
                    frame = arr.reshape(shape)
                    video_out.write(frame)
                finally:
                    lock.release()
            except Exception as e:
                print(e)
            finally:
                prev_time = time.time()
    finally:
        video_out.release()


def video_loop():
    cap = cv2.VideoCapture(0)

    video_out = None
    model_process = None
    shm = None
    shm_buf = None
    lock = Lock()

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret and not video_out:
                height, width = frame.shape[:2]
                video_out = cv2.VideoWriter(f'out/main_{int(time.time())}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0,
                                            (width, height))
            if not shm:
                d_size = frame.itemsize * np.prod(frame.shape)
                shm = SharedMemory(create=True, size=d_size, name="video_shared")
                shm_buf = np.ndarray(shape=frame.shape, dtype=frame.dtype, buffer=shm.buf)
                model_process = Process(target=model_loop, args=(shm, frame.dtype, frame.shape, lock))
                model_process.start()

            lock.acquire()
            try:
                shm_buf[:] = frame[:]
            finally:
                lock.release()

            video_out.write(frame)
    finally:
        cap.release()
        if video_out:
            video_out.release()
        if model_process:
            model_process.terminate()
        if shm:
            shm.close()


def main():
    video_loop()


if __name__ == '__main__':
    main()
