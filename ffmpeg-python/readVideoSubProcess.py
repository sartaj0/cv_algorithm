import os, cv2
import subprocess
import numpy as np 


class VideoCapture():
	def __init__(self, ffmpeg_path, ffprobe_path, src):
		self.src = src 
		self.ffmpeg_path = ffmpeg_path
		self.ffprobe_path = ffprobe_path

		self.getWidthHeight()

		command = [ffmpeg_path,
			"-i", self.src,
			'-f', 'image2pipe',
			'-pix_fmt', 'rgb24',
			'-filter:v', 'fps=1',
			# '-vf', f'scale={self.w}:{self.h}',
			'-vcodec', 'rawvideo', '-']
		self.pipe = subprocess.Popen(command, stdout = subprocess.PIPE, bufsize=10**8)

	def getWidthHeight(self):
		ffprobe_cmd = f'{self.ffprobe_path} -v error -select_streams v:0 -show_entries stream=height,width -of csv=s=x:p=0 {self.src}'
		s = subprocess.Popen(ffprobe_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		ffprobe_out, err = s.communicate() 
		w, h = ffprobe_out.decode("utf-8").split("x")
		
		# self.w, self.h = int(w) // 2, int(h) // 2
		self.w, self.h = int(w), int(h)

	def read(self):
		if self.pipe.poll() is None:
			raw_image = self.pipe.stdout.read(self.h * self.w * 3)
			image = np.fromstring(raw_image, dtype='uint8')
			if image.shape[0] == 0:
				return None
			image = image.reshape((self.h, self.w, 3))

			return image
		else:
			return None

	def release(self):
		self.pipe.terminate()

if __name__ == "__main__":
	ffmpeg_path = r"E:\Downloads\ffmpeg-2021-11-18-git-85a6b7f7b7-essentials_build\bin\ffmpeg.exe"
	ffprobe_path = r"E:\Downloads\ffmpeg-2021-11-18-git-85a6b7f7b7-essentials_build\bin\ffprobe.exe"

	# src = r"E:\Videos\Testing\horse_race.mp4"
	src = "rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=0_stream=0.sdp"
	
	cap = VideoCapture(ffmpeg_path, ffprobe_path, src)
	# cap = cv2.VideoCapture(src)
	while True:
		frame = cap.read()
		if frame is None:
			break
		cv2.imshow("frame", frame)
		if cv2.waitKey(1) == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()