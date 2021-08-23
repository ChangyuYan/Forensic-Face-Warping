15:47:59 From M. Alex O. Vasilescu to Everyone:
	https://github.com/ondyari/FaceForensics

15:48:44 From M. Alex O. Vasilescu to Everyone:
	Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus
	Thies, and Matthias Nießner. Faceforensics: A large-scale video dataset for
	forgery detection in human faces. arXiv, 2018.
	[20] Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies,
	and Matthias Nießner. FaceForensics++: Learning to detect manipulated facial
	images. In International Conference on Computer Vision (ICCV), 2019

15:50:04 From M. Alex O. Vasilescu to Everyone:
	ince all videos have constant frame rate 30
	fps, we extracted up to 7 frames for each video by snapping almost
	one frame per each 30 seconds using OpenCV library in Python.
	Moreover, for detecting facial landmarks, we used pretrained dlib
	face detector8
	
15:50:55 From M. Alex O. Vasilescu to Everyone:
	http://dlib.net/face_landmark_detection.py.html

15:52:50 From M. Alex O. Vasilescu to Everyone:
	https://opencv.org/

16:02:05 From M. Alex O. Vasilescu to Everyone:
	https://www.google.com/search?q=dlib+facial+markers&source=lnms&tbm=isch&sa=X&ved=2ahUKEwijn9242sDyAhXiDjQIHY3xC8MQ_AUoAXoECAEQAw&biw=802&bih=493#imgrc=fLz0J9hUrZL8XM

16:06:04 From M. Alex O. Vasilescu to Everyone:
	https://www.google.com/search?q=image+warping&tbm=isch&ved=2ahUKEwj-vZio28DyAhUIo54KHWrlBxYQ2-cCegQIABAA&oq=image+warping&gs_lcp=CgNpbWcQAzIFCAAQgAQyBggAEAUQHjIGCAAQBRAeMgQIABAYMgQIABAYMgQIABAYMgQIABAYMgQIABAYMgQIABAYMgQIABAYUO4SWKcWYLoXaABwAHgAgAFwiAHBApIBAzAuM5gBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=SDUgYf75EIjG-gTqyp-wAQ&bih=493&biw=802

16:11:52 From M. Alex O. Vasilescu to Everyone:
	1. navigate database
	2. download database
	3. for every frame in all videos find the dlib facial markers
	4. save dlib marker for every frame
	5. warp images to template based on dlib markers 
	6. save warped images