import cv2

# Haar Cascade モデルの定義
face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# Web カメラから映像を取り込むため，VideoCapture オブジェクトを生成
cap = cv2.VideoCapture(0)

while(True):
    # 画像の読み込み
    ret, frame = cap.read()

    # 顔検出に cascade モデルを利用する場合，
    # 予めグレイスケールへ変換する必要がある。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 顔領域の検出
    front_faces = face_cascade.detectMultiScale(gray)

    # オリジナルの画像に顔領域の bounding box を重ねつつ，表示
    if len(front_faces) > 0:
        for (x, y, w, h) in front_faces:
            print("Bbox: ", [x, y, w, h])
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('sample', frame)
    else:
        print("Failed")

    if cv2.waitKey(1) == 27:
        break

# 後片付け：カメラの開放 ＆ ウィンドウの削除
cap.release()
cv2.destroyWindow("frame")

