# OpenCV のインポート
import cv2

# Web カメラから映像を取り込むため，VideoCapture オブジェクトを生成
cap = cv2.VideoCapture(0)

while(True):
    # 画像の読み込み
    ret, frame = cap.read()

    if ret is True: # フレームのキャプチャが正常に出来たならば
        # 取り込まれた画像(フレーム)を表示 (ウィンドウ名 'Camera Sample')
        cv2.imshow('Camera Sample', frame)
        
    # ESC キーが押されるまで 10 milliseconds，待つ
    if cv2.waitKey(1) == 27:
        break

# 後片付け：カメラの開放 ＆ ウィンドウの削除
cap.release()
cv2.destroyWindow("frame")

