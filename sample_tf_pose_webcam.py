'''
run_webcam.py から本質のみを抜き出し，かつ keypoint 取得の関数を追加
'''
import cv2
import numpy as np
import time

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

fps_time = 0

'''
キーポイントの取得
'''
def get_keypoints(humans, image):
    if len(humans) > 0:
        # 最初に検出した人のみを対象とする
        human = humans[0]

        if len(human.body_parts) == 0:
            return None

        # キーポイントの取得
        # (メモ) human.body_parts は辞書形式
        keypoints = {}
        for _, body_part in human.body_parts.items():
            keypoints[body_part.get_part_name()] = (
                int(body_part.x * image.shape[1] + 0.5),
                int(body_part.y * image.shape[1] + 0.5))
        return keypoints
    else:
        return None


if __name__ == '__main__':
    # Pose Estimator の定義
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'),
                        target_size=(304, 272), trt_bool=False)

    # カメラの設定
    cam = cv2.VideoCapture(0)
    ret_val, image = cam.read()

    while True:
        # 画像の取り込み
        ret_val, image = cam.read()

        # Pose Estimator による推論
        # 返し値 humans: 複数の人間の関節情報が含まれる
        humans = e.inference(image, resize_to_default=True, upsample_size=2.0)

        # キーポイントの取得 ＆ 表示
        keypoints = get_keypoints(humans, image)
        if keypoints:
            for part, (x, y) in keypoints.items():
                print("{}: x={}, y={}".format(part, x, y))
        
        # logger.debug('postprocess+')
        # カメラから取り込んだ画像に関節情報を重ねた画像を生成
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # logger.debug('show+')
        # FPS の値を画像上に重ねる
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # 画像の表示
        cv2.imshow('tf-pose-estimation result', image)

        # FPS 算出のため，現時点の時刻を記録
        fps_time = time.time()

        # ESC キーが押されたら終了
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
