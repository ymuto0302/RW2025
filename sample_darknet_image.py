import darknet
import cv2

def main():
    # YOLOv3 設定ファイルおよび重みファイルのパス
    config_file = 'cfg/yolov3.cfg'
    data_file = 'cfg/coco.data'
    weights = 'yolov3.weights'

    # Darknet モデルの読み込み
    network, class_names, class_colors = darknet.load_network(
            config_file,
            data_file,
            weights,
            batch_size=1
        )

    width = darknet.network_width(network)
    height = darknet.network_height(network)

    # Darknet が画像を処理するための領域を確保
    darknet_image = darknet.make_image(width, height, 3) # 3: color channels

    # 画像の読み込み
    image = cv2.imread('data/dog.jpg')
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Darknet の規定するサイズへ画像をリサイズ
    resized_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Darknet 用に画像形式を変換
    darknet.copy_image_from_bytes(darknet_image, resized_image.tobytes())

    # 推論の実行
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)

    # 検出されたオブジェクトの属性(ラベル，信頼度，bounding box)を出力
    # (メモ) 簡易表示は darknet.print_detections(detections) により可能
    # (メモ) bbox 内の値は x, y, w, h = bbox
    for detection in detections:
        label, confidence, bbox = detection
        print("\nObject:")
        print(f"Label: {label}")
        print(f"Score: {confidence}")
        print(f"Bbox: {bbox}")
        

    # 元画像に bounding box を重ねて描画
    object_image = darknet.draw_boxes(detections, resized_image, class_colors)

    # 画像の表示
    disp_image = cv2.cvtColor(object_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", disp_image)

    cv2.waitKey()

    # 後片付け：ウィンドウの削除
    cv2.destroyWindow("frame")
    
if __name__ == '__main__':
    main()
    
