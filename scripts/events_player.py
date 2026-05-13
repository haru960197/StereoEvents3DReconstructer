import sys
import cv2
import numpy as np
from metavision_core.event_io import EventsIterator

def events_to_image(events, height, width):
    """
    イベントデータを可視化用画像(BGR)に変換するヘルパー関数
    """
    # 背景をグレー(128)で初期化
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    
    # イベントが存在する場合のみ色を塗る
    if events.size > 0:
        # 極性pが1(ONイベント)なら白、0(OFFイベント)なら黒に設定
        img[events['y'], events['x']] = np.where(events['p'][:, None] == 1, [255, 255, 255], [0, 0, 0])
    return img

def main():
    # 1. 実行時引数の処理（引数がなければデフォルトパスを使用）
    file_path = sys.argv[1] if len(sys.argv) > 1 else "./input/events_master.raw"
    print(f"Reading file: {file_path}")

    # 2. イベントイテレータの初期化
    delta_t_us = 20000
    try:
        mv_iterator = EventsIterator(input_path=file_path, delta_t=delta_t_us)
        mv_iter = iter(mv_iterator)
    except Exception as e:
        print(f"Error opening file: {e}")
        sys.exit(1)

    height, width = mv_iterator.get_size()

    # 3. GUIの初期化 (OpenCV)
    window_name = "Event Video Player"
    cv2.namedWindow(window_name)

    # シークバー用の変数管理
    max_time_ms = 60000  # FIXME: 動画の長さに合わせて変更してください（例: 60秒 = 60000）
    current_time_ms = 0
    seek_requested = False
    target_time_us = 0

    def on_trackbar(val):
        nonlocal seek_requested, target_time_us, current_time_ms
        # cv2.setTrackbarPosによるプログラムからの自動更新と、ユーザーの手動シーク操作を区別する
        if abs(val - current_time_ms) > 100:
            target_time_us = val * 1000 # ms を us に変換
            seek_requested = True

    # ウィンドウにシークバーをアタッチ
    cv2.createTrackbar("Time (ms)", window_name, 0, max_time_ms, on_trackbar)

    print("Playing... Press 'q' or 'ESC' to exit.")

    # 4. メインループ
    while True:
        # シーク要求があった場合、イテレータを開始時間を指定して作り直す
        if seek_requested:
            seek_requested = False
            mv_iterator = EventsIterator(input_path=file_path, start_ts=target_time_us, delta_t=delta_t_us)
            mv_iter = iter(mv_iterator)
            current_time_ms = target_time_us // 1000

        try:
            # 次の20ms分のイベント群を取得
            evs = next(mv_iter)
        except StopIteration:
            print("End of file reached.")
            break

        # イベントを画像(2D NumPyアレイ)に変換
        img = events_to_image(evs, height, width)

        # 現在の時刻をシークバーに反映させる
        current_time_ms = mv_iterator.current_time // 1000
        cv2.setTrackbarPos("Time (ms)", window_name, current_time_ms)

        # 描画
        cv2.imshow(window_name, img)

        # 20ms待機 (C++のFPS=50相当のウェイト) & キー入力受付
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'): # ESC または q
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()