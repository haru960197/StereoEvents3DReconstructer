import sys
import cv2
import numpy as np
from metavision_core.event_io import EventsIterator

def events_to_image(events, height, width):
    """イベントデータを可視化用画像(BGR)に変換する"""
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    if events.size > 0:
        img[events['y'], events['x']] = np.where(events['p'][:, None] == 1, [255, 255, 255], [0, 0, 0])
    return img

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "./input/events_master.raw"
    print(f"Reading file: {file_path}")

    delta_t_us = 20000
    try:
        mv_iterator = EventsIterator(input_path=file_path, delta_t=delta_t_us)
        mv_iter = iter(mv_iterator)
    except Exception as e:
        print(f"Error opening file: {e}")
        sys.exit(1)

    height, width = mv_iterator.get_size()

    window_name = "Event Video Player"
    cv2.namedWindow(window_name)

    # 状態管理変数
    max_time_ms = 60000  # FIXME: 動画の長さに合わせて変更
    current_time_ms = 0
    seek_requested = False
    target_time_us = 0
    is_playing = True

    def on_time_trackbar(val):
        nonlocal seek_requested, target_time_us, current_time_ms
        if abs(val - current_time_ms) > max(100, delta_t_us // 1000):
            # ★修正: start_ts を delta_t の倍数に切り捨てる
            raw_target_us = val * 1000
            target_time_us = (raw_target_us // delta_t_us) * delta_t_us
            seek_requested = True

    def on_play_trackbar(val):
        nonlocal is_playing
        is_playing = (val == 1)

    # シークバーの配置
    cv2.createTrackbar("Time (ms)", window_name, 0, max_time_ms, on_time_trackbar)
    cv2.createTrackbar("Play(1)/Pause(0)", window_name, 1, 1, on_play_trackbar)

    print("Controls:\n- Spacebar: Play/Pause\n- 'q' or 'ESC': Exit")

    # 初期画像の用意
    img = np.full((height, width, 3), 128, dtype=np.uint8)

    while True:
        # シーク処理
        if seek_requested:
            seek_requested = False
            mv_iterator = EventsIterator(input_path=file_path, start_ts=target_time_us, delta_t=delta_t_us)
            mv_iter = iter(mv_iterator)
            current_time_ms = target_time_us // 1000
            
            # シーク直後は一時停止中でも1フレームだけ描画を更新する
            try:
                evs = next(mv_iter)
                img = events_to_image(evs, height, width)
            except StopIteration:
                pass

        # 再生中の処理
        elif is_playing:
            try:
                evs = next(mv_iter)
                img = events_to_image(evs, height, width)
                # シークバーの位置を同期
                current_time_ms = mv_iterator.current_time // 1000
                cv2.setTrackbarPos("Time (ms)", window_name, current_time_ms)
            except StopIteration:
                print("End of file reached.")
                is_playing = False
                cv2.setTrackbarPos("Play(1)/Pause(0)", window_name, 0)

        # 描画
        cv2.imshow(window_name, img)

        # キーボード入力受付
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):  # ESC または q
            break
        elif key == 32:  # スペースキーで再生/一時停止
            is_playing = not is_playing
            cv2.setTrackbarPos("Play(1)/Pause(0)", window_name, 1 if is_playing else 0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()