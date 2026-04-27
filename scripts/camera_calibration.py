import numpy as np
import cv2
from metavision_core.event_io import EventsIterator

def events_to_image(events, height, width):
    """
    イベント配列から2Dグレースケール画像を生成する関数
    """
    # 0で初期化された画像配列を作成
    img = np.zeros((height, width), dtype=np.float32)
    
    if len(events) == 0:
        return img.astype(np.uint8)

    # ピクセルごとにイベント数をカウント（ヒストグラム化）
    # np.add.at は重複するインデックスの値を正しく加算します
    np.add.at(img, (events['y'], events['x']), 1)
    
    # コントラストを強調するための正規化（外れ値の影響を下げるためパーセンタイルを使用）
    max_val = np.percentile(img, 99) 
    if max_val > 0:
        img = np.clip(img / max_val * 255.0, 0, 255)
    
    return img.astype(np.uint8)

def main():
    # ==========================================
    # 1. キャリブレーション設定
    # ==========================================
    RAW_LEFT = "calib_master.raw"
    RAW_RIGHT = "calib_slave.raw"
    
    # チェスボードの設定（内側の交点の数）
    BOARD_SIZE = (9, 6) 
    SQUARE_SIZE = 0.025  # 1マスのサイズ（メートル単位、例: 25mm）
    
    # タイムウィンドウ（マイクロ秒）。例: 50ms = 50000us
    # 点滅ディスプレイの周波数に合わせて調整してください
    DELTA_T = 50000 
    
    # OpenCVのコーナー検出サブピクセル精度の終了基準
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D空間上のチェスボードのコーナー座標を準備 (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # 画像上の点と3D上の点を保存するリスト
    objpoints = [] # 3D空間上の点
    imgpoints_left = [] # 左画像の2D点
    imgpoints_right = [] # 右画像の2D点

    # ==========================================
    # 2. RAWファイルの同期読み込みとコーナー検出
    # ==========================================
    # EventsIterator を使うと、指定した時間(delta_t)ごとにイベントを取得できます
    mv_it_left = EventsIterator(input_path=RAW_LEFT, delta_t=DELTA_T)
    mv_it_right = EventsIterator(input_path=RAW_RIGHT, delta_t=DELTA_T)
    
    height, width = mv_it_left.get_size()
    
    print("イベントデータの解析とコーナー検出を開始します...")
    
    frame_count = 0
    success_count = 0

    # 左右のイテレータを同時に回す（ハードウェア同期されている前提）
    for ev_left, ev_right in zip(mv_it_left, mv_it_right):
        frame_count += 1
        
        # イベントを画像に変換 
        img_left = events_to_image(ev_left, height, width)
        img_right = events_to_image(ev_right, height, width)
        
        # チェスボードのコーナー検出
        ret_l, corners_l = cv2.findChessboardCorners(img_left, BOARD_SIZE, None)
        ret_r, corners_r = cv2.findChessboardCorners(img_right, BOARD_SIZE, None)

        # 左右両方の画像でコーナーが見つかった場合のみペアとして保存
        if ret_l and ret_r:
            success_count += 1
            
            # サブピクセル精度に細線化
            corners2_l = cv2.cornerSubPix(img_left, corners_l, (11, 11), (-1, -1), criteria)
            corners2_r = cv2.cornerSubPix(img_right, corners_r, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints_left.append(corners2_l)
            imgpoints_right.append(corners2_r)
            
            print(f"成功ペア追加: {success_count} (Frame: {frame_count})")
            
            # 検出結果を確認したい場合は以下のコメントアウトを外す
            cv2.drawChessboardCorners(img_left, BOARD_SIZE, corners2_l, ret_l)
            cv2.drawChessboardCorners(img_right, BOARD_SIZE, corners2_r, ret_r)
            cv2.imshow('Left', img_left)
            cv2.imshow('Right', img_right)
            cv2.waitKey(1)

    print(f"\nコーナー検出終了: 全 {frame_count} フレーム中、{success_count} ペアのコーナーを取得しました。")
    cv2.destroyAllWindows()

    if success_count < 10:
        print("エラー: 有効なコーナーペアが少なすぎます（最低10〜20ペア推奨）。DELTA_Tや撮影条件を見直してください。")
        return

    # ==========================================
    # 3. 単眼カメラキャリブレーション（内部パラメータの初期推定）
    # ==========================================
    print("\n左カメラのキャリブレーション中...")
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_left, (width, height), None, None)
    
    print("右カメラのキャリブレーション中...")
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_right, (width, height), None, None)

    # ==========================================
    # 4. ステレオキャリブレーション（外部パラメータの算出）
    # ==========================================
    print("\nステレオキャリブレーション中...")
    flags = cv2.CALIB_FIX_INTRINSIC
    
    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l,
        mtx_r, dist_r,
        (width, height),
        criteria=criteria,
        flags=flags
    )

    print(f"ステレオキャリブレーションの再投影誤差 (RMS): {ret_stereo:.4f}")
    print("\n--- 結果 ---")
    print("[Left Camera Matrix]")
    print(mtx_l)
    print("\n[Right Camera Matrix]")
    print(mtx_r)
    print("\n[Rotation Matrix (Left to Right)]")
    print(R)
    print("\n[Translation Vector (Left to Right)]")
    print(T)

    # 必要に応じてJSONやnpz形式で保存する処理を追加

if __name__ == "__main__":
    main()