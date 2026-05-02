import numpy as np
import cv2
import os
from metavision_core.event_io import EventsIterator

DEBUG_MODE = True

def events_to_image(events, height, width):
    """
    イベント配列から2Dグレースケール画像を生成する関数
    """
    # 0で初期化された画像配列を作成
    img = np.zeros((height, width), dtype=np.float32)
    
    if len(events) == 0:
        return img.astype(np.uint8)

    # p == 1 がONイベント（明るくなった）、p == 0 がOFFイベント（暗くなった）
    on_events = events[events['p'] == 1]
    
    if len(on_events) == 0:
        return img.astype(np.uint8)

    # 抽出したONイベントだけでヒストグラムを作成
    np.add.at(img, (on_events['y'], on_events['x']), 1)
    
    # コントラストを強調するための正規化
    max_val = np.percentile(img, 99) 
    if max_val > 0:
        img = np.clip(img / max_val * 255.0, 0, 255)
    
    return img.astype(np.uint8)

def main():
    # ==========================================
    # 1. キャリブレーション設定
    # ==========================================
    RAW_LEFT = "events_master.raw"
    RAW_RIGHT = "events_slave.raw"
    
    # チェスボードの設定（内側の交点の数）
    # 実際の四角形の数ではなく、交点の数を指定します。例: 10x7のチェスボードなら、内側の交点は9x6になります。
    BOARD_SIZE = (6, 4) 
    SQUARE_SIZE = 0.0345714285714  # 1マスのサイズ（メートル単位）
    
    # スライディングウィンドウの設定
    STEP_MS = 10000  # 10ms（10000us）ごとにイベントを取得
    WINDOW_SIZE = 4  # 10ms × 4 = 40ms分のイベントを1つの画像に合成する

    OUTPUT_DIR = "scripts/calib_results"
    
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
    mv_it_left = EventsIterator(input_path=RAW_LEFT, delta_t=STEP_MS)
    mv_it_right = EventsIterator(input_path=RAW_RIGHT, delta_t=STEP_MS)
    
    height, width = mv_it_left.get_size()
    
    print("イベントデータの解析とコーナー検出を開始します...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    frame_count = 0
    success_count = 0

    # 過去のイベントを保持するリスト（スライディングウィンドウ用）
    history_left = []
    history_right = []

    # 左右のイテレータを同時に回す（ハードウェア同期されている前提）
    for ev_left, ev_right in zip(mv_it_left, mv_it_right):
        frame_count += 1

        # 新しいイベントをリストに追加
        history_left.append(ev_left)
        history_right.append(ev_right)
        
        # 規定のサイズ（4つ）を超えたら、一番古いものを捨てる
        if len(history_left) > WINDOW_SIZE:
            history_left.pop(0)
            history_right.pop(0)
            
        # まだ40ms分（4つ）溜まっていないうちは画像化をスキップ
        if len(history_left) < WINDOW_SIZE:
            continue

        # 過去40ms分のイベント配列をガッチャンコ（結合）して1つの配列にする
        merged_ev_left = np.concatenate(history_left)
        merged_ev_right = np.concatenate(history_right)
        
        # 結合したイベントを使って画像化（前回実装した極性フィルタリングが含まれる関数）
        img_left = events_to_image(merged_ev_left, height, width)
        img_right = events_to_image(merged_ev_right, height, width)
        
        # 点の隙間を埋めるためのぼかし
        img_left = cv2.GaussianBlur(img_left, (5, 5), 0)
        img_right = cv2.GaussianBlur(img_right, (5, 5), 0)
        
        # チェスボードのコーナー検出
        ret_l, corners_l = cv2.findChessboardCorners(img_left, BOARD_SIZE, None)
        ret_r, corners_r = cv2.findChessboardCorners(img_right, BOARD_SIZE, None)

        if DEBUG_MODE:
            cv2.imshow('Left', img_left)
            cv2.imshow('Right', img_right)
            cv2.waitKey(1)

            if ret_l or ret_r:
                print(f"フレーム {frame_count}: コーナー検出 - 左: {'成功' if ret_l else '失敗'}, 右: {'成功' if ret_r else '失敗'}")

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

            if DEBUG_MODE:
                # 左右画像を横に連結して、フレーム番号で保存
                merged = cv2.hconcat([img_left, img_right])
                save_path = os.path.join(OUTPUT_DIR, f"{frame_count}.png")
                cv2.imwrite(save_path, merged)

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