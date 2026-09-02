import os
import sys
import json
import argparse
import threading
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

DEFAULT_CONFIG = {
    "csv_path": "output/points3d.csv",
    "output_video_path": "output/events_3d.mp4",
    "fps": 30,
    "time_window_us": 8000,
    "step_time_us": 8000,
    "video_width": 1280,
    "video_height": 720,
    "point_size": 3.0,
    "remove_outliers": False,
    "outlier_nb_neighbors": 20,
    "outlier_std_ratio": 2.0,
}

def load_config(config_path):
    """設定ファイルを読み込む。存在しない場合はデフォルト値を返す。"""
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                config.update(user_config)
            print(f"設定ファイルを読み込みました: {config_path}")
        except Exception as e:
            print(f"警告: 設定ファイルの読み込みに失敗しました ({e})。デフォルト設定を使用します。")
    return config


class EventCloudViewer:
    def __init__(self, config):
        self.config = config
        csv_path = config["csv_path"]

        print(f"{csv_path} を読み込んでいます...")
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"CSV読み込みエラー: {e}")
            sys.exit(1)

        if len(self.df) == 0:
            print("CSVファイルにデータがありません。")
            sys.exit(1)

        # タイムスタンプの最小値・最大値を取得
        self.min_ts = int(self.df["timestamp"].min())
        self.max_ts = int(self.df["timestamp"].max())
        self.time_window = int(config["time_window_us"])
        self.step_time = int(config["step_time_us"])
        self.fps = int(config["fps"])
        self.output_video_path = config["output_video_path"]
        self.video_width = int(config["video_width"])
        self.video_height = int(config["video_height"])
        self.point_size = float(config["point_size"])
        self.remove_outliers = bool(config.get("remove_outliers", False))

        self.current_ts = self.min_ts
        self.slider_max = max(self.min_ts + 1, self.max_ts - self.time_window)

        # 再生・エクスポート管理
        self.is_playing = False
        self.is_exporting = False
        self.play_thread = None

        # 1. GUIアプリケーションの初期化
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Event 3D Interactive Viewer", 1280, 800
        )

        # 2. 3Dシーンを表示するウィジェット
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0, 0, 0, 1])

        # キーイベントのハンドリング (スペースキーでPlay/Pause)
        self.scene_widget.set_on_key(self.on_key_event)

        # 3. コントロールパネルの構築
        em = self.window.theme.font_size
        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))

        # タイムスタンプ情報ラベル
        self.time_label = gui.Label("Timestamp: ")
        self.panel.add_child(self.time_label)

        # スライダー（シークバー）
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(self.min_ts, self.slider_max)
        self.slider.set_on_value_changed(self.on_slider_changed)
        self.panel.add_child(self.slider)

        # コントロールボタンバー (Play/Pause, Reset, Export MP4, Status)
        self.controls_layout = gui.Horiz(em)

        self.play_button = gui.Button("Play")
        self.play_button.set_on_clicked(self.toggle_play)
        self.controls_layout.add_child(self.play_button)

        self.reset_button = gui.Button("Reset")
        self.reset_button.set_on_clicked(self.on_reset_clicked)
        self.controls_layout.add_child(self.reset_button)

        self.export_button = gui.Button("Export MP4")
        self.export_button.set_on_clicked(self.start_export_mp4)
        self.controls_layout.add_child(self.export_button)

        self.status_label = gui.Label(f"Ready | FPS: {self.fps} | Step: {self.step_time}us")
        self.controls_layout.add_child(self.status_label)

        self.panel.add_child(self.controls_layout)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)

        self.window.set_on_layout(self.on_layout)

        # 4. 点群マテリアル設定
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = self.point_size

        # 色付けの基準となるZ座標の最小・最大値を事前に計算
        self.z_min = float(np.percentile(self.df["Z"], 1))
        self.z_max = float(np.percentile(self.df["Z"], 99))
        self.cmap = plt.get_cmap("jet")

        # 初期データの描画
        self.update_geometry(self.min_ts)

        # 初期カメラ位置の設定
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def on_layout(self, layout_context):
        r = self.window.content_rect
        panel_height = 110

        # シーンは上の領域
        self.scene_widget.frame = gui.Rect(
            r.x, r.y, r.width, max(1, r.height - panel_height)
        )

        # パネルは下の領域
        self.panel.frame = gui.Rect(
            r.x, r.y + max(0, r.height - panel_height), r.width, panel_height
        )

    def on_key_event(self, event):
        """キーボード操作の受付（スペースキーで再生/一時停止）"""
        if event.type == gui.KeyEvent.UP:
            if event.key == gui.KeyName.SPACE or event.key == 32:
                self.toggle_play()
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def on_slider_changed(self, value):
        """シークバーを手動で操作したときのコールバック"""
        self.current_ts = int(value)
        self.update_geometry(self.current_ts)

    def on_reset_clicked(self):
        """タイムスタンプを先頭に戻す"""
        if self.is_exporting:
            return
        self.current_ts = self.min_ts
        self.slider.int_value = self.current_ts
        self.update_geometry(self.current_ts)

    def toggle_play(self):
        """再生/一時停止の切り替え"""
        if self.is_exporting:
            return
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()

    def start_play(self):
        self.is_playing = True
        self.play_button.text = "Pause"
        self.status_label.text = "Playing..."
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()

    def stop_play(self):
        self.is_playing = False
        self.play_button.text = "Play"
        self.status_label.text = "Paused"

    def _play_loop(self):
        """バックグラウンドで一定周期ごとに次フレーム描画をリクエスト"""
        interval = 1.0 / max(1, self.fps)
        while self.is_playing:
            t0 = time.time()
            gui.Application.instance.post_to_main_thread(
                self.window, self._advance_frame
            )
            elapsed = time.time() - t0
            sleep_sec = max(0.001, interval - elapsed)
            time.sleep(sleep_sec)

    def _advance_frame(self):
        """次フレームへ進める（メインスレッドで実行）"""
        if not self.is_playing:
            return
        next_ts = self.current_ts + self.step_time
        if next_ts > self.slider_max:
            # 終端に達したら最初からループ再生
            self.current_ts = self.min_ts
        else:
            self.current_ts = next_ts

        self.slider.int_value = self.current_ts
        self.update_geometry(self.current_ts)

    def update_geometry(self, start_ts):
        """指定したタイムスタンプの点群データを描画"""
        self.time_label.text = (
            f"Timestamp: {start_ts}  ~  {start_ts + self.time_window} (us)"
        )

        mask = (self.df["timestamp"] >= start_ts) & (
            self.df["timestamp"] < start_ts + self.time_window
        )
        sub_df = self.df[mask]

        if self.scene_widget.scene.has_geometry("events_points"):
            self.scene_widget.scene.remove_geometry("events_points")

        if len(sub_df) == 0:
            self.window.post_redraw()
            return

        points = sub_df[["X", "Y", "Z"]].values
        z_values = points[:, 2]

        # Z座標の正規化とカラーマップ適用
        z_norm = np.clip(
            (z_values - self.z_min) / (self.z_max - self.z_min + 1e-6), 0.0, 1.0
        )
        colors = self.cmap(z_norm)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 外れ値除去（オプション）
        if self.remove_outliers and len(pcd.points) > self.config.get("outlier_nb_neighbors", 20):
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=int(self.config.get("outlier_nb_neighbors", 20)),
                std_ratio=float(self.config.get("outlier_std_ratio", 2.0)),
            )
            pcd = pcd.select_by_index(ind)

        # ジオメトリの安全な更新
        self.scene_widget.scene.add_geometry("events_points", pcd, self.material)
        self.window.post_redraw()

    def start_export_mp4(self):
        """現在のカメラ視点のまま全フレームをレンダリングしてMP4出力"""
        if self.is_exporting:
            return
        if self.is_playing:
            self.stop_play()

        self.is_exporting = True
        self.play_button.enabled = False
        self.reset_button.enabled = False
        self.export_button.enabled = False
        self.slider.enabled = False

        threading.Thread(target=self._export_worker, daemon=True).start()

    def _export_worker(self):
        """動画エクスポート処理用ワーカースレッド"""
        os.makedirs(os.path.dirname(self.output_video_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.fps,
            (self.video_width, self.video_height),
        )

        timestamps = list(range(self.min_ts, self.slider_max + 1, self.step_time))
        total_frames = len(timestamps)

        print(f"\n[MP4 Export 開始]")
        print(f"出力先: {self.output_video_path}")
        print(f"解像度: {self.video_width}x{self.video_height}, FPS: {self.fps}, 総フレーム数: {total_frames}")

        event = threading.Event()
        captured_img = [None]

        def _render_and_capture(ts):
            self.current_ts = ts
            self.slider.int_value = ts
            self.update_geometry(ts)
            img = gui.Application.instance.render_to_image(
                self.scene_widget.scene, self.video_width, self.video_height
            )
            captured_img[0] = np.asarray(img)
            event.set()

        for idx, ts in enumerate(timestamps):
            event.clear()
            gui.Application.instance.post_to_main_thread(
                self.window, lambda t=ts: _render_and_capture(t)
            )
            event.wait()

            # RGB -> BGR に変換して書き込み
            bgr_frame = cv2.cvtColor(captured_img[0], cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

            progress = (idx + 1) * 100 // total_frames
            if (idx + 1) % 10 == 0 or idx + 1 == total_frames:
                print(f"エクスポート進行状況: {progress}% ({idx + 1}/{total_frames})", end="\r")
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda p=progress, i=idx+1, tot=total_frames: setattr(
                        self.status_label, "text", f"Exporting: {p}% ({i}/{tot})"
                    ),
                )

        writer.release()
        print(f"\n[MP4 Export 完了] 保存先: {self.output_video_path}")

        def _finish_export():
            self.is_exporting = False
            self.play_button.enabled = True
            self.reset_button.enabled = True
            self.export_button.enabled = True
            self.slider.enabled = True
            self.status_label.text = f"Saved: {os.path.basename(self.output_video_path)}"

        gui.Application.instance.post_to_main_thread(self.window, _finish_export)


def export_headless(config):
    """GUIなしでオフスクリーンレンダラーを使用してMP4を生成する"""
    csv_path = config["csv_path"]
    output_path = config["output_video_path"]
    fps = int(config["fps"])
    time_window = int(config["time_window_us"])
    step_time = int(config["step_time_us"])
    width = int(config["video_width"])
    height = int(config["video_height"])
    point_size = float(config["point_size"])
    remove_outliers = bool(config.get("remove_outliers", False))

    print(f"[Headless Export] {csv_path} を読み込んでいます...")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("CSVファイルにデータがありません。")
        return

    min_ts = int(df["timestamp"].min())
    max_ts = int(df["timestamp"].max())
    slider_max = max(min_ts + 1, max_ts - time_window)

    # オフスクリーンレンダラー初期化
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0, 0, 0, 1])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size

    z_min = float(np.percentile(df["Z"], 1))
    z_max = float(np.percentile(df["Z"], 99))
    cmap = plt.get_cmap("jet")

    # 全体のバウンディングボックスから適切なカメラ視点を自動計算
    all_points = df[["X", "Y", "Z"]].values
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(all_points)
    bbox = pcd_all.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    diag = float(np.linalg.norm(extent))
    fov = 60.0
    dist = (diag / 2.0) / np.tan(np.radians(fov / 2.0))
    # 点群の手前（-Z方向）から中心を見上げる
    eye = center + np.array([0.0, 0.0, -dist * 1.2])
    up = np.array([0.0, -1.0, 0.0])
    near_clip = max(0.001, dist * 0.01)
    far_clip = max(100.0, dist * 10.0)
    renderer.setup_camera(fov, center, eye, up, near_clip=near_clip, far_clip=far_clip)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    timestamps = list(range(min_ts, slider_max + 1, step_time))
    total_frames = len(timestamps)

    print(f"レンダリング開始: {total_frames} フレーム, 出力先: {output_path}")
    t0 = time.time()

    for idx, ts in enumerate(timestamps):
        mask = (df["timestamp"] >= ts) & (df["timestamp"] < ts + time_window)
        sub_df = df[mask]

        if renderer.scene.has_geometry("events"):
            renderer.scene.remove_geometry("events")

        if len(sub_df) > 0:
            points = sub_df[["X", "Y", "Z"]].values
            z_values = points[:, 2]
            z_norm = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0.0, 1.0)
            colors = cmap(z_norm)[:, :3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if remove_outliers and len(pcd.points) > config.get("outlier_nb_neighbors", 20):
                cl, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=int(config.get("outlier_nb_neighbors", 20)),
                    std_ratio=float(config.get("outlier_std_ratio", 2.0)),
                )
                pcd = pcd.select_by_index(ind)

            renderer.scene.add_geometry("events", pcd, mat)

        img = renderer.render_to_image()
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        writer.write(bgr)

        if (idx + 1) % 20 == 0 or idx + 1 == total_frames:
            progress = (idx + 1) * 100 // total_frames
            print(f"進行状況: {progress}% ({idx + 1}/{total_frames})", end="\r")

    writer.release()
    t1 = time.time()
    print(f"\n[完了] {total_frames} フレームを {t1 - t0:.2f} 秒で出力しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="3D時系列イベント点群ビューワー & 動画エクスポートツール")
    parser.add_argument("csv_path", nargs="?", help="入力点群CSVファイルのパス")
    parser.add_argument("--config", default="config/visualize_3d.json", help="設定ファイルのパス")
    parser.add_argument("--output-video", help="出力先MP4ファイルのパス")
    parser.add_argument("--fps", type=int, help="動画フレームレート")
    parser.add_argument("--time-window", type=int, help="表示時間幅 (マイクロ秒)")
    parser.add_argument("--step-time", type=int, help="1フレームごとの進行時間 (マイクロ秒)")
    parser.add_argument("--point-size", type=float, help="点群の描画サイズ")
    parser.add_argument("--export-video", action="store_true", help="GUIを開かずに直接動画を生成して終了する")
    parser.add_argument("--headless", action="store_true", help="オフスクリーンレンダラーで動画を出力する")

    args = parser.parse_args()

    # 設定の読み込みとコマンドライン引数による上書き
    config = load_config(args.config)

    if args.csv_path:
        config["csv_path"] = args.csv_path
    if args.output_video:
        config["output_video_path"] = args.output_video
    if args.fps is not None:
        config["fps"] = args.fps
    if args.time_window is not None:
        config["time_window_us"] = args.time_window
    if args.step_time is not None:
        config["step_time_us"] = args.step_time
    if args.point_size is not None:
        config["point_size"] = args.point_size

    # CSVの存在確認
    if not os.path.exists(config["csv_path"]):
        print(f"エラー: 指定されたCSVファイルが見つかりません: {config['csv_path']}")
        sys.exit(1)

    if args.export_video or args.headless:
        export_headless(config)
    else:
        try:
            viewer = EventCloudViewer(config)
            gui.Application.instance.run()
        except Exception as e:
            print(f"致命的なエラー: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()