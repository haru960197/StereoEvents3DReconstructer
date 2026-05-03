import pandas as pd
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt

class EventCloudViewer:
    def __init__(self, csv_path):
        print(f"{csv_path} を読み込んでいます...")
        self.df = pd.read_csv(csv_path)
        
        if len(self.df) == 0:
            raise ValueError("CSVファイルにデータがありません。")

        # タイムスタンプの最小値・最大値を取得
        self.min_ts = int(self.df['timestamp'].min())
        self.max_ts = int(self.df['timestamp'].max())
        
        # ---------------------------------------------------
        # 重要設定：表示する「時間枠（Time Window）」
        # イベントカメラは一瞬では点群が少なすぎるため、
        # 「スライダーの位置から何マイクロ秒分のデータを表示するか」を決めます
        # 100000us = 100ms（顔の動きを見るのにちょうどいい時間枠です）
        # ---------------------------------------------------
        self.time_window = 100000 

        # 1. GUIアプリケーションの初期化
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Event 3D Interactive Viewer", 1280, 800)

        # 2. 3Dシーンを表示するウィジェット（メイン画面）
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0, 0, 0, 1]) # 背景を黒に設定

        # 3. コントロールパネルとスライダーの構築
        em = self.window.theme.font_size
        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))
        
        # スライダーの作成（最小値 〜 最大値-時間枠）
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(self.min_ts, max(self.min_ts + 1, self.max_ts - self.time_window))
        self.slider.set_on_value_changed(self.on_slider_changed)
        
        # ラベルとスライダーをパネルに追加
        self.time_label = gui.Label("Timestamp: ")
        self.panel.add_child(self.time_label)
        self.panel.add_child(self.slider)

        # ウィンドウにウィジェットを追加
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)

        # ウィンドウサイズ変更時にレイアウトを整えるコールバック
        self.window.set_on_layout(self.on_layout)

        # 4. 点群描画用のマテリアル（質感）設定
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit" # 影をつけずにそのままの色を出す
        self.material.point_size = 3.0        # 点の大きさ（見えにくければ数値を上げる）

        # 初期データの描画
        self.update_geometry(self.min_ts)
        
        # カメラ位置のリセット（点群が画面の真ん中に来るようにする）
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def on_layout(self, layout_context):
        # 画面サイズが変更されたときのUI配置設定
        r = self.window.content_rect
        panel_height = 80 # 下部のUIパネルの高さ
        
        # 3Dシーンは上の領域、パネルは下の領域に配置
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width, r.height - panel_height)
        self.panel.frame = gui.Rect(r.x, r.bottom - panel_height, r.width, panel_height)

    def on_slider_changed(self, value):
        # スライダーが動かされたらジオメトリ（点群）を更新
        self.update_geometry(int(value))

    def update_geometry(self, start_ts):
        # UIラベルの更新
        self.time_label.text = f"Timestamp: {start_ts}  ~  {start_ts + self.time_window} (us)"
        
        # 指定された時間枠のデータだけをPandasで高速に抽出
        mask = (self.df['timestamp'] >= start_ts) & (self.df['timestamp'] < start_ts + self.time_window)
        sub_df = self.df[mask]

        if len(sub_df) == 0:
            return

        points = sub_df[['X', 'Y', 'Z']].values

        # Z座標（奥行き）で色付けする処理（前回のコードと同じ）
        z_values = points[:, 2]
        z_min, z_max = np.percentile(self.df['Z'], 1), np.percentile(self.df['Z'], 99) # 色の基準は全体で固定
        z_norm = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0.0, 1.0)
        
        cmap = plt.get_cmap('jet')
        colors = cmap(z_norm)[:, :3]

        # Open3Dの点群オブジェクトを作成
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # シーンから古い点群を削除し、新しい点群を再登録する
        self.scene_widget.scene.remove_geometry("events_points")
        self.scene_widget.scene.add_geometry("events_points", pcd, self.material)
        
        # 再描画を要求
        self.window.post_redraw()

def main():
    try:
        # ご自身のCSVファイル名を指定してください
        viewer = EventCloudViewer("points3d.csv")
        gui.Application.instance.run()
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()