import pandas as pd
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import sys

class EventCloudViewer:
    def __init__(self, csv_path):
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
        self.min_ts = int(self.df['timestamp'].min())
        self.max_ts = int(self.df['timestamp'].max())
        
        # ---------------------------------------------------
        # 表示する時間枠（Time Window）: マイクロ秒単位
        # 500000us = 500ms
        # ---------------------------------------------------
        self.time_window = 500000 

        # 1. GUIアプリケーションの初期化
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Event 3D Interactive Viewer", 1280, 800)

        # 2. 3Dシーンを表示するウィジェット
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0, 0, 0, 1])

        # 3. コントロールパネルの構築
        em = self.window.theme.font_size
        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))
        
        # スライダーの作成
        self.slider = gui.Slider(gui.Slider.INT)
        slider_max = max(self.min_ts + 1, self.max_ts - self.time_window)
        self.slider.set_limits(self.min_ts, slider_max)
        self.slider.set_on_value_changed(self.on_slider_changed)
        
        self.time_label = gui.Label("Timestamp: ")
        self.panel.add_child(self.time_label)
        self.panel.add_child(self.slider)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)

        self.window.set_on_layout(self.on_layout)

        # 4. 点群マテリアル設定
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = 3.0

        # 色付けの基準となるZ座標の最小・最大値を事前に計算
        self.z_min = np.percentile(self.df['Z'], 1)
        self.z_max = np.percentile(self.df['Z'], 99)

        # 初期データの描画
        self.update_geometry(self.min_ts)
        
        # 初期カメラ位置の設定
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def on_layout(self, layout_context):
        r = self.window.content_rect
        panel_height = 80
        
        # シーンは上の領域
        self.scene_widget.frame = gui.Rect(
            r.x, 
            r.y, 
            r.width, 
            r.height - panel_height
        )
        
        # パネルは下の領域
        self.panel.frame = gui.Rect(
            r.x, 
            r.y + r.height - panel_height, 
            r.width, 
            panel_height
        )

    def on_slider_changed(self, value):
        self.update_geometry(int(value))

    def update_geometry(self, start_ts):
        self.time_label.text = f"Timestamp: {start_ts}  ~  {start_ts + self.time_window} (us)"
        
        # 高速なデータ抽出
        mask = (self.df['timestamp'] >= start_ts) & (self.df['timestamp'] < start_ts + self.time_window)
        sub_df = self.df[mask]

        if len(sub_df) == 0:
            return

        points = sub_df[['X', 'Y', 'Z']].values
        z_values = points[:, 2]
        
        # 事前計算した全体Zのmin/maxを使って色を計算
        z_norm = np.clip((z_values - self.z_min) / (self.z_max - self.z_min + 1e-6), 0.0, 1.0)
        
        cmap = plt.get_cmap('jet')
        colors = cmap(z_norm)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # ★ セグフォ対策: ジオメトリの安全な更新 ★
        if self.scene_widget.scene.has_geometry("events_points"):
            self.scene_widget.scene.remove_geometry("events_points")
            
        self.scene_widget.scene.add_geometry("events_points", pcd, self.material)
        self.window.post_redraw()

def main():
    try:
        viewer = EventCloudViewer("./output/points3d.csv")
        gui.Application.instance.run()
    except Exception as e:
        print(f"致命的なエラー: {e}")

if __name__ == "__main__":
    main()