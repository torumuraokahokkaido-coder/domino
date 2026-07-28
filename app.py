import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="ドミノ運動解析 (Pro版)")

st.title("ドミノ運動解析アプリ 🎲 (最大30個対応 / 伝播速度対応)")

# サイドバー: パラメータ設定
st.sidebar.header("1. 解析パラメータ設定")
min_area = st.sidebar.slider("マーカーの最小サイズ (ピクセル)", 10, 500, 30)
movement_threshold = st.sidebar.slider("「動き出し」と判定する変位 (ピクセル)", 1, 30, 5, help="このピクセル数以上動いたら「倒れ始めた」と判定します")

# キャリブレーション（ピクセル -> mm 変換）設定
st.sidebar.header("2. スケール設定 (ピクセル → mm)")
st.sidebar.write("プレビュー画像で基準となる2点のX座標を読み取って入力してください。")
pixel_x1 = st.sidebar.number_input("点1の X座標 (px)", value=100)
pixel_x2 = st.sidebar.number_input("点2の X座標 (px)", value=300)
real_length_mm = st.sidebar.number_input("その間の実際の長さ (mm)", value=50.0)

# ピクセルからmmへの変換係数
if abs(pixel_x2 - pixel_x1) > 0:
    pixel_to_mm = real_length_mm / abs(pixel_x2 - pixel_x1)
else:
    pixel_to_mm = 1.0

uploaded_file = st.file_uploader("ドミノの動画を選択 (MP4 / MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 解析を一度だけ実行し、結果をキャッシュに保存する工夫
    @st.cache_data(show_spinner=False)
    def analyze_video(video_path, min_area, max_dominoes=30):
        cap = cv2.VideoCapture(video_path)
        records = []
        frame_images = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 赤色抽出
            lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([170, 100, 100]), np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_centers = []
            for c in contours:
                if cv2.contourArea(c) >= min_area:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        valid_centers.append((cx, cy))
            
            # X座標で並び替え（最大30個まで制限）
            valid_centers.sort(key=lambda pt: pt[0])
            valid_centers = valid_centers[:max_dominoes]
            
            # フレーム画像に認識点を描画して保存（プレビュー用）
            preview_img = frame_rgb.copy()
            for idx, (cx, cy) in enumerate(valid_centers):
                cv2.circle(preview_img, (cx, cy), 5, (0, 255, 0), -1) # 認識した点を緑で描画
                cv2.putText(preview_img, str(idx+1), (cx-10, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                records.append({
                    "Frame": frame_count,
                    "Time (s)": frame_count / fps,
                    "Domino ID": f"Domino {idx + 1:02d}",
                    "Px X": cx,
                    "Px Y": cy,
                })
                
            frame_images.append(preview_img)
            frame_count += 1
            
        cap.release()
        return pd.DataFrame(records), frame_images

    with st.spinner('動画を解析中...（数秒〜数十秒かかります）'):
        raw_df, frame_images = analyze_video(tfile.name, min_area)
    
    if not raw_df.empty:
        st.success("解析完了！")
        
        # --- プレビュー機能（コマ送り / スロー再生の代替） ---
        st.header("動画プレビューと認識結果の確認")
        frame_slider = st.slider("フレームを選択 (コマ送り)", 0, len(frame_images)-1, 0)
        
        # Plotlyを使って画像を表示（マウスオーバーで座標が確認できます）
        fig_img = px.imshow(frame_images[frame_slider])
        fig_img.update_layout(title="マウスを乗せるとピクセル座標(x, y)が表示されます", margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_img, use_container_width=True)
        
        # --- データ変換 (ピクセル -> mm) ---
        df = raw_df.copy()
        df["Position X (mm)"] = df["Px X"] * pixel_to_mm
        df["Position Y (mm)"] = df["Px Y"] * pixel_to_mm
        
        # 変位の計算
        df["Disp X (mm)"] = df.groupby("Domino ID")["Position X (mm)"].transform(lambda x: x - x.iloc[0])
        df["Disp Y (mm)"] = df.groupby("Domino ID")["Position Y (mm)"].transform(lambda x: -(x - x.iloc[0])) # Yは下向き正
        df["Total Disp (mm)"] = np.sqrt(df["Disp X (mm)"]**2 + df["Disp Y (mm)"]**2) # 総合変位
        
        # --- インタラクティブグラフ (表示/非表示可能) ---
        st.header("ドミノ変位グラフ (凡例クリックで表示切替)")
        fig_disp = px.line(df, x="Time (s)", y="Disp Y (mm)", color="Domino ID", 
                           title="各ドミノの縦方向(Y)の変位", markers=True)
        st.plotly_chart(fig_disp, use_container_width=True)
        
        # --- 伝播速度 (v-t) の計算 ---
        st.header("ドミノ伝播速度解析")
        
        start_times = []
        domino_ids = sorted(df["Domino ID"].unique())
        
        # 各ドミノが「閾値(movement_threshold)」を超えて動いた最初の時間を探す
        for d_id in domino_ids:
            d_data = df[df["Domino ID"] == d_id]
            # 変位が設定値を超えたフレームを抽出
            moved_data = d_data[d_data["Total Disp (mm)"] > (movement_threshold * pixel_to_mm)]
            
            if not moved_data.empty:
                start_time = moved_data.iloc[0]["Time (s)"]
                start_x = d_data.iloc[0]["Position X (mm)"]
                start_times.append({"Domino ID": d_id, "Start Time (s)": start_time, "Initial X (mm)": start_x})
        
        start_df = pd.DataFrame(start_times)
        
        if len(start_df) > 1:
            # 伝播速度の計算 (隣り合うドミノ間)
            start_df = start_df.sort_values("Initial X (mm)")
            start_df["Delta X (mm)"] = start_df["Initial X (mm)"].diff()
            start_df["Delta T (s)"] = start_df["Start Time (s)"].diff()
            
            # 速度 v = dx / dt (mm/s)
            start_df["Velocity (mm/s)"] = start_df["Delta X (mm)"] / start_df["Delta T (s)"]
            
            # v-t グラフの描画
            fig_v = px.line(start_df.dropna(), x="Start Time (s)", y="Velocity (mm/s)", 
                            markers=True, title="ドミノ間の伝播速度 (v-tグラフ)")
            fig_v.update_traces(line_color="red")
            st.plotly_chart(fig_v, use_container_width=True)
            
            st.write("各ドミノの動き出し時間と計算速度データ:")
            st.dataframe(start_df)
        else:
            st.warning("動き出しを検知できたドミノが少ないため、速度計算ができませんでした。サイドバーの「動き出し判定」の数値を下げてみてください。")

    else:
        st.error("マーカーが検出されませんでした。")
