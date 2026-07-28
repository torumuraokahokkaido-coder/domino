import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="ドミノ解析 Pro (伝播＆個別速度対応)")

st.title("ドミノ運動解析アプリ 🎲 (伝播速度 ＆ 個別倒伏速度対応)")

# --- サイドバー設定 ---
st.sidebar.header("1. 解析パラメータ設定")
min_area = st.sidebar.slider("マーカーの最小サイズ (ピクセル)", 10, 500, 30)
movement_threshold = st.sidebar.slider("「動き出し」判定の変位 (ピクセル)", 1, 30, 5)

st.sidebar.header("2. 速度計算ノイズ除去")
smooth_window = st.sidebar.slider("速度の平滑化（移動平均フレーム数）", 1, 10, 3, 
                               help="数値が大きいほどグラフのギザギザが滑らかになります")

st.sidebar.header("3. スケール設定 (ピクセル → mm)")
pixel_x1 = st.sidebar.number_input("点1の X座標 (px)", value=100)
pixel_x2 = st.sidebar.number_input("点2の X座標 (px)", value=300)
real_length_mm = st.sidebar.number_input("実際の長さ (mm)", value=50.0)

pixel_to_mm = real_length_mm / abs(pixel_x2 - pixel_x1) if abs(pixel_x2 - pixel_x1) > 0 else 1.0

uploaded_file = st.file_uploader("ドミノの動画を選択 (MP4 / MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    @st.cache_data(show_spinner=False)
    def analyze_video(video_path, min_area, max_dominoes=30):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
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
            
            valid_centers.sort(key=lambda pt: pt[0])
            valid_centers = valid_centers[:max_dominoes]
            
            # プレビュー画像生成
            preview_img = frame_rgb.copy()
            for idx, (cx, cy) in enumerate(valid_centers):
                cv2.circle(preview_img, (cx, cy), 5, (0, 255, 0), -1)
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
        return pd.DataFrame(records), frame_images, fps

    with st.spinner('動画を解析中...'):
        raw_df, frame_images, fps = analyze_video(tfile.name, min_area)
    
    if not raw_df.empty:
        st.success("解析完了！")
        
        # --- 1. プレビュー画面 ---
        st.header("1. 動画プレビューと認識チェック")
        frame_slider = st.slider("フレームを選択 (コマ送り)", 0, len(frame_images)-1, 0)
        fig_img = px.imshow(frame_images[frame_slider])
        fig_img.update_layout(title="マウスオーバーでピクセル座標(x, y)を確認できます", margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_img, use_container_width=True)
        
        # --- 2. 座標と変位の物理単位変換 ---
        df = raw_df.copy()
        df["Position X (mm)"] = df["Px X"] * pixel_to_mm
        df["Position Y (mm)"] = df["Px Y"] * pixel_to_mm
        
        df["Disp X (mm)"] = df.groupby("Domino ID")["Position X (mm)"].transform(lambda x: x - x.iloc[0])
        # Y軸は下がプラス方向のため、下がっていく動き（落下）を正（プラス）として計算
        df["Disp Y (mm)"] = df.groupby("Domino ID")["Position Y (mm)"].transform(lambda x: x - x.iloc[0])
        df["Total Disp (mm)"] = np.sqrt(df["Disp X (mm)"]**2 + df["Disp Y (mm)"]**2)
        
        # --- 3. 個別速度（Y軸方向の落下速度）の計算 ---
        # 時間の差分 dt
        dt = 1.0 / fps
        
        # 各ドミノごとにフレーム間の移動量(dY)から速度 dy/dt (mm/s) を算出
        df["Raw Velocity Y (mm/s)"] = df.groupby("Domino ID")["Disp Y (mm)"].diff() / dt
        
        # ノイズ除去（移動平均フィルター）
        df["Velocity Y (mm/s)"] = df.groupby("Domino ID")["Raw Velocity Y (mm/s)"].transform(
            lambda x: x.rolling(window=smooth_window, min_periods=1).mean()
        )
        
        # --- 4. グラフ表示部 ---
        st.header("2. 解析グラフ")
        
        tab1, tab2, tab3 = st.tabs(["① 変位グラフ", "② ドミノ個体の倒伏速度 (v-t)", "③ 波の伝播速度"])
        
        # タブ①: 変位
        with tab1:
            fig_disp = px.line(df, x="Time (s)", y="Disp Y (mm)", color="Domino ID", 
                               title="各ドミノのY軸方向の変位（落下量）", markers=True)
            st.plotly_chart(fig_disp, use_container_width=True)
            
        # タブ②: 各ドミノ個体の倒れる速度 (新機能)
        with tab2:
            st.subheader("各ドミノ個体が倒れ込む速度 (Y軸方向の落下速度)")
            st.write("ドミノが倒れて頭部が落下していくスピードの変化です。凡例をクリックして特定のドミノのみを表示できます。")
            
            fig_ind_v = px.line(df, x="Time (s)", y="Velocity Y (mm/s)", color="Domino ID",
                                 title="ドミノ個体の倒伏速度 (mm/s)", markers=True)
            st.plotly_chart(fig_ind_v, use_container_width=True)
            
            # 各ドミノの最大倒下速度のまとめ
            max_v_df = df.groupby("Domino ID")["Velocity Y (mm/s)"].max().reset_index()
            max_v_df.columns = ["Domino ID", "最大倒伏速度 (mm/s)"]
            
            st.write("各ドミノの最高倒下速度一覧:")
            st.dataframe(max_v_df.T)

        # タブ③: ドミノ間の伝播速度
        with tab3:
            st.subheader("ドミノ間の伝播速度（衝撃波の伝達スピード）")
            start_times = []
            domino_ids = sorted(df["Domino ID"].unique())
            
            for d_id in domino_ids:
                d_data = df[df["Domino ID"] == d_id]
                moved_data = d_data[d_data["Total Disp (mm)"] > (movement_threshold * pixel_to_mm)]
                
                if not moved_data.empty:
                    start_time = moved_data.iloc[0]["Time (s)"]
                    start_x = d_data.iloc[0]["Position X (mm)"]
                    start_times.append({"Domino ID": d_id, "Start Time (s)": start_time, "Initial X (mm)": start_x})
            
            start_df = pd.DataFrame(start_times)
            
            if len(start_df) > 1:
                start_df = start_df.sort_values("Initial X (mm)")
                start_df["Delta X (mm)"] = start_df["Initial X (mm)"].diff()
                start_df["Delta T (s)"] = start_df["Start Time (s)"].diff()
                start_df["Wave Velocity (mm/s)"] = start_df["Delta X (mm)"] / start_df["Delta T (s)"]
                
                fig_wave_v = px.line(start_df.dropna(), x="Start Time (s)", y="Wave Velocity (mm/s)", 
                                     markers=True, title="ドミノ間を伝わる波の伝播速度 (mm/s)")
                fig_wave_v.update_traces(line_color="red")
                st.plotly_chart(fig_wave_v, use_container_width=True)
                st.dataframe(start_df)
            else:
                st.warning("動き出しを検知できたドミノが少ないため、伝播速度が計算できませんでした。")

    else:
        st.error("マーカーが検出されませんでした。")
