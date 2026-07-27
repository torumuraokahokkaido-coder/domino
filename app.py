import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

# Webアプリのタイトル設定
st.title("ドミノ倒し 運動解析アプリ 🎲")
st.write("ドミノの動画（赤い印つき）をアップロードすると、速度や位置を解析します。")

# 動画ファイルのアップロードボタン
uploaded_file = st.file_uploader("ドミノの動画を選択してください (MP4 / MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    # アップロードされた動画を一時ファイルとして保存
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # 解析実行ボタン
    if st.button("解析を開始する"):
        st.info("動画を解析中...")
        
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        time_list = []
        x_positions = []
        frame_count = 0
        pixel_to_cm_ratio = 0.1  # 縮尺設定

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 赤色抽出
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_contour) > 50:
                    M = cv2.moments(max_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        time_list.append(frame_count / fps)
                        x_positions.append(cx * pixel_to_cm_ratio)

            frame_count += 1

        cap.release()

        if time_list:
            st.success("解析が完了しました！")
            df = pd.DataFrame({'Time (s)': time_list, 'Position X (cm)': x_positions})

            # グラフの描画
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['Time (s)'], df['Position X (cm)'], marker='o', color='red', linestyle='-')
            ax.set_title('ドミノの伝播: 位置 - 時間グラフ')
            ax.set_xlabel('時間 (秒)')
            ax.set_ylabel('X座標 (cm)')
            ax.grid(True)

            # Streamlit上にグラフと表を表示
            st.pyplot(fig)
            st.dataframe(df)
        else:
            st.error("赤いマーカーが検出されませんでした。")
