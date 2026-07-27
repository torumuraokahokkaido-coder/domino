import tempfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide") # グラフを広く表示するために追加

st.title("ドミノごとの運動解析アプリ 🎲")
st.write("動画内の赤い印を検出し、各ドミノの「時間と変位」のグラフを個別に表示します。")

st.sidebar.header("解析パラメータ設定")
min_area = st.sidebar.slider("マーカーの最小サイズ (ピクセル)", 10, 500, 30)
pixel_to_cm = st.sidebar.number_input("1ピクセルあたりの距離 (cm)", value=0.1, format="%.3f")

uploaded_file = st.file_uploader("ドミノの動画を選択してください (MP4 / MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if st.button("解析を開始する"):
        st.info("動画を解析中...")

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        records = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 赤色の抽出範囲
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])

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

            # X座標の小さい順（左から右）に並び替える
            valid_centers.sort(key=lambda pt: pt[0])
            current_time = frame_count / fps

            for idx, (cx, cy) in enumerate(valid_centers):
                records.append({
                    "Time (s)": current_time,
                    "Domino ID": f"Domino {idx + 1:02d}", # 番号を01, 02...と揃える
                    "Position X (cm)": cx * pixel_to_cm,
                    "Position Y (cm)": cy * pixel_to_cm,
                })

            frame_count += 1
        cap.release()

        if records:
            st.success("解析が完了しました！")
            df = pd.DataFrame(records)

            # --- 変位の計算 ---
            # 各ドミノの最初のフレームの位置を基準（0）とした「変位」を計算します。
            # 画像のY座標は「下」がプラスなので、直感的にするため上下を反転させています。
            df["Displacement X (cm)"] = df.groupby("Domino ID")["Position X (cm)"].transform(lambda x: x - x.iloc[0])
            df["Displacement Y (cm)"] = df.groupby("Domino ID")["Position Y (cm)"].transform(lambda x: -(x - x.iloc[0]))

            # --- 個別のグラフ描画 ---
            st.subheader("各ドミノの時間 - 変位グラフ")
            
            # iPadの横画面やPCで見やすいように2列で表示
            cols = st.columns(2)
            domino_ids = sorted(df["Domino ID"].unique())

            for i, domino_id in enumerate(domino_ids):
                group = df[df["Domino ID"] == domino_id]

                fig, ax = plt.subplots(figsize=(6, 4))
                # X方向とY方向の変位を重ねてプロット
                ax.plot(group["Time (s)"], group["Displacement X (cm)"], marker=".", label="X (Horizontal)", color="blue")
                ax.plot(group["Time (s)"], group["Displacement Y (cm)"], marker=".", label="Y (Vertical)", color="red")

                ax.set_title(f"{domino_id} Displacement")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Displacement (cm)")
                ax.legend()
                ax.grid(True)

                # 2列のカラムに交互にグラフを配置する
                cols[i % 2].pyplot(fig)

            # データダウンロード機能
            st.subheader("解析データ")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="解析データ(CSV)をダウンロード",
                data=csv,
                file_name="domino_20_analysis.csv",
                mime="text/csv",
            )
            st.dataframe(df)

        else:
            st.error("赤いマーカーが検出されませんでした。")
