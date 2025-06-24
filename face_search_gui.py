# Face Search GUI - 2025-06-20 (모든 기능, 기준 인물 추출 시간구간, UI 포함)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
import threading
#from sklearn.cluster import DBSCAN
import hdbscan
from PIL import Image, ImageTk
from datetime import datetime
from openpyxl import Workbook
import face_recognition

# 얼굴 품질 확인
def is_valid_face(face_crop):
    h, w = face_crop.shape[:2]
    if h < 100 or w < 100:
        return False
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var >= 40  # 기존은 30

# 탐지기 선택
def detect_faces_custom(image, detector_mode):
    if detector_mode == "hog":
        face_locations = face_recognition.face_locations(image, model="hog")
    elif detector_mode == "cnn":
        face_locations = face_recognition.face_locations(image, model="cnn")
    elif detector_mode == "hybrid":
        hog_faces = face_recognition.face_locations(image, model="hog")
        face_locations = []
        for (top, right, bottom, left) in hog_faces:
            face_img = image[top:bottom, left:right]
            if face_img.size == 0:
                continue
            cnn_faces = face_recognition.face_locations(face_img, model="cnn")
            if cnn_faces:
                face_locations.append((top, right, bottom, left))
    else:
        face_locations = []
    encodings = face_recognition.face_encodings(image, face_locations, num_jitters=5, model="large")
    return face_locations, encodings

# 상태/디버그 메시지 출력
def update_status(msg):
    status_var.set(msg)
    status_label.update()
    log_text.insert(tk.END, msg + "\n")
    log_text.see(tk.END)

def save_clustered_faces_debug(debug_dir, clustered_faces):
    for label, faces in clustered_faces.items():
        cluster_dir = os.path.join(debug_dir, f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)
        for idx, (_, thumb_img) in enumerate(faces):
            save_path = os.path.join(cluster_dir, f"face_{idx + 1}.jpg")
            thumb_img.save(save_path)

# 기준 인물 추출
def process_video(video_path, eps=0.4, min_samples=3, frame_sample_rate=5, start_sec=0, end_sec=None):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec else total_frames - 1
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    update_status(f"[DEBUG] 추출 범위: {start_frame}~{end_frame} (fps={fps}, 총프레임={total_frames})")

    face_encodings = []
    thumbnails = []
    debug_dir = os.path.join(os.path.dirname(video_path), "debug_faces")
    os.makedirs(debug_dir, exist_ok=True)

    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret, frame = video.read()
        if not ret:
            break
        if frame_idx % frame_sample_rate == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, encs = detect_faces_custom(rgb, detector_var.get())
            for i, (top, right, bottom, left) in enumerate(faces):
                if i >= len(encs): continue
                face_crop = frame[top:bottom, left:right]
                if not is_valid_face(face_crop):
                    continue
                face_encodings.append(encs[i])
                thumb = cv2.resize(face_crop, (100, 100))
                thumb_img = Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
                thumbnails.append(thumb_img)
                debug_path = os.path.join(debug_dir, f"frame{frame_idx}_face{len(face_encodings)}.jpg")
                cv2.imwrite(debug_path, face_crop)
        progress = ((frame_idx - start_frame) / (end_frame - start_frame + 1)) * 100
        progress_var.set(progress)
        progress_bar.update()
        update_status(f"{os.path.basename(video_path)}: {frame_idx}/{end_frame} 프레임")
        frame_idx += 1

    video.release()

    if not face_encodings:
        messagebox.showerror("오류", "얼굴을 인식하지 못했습니다.")
        return None

    #clustering = DBSCAN(eps=eps, min_samples=min_samples)
    #labels = clustering.fit_predict(face_encodings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(face_encodings)
    clustered = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clustered.setdefault(label, []).append((face_encodings[idx], thumbnails[idx]))
    unique_labels = set(labels)
    update_status(f"[DEBUG] 클러스터 개수: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
    # 🆕 인물별 폴더 저장
    save_clustered_faces_debug(debug_dir, clustered)
    return clustered

# 대표 얼굴 선택
def show_face_selection(clustered_faces, video_path):
    win = tk.Toplevel(root)
    win.title("기준 인물 선택")
    tk.Label(win, text="기준 인물로 사용할 얼굴을 클릭하세요").pack()
    def on_select(label_key):
        encodings = [e for e, _ in clustered_faces[label_key]]
        avg_encoding = np.mean(encodings, axis=0)
        save_path = os.path.join(os.path.dirname(video_path), "reference_encoding.npy")
        np.save(save_path, avg_encoding)
        messagebox.showinfo("완료", f"저장됨:\n{save_path}")
        update_status("기준 인물 인코딩 저장 완료")
        win.destroy()
    frame = tk.Frame(win)
    frame.pack()
    for label_key, faces in clustered_faces.items():
        _, img = faces[0]
        img_tk = ImageTk.PhotoImage(img)
        btn = tk.Button(frame, image=img_tk, command=lambda k=label_key: on_select(k))
        btn.image = img_tk
        btn.pack(side="left", padx=5)

# 기준 인물 추출 실행
def run_extract_reference():
    video_path = filedialog.askopenfilename(title="기준 인물 추출용 영상 선택")
    if not video_path:
        return
    try:
        rate = int(frame_sample_rate_entry.get())
        eps_val = float(eps_entry.get())
        min_s = int(min_samples_entry.get())
        start_sec = int(start_time_min_entry.get()) * 60 + int(start_time_sec_entry.get())
        end_sec = int(end_time_min_entry.get()) * 60 + int(end_time_sec_entry.get())
    except ValueError:
        messagebox.showerror("입력 오류", "입력값이 잘못되었습니다.")
        return
    def task():
        update_status(f"[DEBUG] 기준 인물 추출 시작 - {start_sec}s ~ {end_sec}s")
        clustered = process_video(video_path, eps=eps_val, min_samples=min_s, frame_sample_rate=rate,
                                  start_sec=start_sec, end_sec=end_sec)
        if clustered:
            show_face_selection(clustered, video_path)
    threading.Thread(target=task).start()

# 기준 인물 검색 실행
def run_search_by_reference():
    ref_path = filedialog.askopenfilename(title="기준 인물 인코딩 선택", filetypes=[("Numpy File", "*.npy")])
    video_dir = filedialog.askdirectory(title="검색할 영상 폴더 선택")
    if not ref_path or not video_dir:
        return
    try:
        threshold = float(match_threshold_entry.get())
        frame_gap = int(search_frame_rate_entry.get())
    except ValueError:
        messagebox.showerror("입력 오류", "검색 조건이 올바르지 않습니다.")
        return
    reference_encoding = np.load(ref_path)
    wb = Workbook()
    ws = wb.active
    ws.append(["파일명", "등장 시각"])
    def task():
        video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        for idx, fname in enumerate(video_files):
            update_status(f"🔍 {idx+1}/{len(video_files)}: {fname} 검색 시작")
            progress_var.set((idx / len(video_files)) * 100)
            progress_bar.update()

            fpath = os.path.join(video_dir, fname)
            cap = cv2.VideoCapture(fpath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            hit_times = []
            matched = False

            while current_frame < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                update_status(f"{fname}: {current_frame}/{total_frames} 프레임 진행 중")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locs, encs = detect_faces_custom(rgb, detector_var.get())
                for enc in encs:
                    dist = np.linalg.norm(reference_encoding - enc)
                    if dist < threshold:
                        sec = current_frame / fps
                        hit_times.append(f"{int(sec//60):02}:{int(sec%60):02}")
                        matched = True
                        break
                current_frame += frame_gap
                
            cap.release()
            if hit_times:
                ws.append([fname, ", ".join(hit_times)])

             # 결과 메시지
            if matched:
                update_status(f"✅ {idx+1}/{len(video_files)}: {fname} ▶ 발견 ({', '.join(hit_times)})")
            else:
                update_status(f"⏭️ {idx+1}/{len(video_files)}: {fname} ▶ 패스")
    
            progress_ratio = (idx + 1) / len(video_files)
            progress_percent = int(progress_ratio * 100)
            update_status(f"[{idx+1}/{len(video_files)}] '{fname}' 검색 중... 진행률: {progress_percent}%")
            progress_var.set(progress_percent)
            progress_bar.update()
        save_path = os.path.join(video_dir, f"search_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        wb.save(save_path)
        messagebox.showinfo("완료", f"검색 결과 저장됨:\n{save_path}")
        update_status("검색 완료")
    threading.Thread(target=task).start()

# UI 구성
root = tk.Tk()
root.title("Face Search GUI (face_recognition 기반 전체 기능)")

# 상단 버튼
btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)
tk.Button(btn_frame, text="기준 인물 추출", command=run_extract_reference).pack(side="left", padx=10)
tk.Button(btn_frame, text="기준 인물로 영상 검색", command=run_search_by_reference).pack(side="left", padx=10)

# 탐지기 설정
detector_var = tk.StringVar(value="hybrid")
detector_frame = tk.Frame(root)
detector_frame.pack()
tk.Label(detector_frame, text="탐지기:").pack(side="left")
for mode in ["hog", "cnn", "hybrid"]:
    tk.Radiobutton(detector_frame, text=mode.upper(), variable=detector_var, value=mode).pack(side="left")

# 검색/추출 설정
param_frame = tk.Frame(root)
param_frame.pack(pady=5)
tk.Label(param_frame, text="기준 추출 프레임 간격:").pack(side="left")
frame_sample_rate_entry = tk.Entry(param_frame, width=5)
frame_sample_rate_entry.insert(0, "5")
frame_sample_rate_entry.pack(side="left")
tk.Label(param_frame, text="검색 프레임 간격:").pack(side="left")
search_frame_rate_entry = tk.Entry(param_frame, width=5)
search_frame_rate_entry.insert(0, "5")
search_frame_rate_entry.pack(side="left")
tk.Label(param_frame, text="매칭 임계값:").pack(side="left")
match_threshold_entry = tk.Entry(param_frame, width=5)
match_threshold_entry.insert(0, "0.45")
match_threshold_entry.pack(side="left")

# DBSCAN 설정
cluster_frame = tk.Frame(root)
cluster_frame.pack(pady=5)
tk.Label(cluster_frame, text="DBSCAN eps:").pack(side="left")
eps_entry = tk.Entry(cluster_frame, width=5)
eps_entry.insert(0, "0.4")
eps_entry.pack(side="left")
tk.Label(cluster_frame, text="min_samples:").pack(side="left")
min_samples_entry = tk.Entry(cluster_frame, width=5)
min_samples_entry.insert(0, "3")
min_samples_entry.pack(side="left")

# 추출 시간 설정
time_frame = tk.Frame(root)
time_frame.pack(pady=5)
tk.Label(time_frame, text="추출 시작 mm:ss").pack(side="left")
start_time_min_entry = tk.Entry(time_frame, width=3)
start_time_min_entry.insert(0, "0")
start_time_min_entry.pack(side="left")
tk.Label(time_frame, text=":").pack(side="left")
start_time_sec_entry = tk.Entry(time_frame, width=3)
start_time_sec_entry.insert(0, "0")
start_time_sec_entry.pack(side="left")
tk.Label(time_frame, text="~ 끝 mm:ss").pack(side="left")
end_time_min_entry = tk.Entry(time_frame, width=3)
end_time_min_entry.insert(0, "999")
end_time_min_entry.pack(side="left")
tk.Label(time_frame, text=":").pack(side="left")
end_time_sec_entry = tk.Entry(time_frame, width=3)
end_time_sec_entry.insert(0, "59")
end_time_sec_entry.pack(side="left")

# 진행률 & 로그창
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(fill="x", padx=10, pady=5)
status_var = tk.StringVar()
status_label = tk.Label(root, textvariable=status_var)
status_label.pack()
log_text = tk.Text(root, height=10)
log_text.pack(fill="both", expand=True, padx=10, pady=5)

update_status("✅ 프로그램 준비 완료")

root.mainloop()
