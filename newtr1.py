import streamlit as st
import face_recognition
import pandas as pd
import numpy as np
import mediapipe as mp
import os
import pickle
import io
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageEnhance
import cv2

# ------------------------------
# โฟลเดอร์พื้นฐาน
# ------------------------------
os.makedirs("student_database", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("attendance_logs", exist_ok=True)
os.makedirs("attendance_realtime", exist_ok=True)

st.set_page_config(page_title="FaceUp เช็คชื่อแบบไม่ต้องพูด", layout="centered")

# ------------------------------
# CSS
# ------------------------------
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #0059b3 !important;
        font-family: 'Kanit', 'Sarabun', Tahoma, Arial, sans-serif;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    div.stButton > button {
        background-color: #0059b3 !important;
        color: white !important;
        border-radius: 5px;
        width: 100%;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #004080 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Mediapipe
# ------------------------------
mp_face = mp.solutions.face_detection

def detect_faces_mediapipe_full(image_array, min_confidence=0.2):
    h, w, _ = image_array.shape
    face_locations = []
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as fd:
        results = fd.process(image_array)
        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                x, y, bw, bh = box.xmin, box.ymin, box.width, box.height
                left, top = int(x * w), int(y * h)
                right, bottom = int((x + bw) * w), int((y + bh) * h)
                face_locations.append((top, right, bottom, left))
    return face_locations

def detect_faces_patch(image_array, grid_size=3):
    h, w, _ = image_array.shape
    patches = []
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            top = i * h // grid_size
            bottom = (i + 1) * h // grid_size
            left = j * w // grid_size
            right = (j + 1) * w // grid_size
            patches.append(image_array[top:bottom, left:right])
            positions.append((top, left))
    all_faces = []
    for patch, (t_offset, l_offset) in zip(patches, positions):
        # ปรับ contrast/brightness เล็กน้อยตอน detect
        patch_pil = Image.fromarray(patch)
        patch_pil = ImageEnhance.Contrast(patch_pil).enhance(1.1)
        patch_pil = ImageEnhance.Brightness(patch_pil).enhance(1.05)
        patch_array = np.array(patch_pil)
        patch_faces = detect_faces_mediapipe_full(patch_array)
        for (top, right, bottom, left) in patch_faces:
            all_faces.append((top+t_offset, right+l_offset, bottom+t_offset, left+l_offset))
    return all_faces

# ------------------------------
# Helper
# ------------------------------
def sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)

def load_encodings():
    path = "data/encodings.pkl"
    if not os.path.exists(path): return {}
    with open(path, "rb") as f: return pickle.load(f)

def save_encodings(enc_dict):
    with open("data/encodings.pkl", "wb") as f:
        pickle.dump(enc_dict, f)

# ------------------------------
# Save student encoding
# ------------------------------
def save_encoding(name, grade, number, image_bytes):
    student_info = {"name": name, "grade": grade, "number": number}
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((1200, 1200))
    image_array = np.array(img)

    # Detect full image
    boxes = detect_faces_mediapipe_full(image_array, min_confidence=0.2)
    # fallback ถ้า Mediapipe ไม่เจอ
    if not boxes:
        boxes = face_recognition.face_locations(image_array)

    encodings = face_recognition.face_encodings(image_array, boxes)
    if not encodings:
        st.error("❌ ไม่พบใบหน้าในภาพ")
        return False

    enc_dict = load_encodings()
    key = f"{name}|{grade}|{number}"
    if key not in enc_dict:
        enc_dict[key] = {"encodings": [], "info": student_info}

    enc_dict[key]["encodings"].extend(encodings)
    save_encodings(enc_dict)

    safe_name = sanitize_filename(name)
    safe_grade = sanitize_filename(grade)
    safe_number = sanitize_filename(number)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    img.save(f"student_database/{safe_name}_{safe_grade}_{safe_number}_{timestamp}.jpg")

    st.success(f"✅ บันทึกนักเรียน '{name} (ชั้น {grade} เลขที่ {number})' สำเร็จ!")
    return True

# ------------------------------
# Attendance check with Voting
# ------------------------------
def mark_attendance_realtime(image_bytes, subject, tolerance=0.45, vote_threshold=0.5):
    enc_dict = load_encodings()
    if not enc_dict:
        return [], None, None

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((1200, 1200))
    image_array = np.array(img)

    boxes = detect_faces_patch(image_array, grid_size=3)
    face_encs = face_recognition.face_encodings(image_array, boxes)

    found, seen_ids = [], set()
    draw = ImageDraw.Draw(img)

    for (top, right, bottom, left), face_enc in zip(boxes, face_encs):
        best_match, best_score = None, 0.0
        for key, data in enc_dict.items():
            distances = face_recognition.face_distance(data["encodings"], face_enc)
            if len(distances)==0: continue
            matches = [d < tolerance for d in distances]
            score = sum(matches)/len(matches)
            if score > best_score:
                best_score = score
                best_match = data["info"]
        if best_match and best_score >= vote_threshold:
            unique_id = f"{best_match['name']}|{best_match['grade']}|{best_match['number']}"
            if unique_id not in seen_ids:
                found.append(best_match)
                seen_ids.add(unique_id)
            draw.rectangle([left, top, right, bottom], outline="green", width=3)
            draw.text((left, top - 10), best_match['name'], fill="green")
        else:
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            draw.text((left, top - 10), "Unknown", fill="red")

    # ------------------------------
    # Save logs
    # ------------------------------
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now()
    df_new = pd.DataFrame([{
        "ชื่อ": s["name"], "ชั้น": s["grade"], "เลขที่": s["number"],
        "วิชา": subject, "last_seen": time_now.strftime("%H:%M:%S")
    } for s in found])

    log_path = f"attendance_logs/{date_str}.csv"
    if os.path.exists(log_path):
        df_old = pd.read_csv(log_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.groupby(["ชื่อ","ชั้น","เลขที่","วิชา"], as_index=False).agg({"last_seen":"last"})
    else:
        df = df_new
    if not df.empty:
        df.to_csv(log_path, index=False, encoding="utf-8-sig")

    realtime_path = f"attendance_realtime/{date_str}.csv"
    if os.path.exists(realtime_path):
        df_r_old = pd.read_csv(realtime_path)
        df_r = pd.concat([df_r_old, df_new], ignore_index=True)
        df_r = df_r.groupby(["ชื่อ","ชั้น","เลขที่","วิชา"], as_index=False).agg({"last_seen":"last"})
    else:
        df_r = df_new
    if not df_r.empty:
        df_r.to_csv(realtime_path, index=False, encoding="utf-8-sig")

    return found, date_str, img

# ------------------------------
# Sidebar + Menu
# ------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://cdn.discordapp.com/attachments/1381571974421151764/1410269362014912644/Shutter_checkdwd.png" width="80">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h2 style='text-align:center;color:#fff;'>Shutter'n Check</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:12px;'>ระบบเช็คชื่อด้วยใบหน้า<br>พัฒนาโดย: SKRz</p>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("📂 เมนู", ["ลงทะเบียนนักเรียน", "เช็คชื่อ (อัปโหลด)", "เช็คชื่อ (เรียลไทม์)", "สถิติ"], index=0)

# ------------------------------
# Register
# ------------------------------
if menu == "ลงทะเบียนนักเรียน":
    st.markdown("## ลงทะเบียนนักเรียน")
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("ชื่อ-นามสกุล")
            grade = st.text_input("ชั้น")
            number = st.text_input("เลขที่")
        with col2:
            uploaded_file = st.file_uploader("📂 อัปโหลดภาพใบหน้า", type=["jpg","png"])
            if uploaded_file:
                st.image(uploaded_file.getvalue(), caption="ภาพที่อัปโหลด", use_column_width=True)
        submitted = st.form_submit_button("💾 บันทึกนักเรียน")
        if submitted and name and grade and number and uploaded_file:
            save_encoding(name, grade, number, uploaded_file.getvalue())

# ------------------------------
# Check Attendance (Upload)
# ------------------------------
elif menu == "เช็คชื่อ (อัปโหลด)":
    st.header("✅ เช็คชื่อรายวิชา (อัปโหลด)")
    subject = st.selectbox("📚 เลือกวิชา", ["คณิตศาสตร์","วิทยาศาสตร์","ภาษาอังกฤษ","ภาษาไทย","สังคม","ศิลปะ","คอมพิวเตอร์"])
    uploaded_file = st.file_uploader("🖼️ อัปโหลดรูปห้องเรียน", type=["jpg","png"])
    if uploaded_file:
        students, date_str, img = mark_attendance_realtime(uploaded_file.getvalue(), subject)
        st.image(img, caption="ผลการตรวจจับ", use_column_width=True)
        if students:
            st.success(f"เจอนักเรียน {len(students)} คน")
            st.table(pd.DataFrame(students))

# ------------------------------
# Check Attendance (Realtime)
# ------------------------------
elif menu == "เช็คชื่อ (เรียลไทม์)":
    st.markdown("## 📡 เช็คชื่อแบบเรียลไทม์")
    st.info("ระบบจะอัปเดตทุก **30 วินาที** อัตโนมัติเมื่อกล้องทำงาน")

    subject = st.selectbox(
        "📚 เลือกวิชา",
        ["คณิตศาสตร์", "วิทยาศาสตร์", "ภาษาอังกฤษ", "ภาษาไทย", "สังคม", "ศิลปะ", "คอมพิวเตอร์"]
    )

    col_btn1, col_btn2 = st.columns([1,1])
    with col_btn1:
        run_realtime = st.toggle("▶️ เริ่มตรวจสอบ Realtime")
    with col_btn2:
        stop_btn = st.button("⏹️ หยุดกล้อง")

    FRAME_INTERVAL = 30  # วินาที

    if run_realtime and not stop_btn:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ ไม่สามารถเปิดกล้องได้")
        else:
            st.markdown(
                """
                <div style="padding:10px; background:#e6f2ff; border-radius:10px; text-align:center;">
                    <b style="color:#0059b3;">📡 กำลังทำงาน...</b><br>
                    ระบบกำลังตรวจจับใบหน้าอยู่
                </div>
                """,
                unsafe_allow_html=True
            )

            placeholder_img = st.empty()
            placeholder_table = st.empty()
            placeholder_text = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ อ่านภาพจากกล้องไม่ได้")
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                buf = io.BytesIO()
                img_pil.save(buf, format="JPEG")

                students, date_str, img_result = mark_attendance_realtime(buf.getvalue(), subject)

                with placeholder_img.container():
                    st.image(img_result, caption=f"📸 ตรวจล่าสุด {datetime.now().strftime('%H:%M:%S')}", use_column_width=True)

                with placeholder_text.container():
                    if students:
                        st.success(f"✅ พบ {len(students)} คนในห้อง")
                    else:
                        st.warning("⚠️ ไม่พบนักเรียนที่รู้จักในรอบนี้")

                with placeholder_table.container():
                    if students:
                        st.markdown("### 👥 รายชื่อนักเรียนที่พบ")
                        st.dataframe(pd.DataFrame(students), use_container_width=True)

                time.sleep(FRAME_INTERVAL)

        cap.release()
    else:
        st.markdown(
            """
            <div style="padding:10px; background:#ffe6e6; border-radius:10px; text-align:center;">
                <b style="color:#cc0000;">⏹️ กล้องยังไม่ทำงาน</b><br>
                กดปุ่ม ▶️ เพื่อเริ่มตรวจสอบแบบ Realtime
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------
# Statistics
# ------------------------------
elif menu == "สถิติ":
    st.header("📊 รายงานการมาเรียน")
    mode = st.radio("เลือกรูปแบบสถิติ", ["📂 จากการอัปโหลดรูป", "🎥 Realtime กล้อง"], horizontal=True)

    if mode == "📂 จากการอัปโหลดรูป":
        log_files = sorted([f for f in os.listdir("attendance_logs") if f.endswith(".csv")])
        if not log_files:
            st.info("ยังไม่มีข้อมูลการมาเรียน")
        else:
            selected_file = st.selectbox("เลือกวันที่", log_files)
            df = pd.read_csv(os.path.join("attendance_logs", selected_file))
            if 'วิชา' not in df.columns: df['วิชา']="ไม่ระบุ"
            subjects = df['วิชา'].unique()
            selected_subject = st.selectbox("เลือกวิชา", subjects)
            df_filtered = df[df['วิชา']==selected_subject]
            st.markdown(f"### รายงานวันที่ {selected_file.replace('.csv','')} (วิชา: {selected_subject})")
            st.write(f"จำนวนคนที่มาเรียน: {df_filtered['ชื่อ'].nunique()} คน")
            view_type = st.radio("รูปแบบการแสดงผล", ["ตาราง", "กราฟแท่ง"], horizontal=True)
            summary = df_filtered.groupby(["ชื่อ","ชั้น","เลขที่"]).size().reset_index(name="จำนวนครั้ง")
            if view_type=="ตาราง":
                st.markdown("### 📝 สรุปจำนวนครั้ง")
                st.table(summary)
            else:
                st.markdown("### 📊 กราฟแท่งแสดงจำนวนครั้ง")
                chart_data = summary.set_index("ชื่อ")["จำนวนครั้ง"]
                st.bar_chart(chart_data)
    elif mode=="🎥 Realtime กล้อง":
        log_files = sorted([f for f in os.listdir("attendance_realtime") if f.endswith(".csv")])
        if not log_files:
            st.info("ยังไม่มีข้อมูล Realtime")
        else:
            selected_file = st.selectbox("เลือกวันที่ (Realtime)", log_files)
            df = pd.read_csv(os.path.join("attendance_realtime", selected_file))
            if 'วิชา' not in df.columns: df['วิชา']="ไม่ระบุ"
            if 'last_seen' not in df.columns: df['last_seen']="--"
            subjects = df['วิชา'].unique()
            selected_subject = st.selectbox("เลือกวิชา", subjects)
            df_filtered = df[df['วิชา']==selected_subject]
            now = datetime.now()
            def check_status(t):
                try:
                    last_time = datetime.strptime(t, "%H:%M:%S")
                    return "✅ อยู่" if (now - last_time).seconds <= 300 else "❌ ไม่อยู่"
                except:
                    return "❌ ไม่อยู่"
            df_filtered["status"] = df_filtered["last_seen"].apply(check_status)
            st.markdown(f"### Realtime รายงานวันที่ {selected_file.replace('.csv','')} (วิชา: {selected_subject})")
            st.write(f"นักเรียนทั้งหมด: {df_filtered['ชื่อ'].nunique()} คน")
            st.table(df_filtered[["ชื่อ","ชั้น","เลขที่","last_seen","status"]])
