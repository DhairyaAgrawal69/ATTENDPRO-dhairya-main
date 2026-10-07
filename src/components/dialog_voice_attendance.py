import streamlit as st

from src.pipelines.voice_pipeline import process_bulk_audio

from src.database.config import supabase

import pandas as pd

from src.components.dialog_attendance_results import show_attendance_result

from datetime import datetime

@st.dialog('Voice Attendance')
def voice_attendance_dialog(selected_subject_id):
    st.write('Record audio of students saying I am present. Then AI will recognize the students')

    audio_data = None

    audio_data = st.audio_input("Record classroom audio")

    if st.button('Analyze Audio', width='stretch', type='primary'):
        with st.spinner('Prcessing Audio data'):
            enrolled_res = supabase.table('subject_students').select("*, students(*)").eq('subject_id',selected_subject_id ).execute()
            enrolled_students = enrolled_res.data

            if not enrolled_students:
                st.warning('No students enrolled in this course')
                return
            candidates_dict = {
                s['students']['student_id'] : s['students']['voice_embedding'] 
                for s in enrolled_students if s['students'].get('voice_embedding')
            }

            if not candidates_dict:
                st.error('No enrolled students have voice profiles registerd')
                return
            
            audio_bytes = audio_data.read()

            detected_scores = process_bulk_audio(audio_bytes, candidates_dict)

            results = []
            attendance_to_log = []
            current_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

            for node in enrolled_students:
                student = node.get('students', {})
                raw_sid = student.get('student_id')
                try:
                    student_id_val = int(raw_sid)
                except (TypeError, ValueError):
                    student_id_val = raw_sid

                raw_score = detected_scores.get(raw_sid, 0.0)
                try:
                    score_val = float(raw_score)
                except (TypeError, ValueError):
                    score_val = 0.0
                is_present = bool(score_val > 0)

                results.append({
                    "Name": str(student.get('name', 'Unknown')),
                    "ID": student_id_val,
                    "Source": f"{score_val:.2f}" if is_present else "-",
                    "Status": "✅ Present" if is_present else "❌ Absent"
                })

                try:
                    sub_id_val = int(selected_subject_id)
                except (TypeError, ValueError):
                    sub_id_val = selected_subject_id

                attendance_to_log.append({
                    'student_id': student_id_val,
                    'subject_id': sub_id_val,
                    'timestamp': str(current_timestamp),
                    'is_present': is_present
                })
            st.session_state.voice_attendance_results = (results, attendance_to_log)

    if st.session_state.get('voice_attendance_results'):
        st.divider()
        results_list, logs = st.session_state.voice_attendance_results
        show_attendance_result(results_list, logs)