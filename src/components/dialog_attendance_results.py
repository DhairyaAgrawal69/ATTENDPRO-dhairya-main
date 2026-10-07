import streamlit as st
import time
from typing import Any, Dict, List, Optional
from src.database.db import create_attendance

def _clean_primitive_value(v: Any):
    """Sanitizes an individual value, ensuring large buffers/arrays are dropped and scalars become native python primitives."""
    # Exclude raw byte buffers and images
    if isinstance(v, (bytes, bytearray)):
        return None, False
    # Exclude multi-dimensional numpy arrays (e.g. image buffers, face embeddings)
    if getattr(v, 'ndim', 0) > 0:
        return None, False
    # Exclude PIL Image instances
    if hasattr(v, 'size') and hasattr(v, 'mode') and hasattr(v, 'save'):
        return None, False

    # Convert numpy scalar types to native python primitives
    if hasattr(v, 'item') and callable(getattr(v, 'item')):
        try:
            v = v.item()
        except Exception:
            pass

    if isinstance(v, bool):
        return bool(v), True
    if isinstance(v, int):
        return int(v), True
    if isinstance(v, float):
        return float(v), True
    if isinstance(v, str) or v is None:
        return v, True

    return str(v), True


def _sanitize_records(records: Any) -> List[Dict[str, Any]]:
    """Convert any records representation (DataFrame, dict, list) to plain Python dicts with primitive values.
    Ensures no large numpy arrays, raw bytes, or image buffers are retained.
    """
    if records is None:
        return []

    # Handle pandas DataFrame if passed
    if hasattr(records, 'to_dict'):
        try:
            records = records.to_dict(orient='records')
        except Exception:
            records = []

    if isinstance(records, dict):
        if 'records' in records:
            records = records['records']
        else:
            records = [records]

    if not isinstance(records, list):
        return []

    clean_list: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        clean_item: Dict[str, Any] = {}
        for k, v in item.items():
            val, keep = _clean_primitive_value(v)
            if keep:
                clean_item[str(k)] = val
        clean_list.append(clean_item)

    return clean_list


def _sanitize_logs(logs: Any) -> List[Dict[str, Any]]:
    """Convert logs to plain Python dicts with primitive types for database insertion."""
    if logs is None:
        return []

    if hasattr(logs, 'to_dict'):
        try:
            logs = logs.to_dict(orient='records')
        except Exception:
            logs = []

    if not isinstance(logs, list):
        return []

    clean_logs: List[Dict[str, Any]] = []
    for item in logs:
        if not isinstance(item, dict):
            continue
        clean_log: Dict[str, Any] = {}
        for k, v in item.items():
            val, keep = _clean_primitive_value(v)
            if keep:
                clean_log[str(k)] = val
        clean_logs.append(clean_log)

    return clean_logs


def show_attendance_result(data: Any = None, logs: Any = None):
    """Renders attendance results using simple native Streamlit markdown without complex widgets or tables."""
    try:
        # Fallback to session_state if parameters are missing
        if data is None:
            if 'attendance_results_data' in st.session_state and isinstance(st.session_state.attendance_results_data, dict):
                data = st.session_state.attendance_results_data.get('records', [])
                if logs is None:
                    logs = st.session_state.attendance_results_data.get('logs', [])
            elif 'voice_attendance_results' in st.session_state and st.session_state.voice_attendance_results:
                data = st.session_state.voice_attendance_results[0]
                if logs is None:
                    logs = st.session_state.voice_attendance_results[1]

        clean_records = _sanitize_records(data)
        clean_logs = _sanitize_logs(logs)

        st.write('Please review attendance before confirming.')

        # Summary badge count
        if clean_records:
            total_students = len(clean_records)
            present_students = sum(1 for r in clean_records if "present" in str(r.get("Status", "")).lower())
            absent_students = total_students - present_students
            st.caption(f"**Total Enrolled:** {total_students} | **Present:** {present_students} | **Absent:** {absent_students}")

        # Render student list using simple native Streamlit markdown (no st.dataframe or st.table to prevent crashes)
        st.write("")
        try:
            if not clean_records:
                st.info("No enrolled students found.")
            else:
                for student in clean_records:
                    stu_id = student.get("ID", "-")
                    stu_name = student.get("Name", "Unknown")
                    status_raw = str(student.get("Status", ""))
                    is_present = "present" in status_raw.lower() or "✅" in status_raw
                    source = student.get("Source", "-")
                    source_str = f" *(via {source})*" if is_present and source and source != "-" else ""

                    if is_present:
                        st.markdown(f"✅ **[{stu_id}]** {stu_name} - Present{source_str}")
                    else:
                        st.markdown(f"❌ **[{stu_id}]** {stu_name} - Absent")
        except Exception as e:
            print(f"[ATTENDANCE_DIALOG] Roster error: {e}", flush=True)
            st.error(f"Roster error: {e}")

        st.write("")
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button('Discard', width='stretch', key='btn_dialog_discard'):
                _cleanup_attendance_session()
                st.rerun()

        with col2:
            if st.button('Confirm & Save', width='stretch', type='primary', key='btn_dialog_confirm'):
                try:
                    if not clean_logs:
                        st.warning('No attendance records to save.')
                    else:
                        with st.spinner('Saving attendance...'):
                            create_attendance(clean_logs)
                        st.toast("Attendance taken successfully!", icon="✅")
                        _cleanup_attendance_session()
                        time.sleep(0.5)
                        st.rerun()
                except Exception as e:
                    print(f"[ATTENDANCE_DIALOG] Sync failed: {e}", flush=True)
                    st.error(f'Sync failed: {e}')

    except Exception as e:
        print(f"[ATTENDANCE_DIALOG] Unexpected error in show_attendance_result: {e}", flush=True)
        st.error(f"Error displaying attendance results: {e}")
        if st.button("Close", key="btn_close_error_attendance"):
            _cleanup_attendance_session()
            st.rerun()


def _cleanup_attendance_session():
    """Defensively cleans up attendance and image buffers from session_state."""
    st.session_state.pop('attendance_results_data', None)
    st.session_state.voice_attendance_results = None
    st.session_state.attendance_images = []
    st.session_state.pop('_processed_upload_keys', None)
    st.session_state.pop('_last_cam_bytes', None)


@st.dialog("Attendance Reports")
def attendance_result_dialog(df=None, logs=None):
    try:
        show_attendance_result(df, logs)
    except Exception as e:
        print(f"[ATTENDANCE_DIALOG] Fatal dialog error: {e}", flush=True)
        st.error(f"Failed to open attendance report: {e}")
        if st.button("Close Dialog", key="btn_fatal_close_dialog"):
            _cleanup_attendance_session()
            st.rerun()