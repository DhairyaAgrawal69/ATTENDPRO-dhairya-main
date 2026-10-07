import streamlit as st
import numpy as np
from PIL import Image, ImageOps


@st.dialog("Capture or upload photos")
def add_photos_dialog():
    st.write('Add classroom photos to scan for attendance')

    if 'attendance_images' not in st.session_state:
        st.session_state.attendance_images = []

    if 'photo_tab' not in st.session_state:
        st.session_state.photo_tab = 'camera'

    t1, t2 = st.columns(2)

    with t1:
        type_camera = "primary" if st.session_state.photo_tab == 'camera' else 'tertiary'
        if st.button('Camera', type=type_camera, width='stretch'):
            st.session_state.photo_tab = 'camera'

    with t2:
        type_upload = "primary" if st.session_state.photo_tab == 'upload' else 'tertiary'
        if st.button('Upload photos', type=type_upload, width='stretch'):
            st.session_state.photo_tab = 'upload'

    if st.session_state.photo_tab == 'camera':
        cam_photo = st.camera_input('Take group photos', key='dialog_cam')
        if cam_photo is not None:
            cam_bytes = cam_photo.getvalue()
            if cam_bytes != st.session_state.get('_last_cam_bytes'):
                st.session_state._last_cam_bytes = cam_bytes
                try:
                    img = Image.open(cam_photo)
                    img = ImageOps.exif_transpose(img).convert('RGB')
                    arr = np.ascontiguousarray(np.array(img, dtype=np.uint8), dtype=np.uint8)
                    st.session_state.attendance_images.append(arr)
                    st.toast('Photo Captured')
                except Exception as e:
                    st.error(f"Failed to process captured photo: {e}")

    if st.session_state.photo_tab == 'upload':
        uploaded_files = st.file_uploader(
            'Choose image files',
            type=['jpg', 'png', 'jpeg'],
            accept_multiple_files=True,
            key='dialog_upload'
        )

        if uploaded_files:
            if '_processed_upload_keys' not in st.session_state:
                st.session_state._processed_upload_keys = set()

            new_count = 0
            for f in uploaded_files:
                file_sig = f"{f.name}_{f.size}"
                if file_sig not in st.session_state._processed_upload_keys:
                    st.session_state._processed_upload_keys.add(file_sig)
                    try:
                        if hasattr(f, 'seek'):
                            f.seek(0)
                        img = Image.open(f)
                        img = ImageOps.exif_transpose(img).convert('RGB')
                        arr = np.ascontiguousarray(np.array(img, dtype=np.uint8), dtype=np.uint8)
                        st.session_state.attendance_images.append(arr)
                        new_count += 1
                    except Exception as e:
                        st.error(f"Failed to load {f.name}: {e}")

            if new_count > 0:
                st.toast(f'Added {new_count} photo(s)')

    num_photos = len(st.session_state.attendance_images)
    st.info(f"{num_photos} photo(s) currently staged for attendance scanning.")

    st.divider()
    if st.button('Done', type='primary', width='stretch'):
        st.session_state.pop('_processed_upload_keys', None)
        st.session_state.pop('_last_cam_bytes', None)
        st.rerun()