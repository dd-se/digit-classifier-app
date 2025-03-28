from resources import *

col, _ = st.columns([20, 34])
with col:
    canvas_result = st_canvas(
        stroke_width=10,
        stroke_color="#000000",
        background_color="#FFFFFF",
        display_toolbar=True,
        height=252,
        width=252,
        fill_color="#000000",
    )
    run = st.button("Prediktera", use_container_width=True)

if run and canvas_result.image_data is not None:
    gray = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_RGBA2GRAY)
    white_pixel_ratio = np.sum(gray == 255) / gray.size

    if white_pixel_ratio > 0.99:
        st.stop()

    gray = 255 - gray
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    gray = gray[y : y + h, x : x + w]
    padding_size = int(0.34 * max(w, h))
    gray = cv2.copyMakeBorder(gray, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
    processed_img = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    pred = st.session_state.model.predict([processed_img.ravel()])[0]
    st.subheader(f"Prediktion: {pred}")

    if hasattr(st.session_state.model, "predict_proba"):
        proba = st.session_state.model.predict_proba([processed_img.ravel()])[0]
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        ax.set_title("Sannolikheter")
        ax.set_yticks(np.arange(10))
        ax.barh(np.arange(10), proba)
        st.pyplot(fig)
