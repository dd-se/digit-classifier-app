from resources import *

if "model" not in st.session_state:
    st.session_state.model = None
if "session_model" not in st.session_state:
    st.session_state.session_model = None
if "training_models" not in st.session_state:
    st.session_state.training_models = {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "STD-ETC": make_pipeline(StandardScaler(), ExtraTreesClassifier(n_estimators=200, n_jobs=-1)),
        "PCA-KNN": make_pipeline(PCA(n_components=0.95), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "PCA-SVC-0.95-can-proba": make_pipeline(PCA(n_components=0.95), SVC(kernel="rbf", random_state=42, probability=True)),
        "MLPClassifier": MLPClassifier(max_iter=1000),
    }

if "dragon_chosen" not in st.session_state:
    st.session_state.dragon_chosen = None


st.session_state.dragon_chosen = st.radio("Skapa Model:", st.session_state.training_models.keys(), horizontal=True)

col, col_2, col_3, col_4 = st.columns(4)

with col:
    validate_your_dragon = st.button("Validera modellen")
if validate_your_dragon:
    with st.spinner("### Validering pågår...", show_time=True):
        st.write(f"Model: {st.session_state.dragon_chosen}")
        st.session_state.session_model: ExtraTreesClassifier = st.session_state.training_models[st.session_state.dragon_chosen]  # type: ignore
        st.write("Tränar på träningsdatan...")
        st.session_state.session_model.fit(X_train, y_train)
        st.write("Validerar på validerings datan...")
        y_pred = st.session_state.session_model.predict(X_val)
        score = accuracy_score(y_val, y_pred) * 100
        st.subheader(f"Accuracy: {score:.2f}%")
        st.session_state.val_accuracy = score
        with st.expander("Visa detaljerad rapport"):
            st.dataframe(classification_report(y_val, y_pred, output_dict=True))
            display_confusion_matrix(y_val, y_pred, st.session_state.dragon_chosen)
        st.success("Klart!")

with col_2:
    test_your_dragon = st.button("Testa modellen")
if test_your_dragon:
    with st.spinner("### Test pågår...", show_time=True):
        if st.session_state.session_model:
            st.write(f"Model: {st.session_state.session_model.__class__.__name__}")
            st.write("Tränar på tränings och validerings datan...")
            st.session_state.session_model.fit(X_train_val, y_train_val)
            st.write("Validerar på test datan...")
            y_pred = st.session_state.session_model.predict(X_test)
            score = accuracy_score(y_test, y_pred) * 100
            st.subheader(f"Validation Accuracy: {st.session_state.val_accuracy:.2f}%")
            st.subheader(f"Test Accuracy: {score:.2f}%")
            st.success("Klart!")
        else:
            st.write("Vänligen validera modellen först.")

with col_3:
    deploy_your_dragon = st.button("Spara modellen i minnet")
if deploy_your_dragon:
    with st.spinner("### Sparar modellen...", show_time=True):
        st.session_state.session_model: ExtraTreesClassifier = st.session_state.training_models[st.session_state.dragon_chosen]
        st.session_state.session_model.fit(X, y)
        st.session_state.models.append(st.session_state.session_model)
        st.success("Klart!")

with col_4:
    save_your_dragon = st.button("Träna och spara modellen på disk")
if save_your_dragon:
    with st.spinner("### Sparar modellen på disk...", show_time=True):
        st.write(f"Model: {st.session_state.dragon_chosen}")
        st.session_state.session_model: ExtraTreesClassifier = st.session_state.training_models[st.session_state.dragon_chosen]
        st.session_state.session_model.fit(X, y)
        joblib.dump(
            st.session_state.session_model,
            Path("resources/models") / f"{st.session_state.dragon_chosen}.gz",
            compress=("gzip", 3),
        )
        st.session_state.models.append(st.session_state.session_model)
        st.success("Klart!")
