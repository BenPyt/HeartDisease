import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Dự đoán bệnh tim AI",
    page_icon="🫀",
    layout="wide"
)

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        color: white; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: #f0f2f6; font-size: 1.1rem;
    }
    .result-container {
        padding: 2rem; border-radius: 15px; margin-top: 1rem; text-align: center;
        font-size: 1.2rem; font-weight: bold; box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .positive-result { background: linear-gradient(135deg, #ff6b6b, #ee5a6f); color: white; }
    .negative-result { background: linear-gradient(135deg, #4ecdc4, #44a08d); color: white; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 0.7rem 2rem; border-radius: 25px;
        font-size: 1.1rem; font-weight: bold; box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease; width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# --- TẢI DỮ LIỆU VÀ MÔ HÌNH ---
@st.cache_resource
def load_data_and_model():
    """Tải dataset và mô hình pipeline đã được huấn luyện."""
    try:
        df = pd.read_csv("heart_disease_dataset_400.csv")
        with open("random_forest_model.pkl", "rb") as file:
            model = pickle.load(file)
        return df, model
    except FileNotFoundError:
        st.error("LỖI: Vui lòng đảm bảo các file 'random_forest_model.pkl' và 'heart_disease_dataset_400.csv' tồn tại!")
        return None, None

df, model = load_data_and_model()


# --- ĐỊNH NGHĨA CÁC TRANG ---

def show_overview_page(df_data):
    """Hiển thị trang tổng quan dữ liệu."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Tổng quan Dataset</h1>
        <p>Thông tin chi tiết về bộ dữ liệu được sử dụng để huấn luyện mô hình.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df_data is not None:
        st.write(f"**Số dòng:** {df_data.shape[0]}", f"**Số cột:** {df_data.shape[1]}")
        
        st.markdown("#### 1️⃣ Một vài dòng đầu tiên:")
        st.dataframe(df_data.head())

        st.markdown("#### 2️⃣ Mô tả các cột trong dataset:")
        with st.expander("📋 Chi tiết các biến"):
            st.markdown("""
            | Cột        | Ý nghĩa                                                                 |
            |------------|-------------------------------------------------------------------------|
            | `age`      | Tuổi của bệnh nhân                                                      |
            | `sex`      | Giới tính (1 = Nam, 0 = Nữ)                                             |
            | `cp`       | Loại đau ngực (0–3)                                                     |
            | `trestbps` | Huyết áp nghỉ (mm Hg)                                                   |
            | `chol`     | Mức cholesterol trong máu (mg/dl)                                       |
            | `fbs`      | Đường huyết lúc đói > 120 mg/dl (1 = Có, 0 = Không)                      |
            | `restecg`  | Kết quả điện tâm đồ khi nghỉ (0–2)                                      |
            | `thalach`  | Nhịp tim tối đa đạt được                                                 |
            | `exang`    | Đau ngực khi gắng sức (1 = Có, 0 = Không)                               |
            | `oldpeak`  | ST depression                                                           |
            | `slope`    | Độ dốc đoạn ST (0 = đi ngang, 1 = lên, 2 = xuống)                       |
            | `ca`       | Số mạch máu chính được phát hiện qua X-quang (0–4)                      |
            | `thal`     | Thalassemia (1 = bình thường, 2 = lỗi cố định, 3 = lỗi có thể hồi phục)|
            | `target`   | Kết quả (1 = có bệnh tim, 0 = không)                                    |
            """, unsafe_allow_html=True)

        st.markdown("#### 3️⃣ Mô tả thống kê cơ bản:")
        st.dataframe(df_data.describe())
        
        st.markdown("#### 4️⃣ Tỷ lệ người mắc bệnh:")
        disease_rate = df_data['target'].value_counts(normalize=True).rename({0: "Không mắc bệnh", 1: "Mắc bệnh"}) * 100
        st.bar_chart(disease_rate)

    else:
        st.warning("Không thể tải dữ liệu.")

def show_eda_page(df_data):
    """Hiển thị trang phân tích và trực quan hóa dữ liệu."""
    st.markdown("""
    <div class="main-header">
        <h1>📈 Phân tích & Trực quan hóa (EDA)</h1>
        <p>Khám phá sự phân bổ và mối tương quan của các đặc trưng trong dữ liệu.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_data is not None:
        st.subheader("📌 Phân phối các biến số theo nhãn bệnh tim")

        num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for col in num_cols:
            fig = px.histogram(df_data, x=col, color='target',
                               barmode='overlay',
                               marginal='box',
                               title=f"{col} phân theo tình trạng bệnh tim",
                               color_discrete_map={0: 'green', 1: 'red'})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Mối tương quan giữa các đặc trưng")
        corr = df_data.corr()
        fig_corr = px.imshow(corr, text_auto=True, 
                             title="Ma trận tương quan",
                             color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("⚖️ So sánh giới tính và tình trạng bệnh")
        fig_sex = px.histogram(df_data, x='sex', color='target',
                               barmode='group',
                               category_orders={"sex": [0, 1]},
                               labels={"sex": "Giới tính", "target": "Tình trạng"},
                               color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig_sex, use_container_width=True)

    else:
        st.warning("Không thể tải dữ liệu.")

def show_prediction_page(model_pipeline):
    """Hiển thị trang dự đoán bệnh tim."""
    st.markdown("""
    <div class="main-header">
        <h1>🫀 Hệ thống dự đoán bệnh tim thông minh</h1>
        <p>Sử dụng AI để phân tích nguy cơ mắc bệnh tim dựa trên các chỉ số sức khỏe.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Nhập thông tin để dự đoán")
        age = st.number_input("Tuổi", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Giới tính", [0, 1], format_func=lambda x: "Nam" if x == 1 else "Nữ")
        cp = st.selectbox("Loại đau ngực (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Huyết áp nghỉ (trestbps)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Đường huyết lúc đói > 120mg/dl (fbs)", [0, 1], format_func=lambda x: "Có" if x == 1 else "Không")
        restecg = st.selectbox("Kết quả ECG nghỉ (restecg)", [0, 1, 2])
        thalach = st.number_input("Nhịp tim tối đa (thalach)", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Đau ngực khi gắng sức (exang)", [0, 1], format_func=lambda x: "Có" if x == 1 else "Không")
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Độ dốc ST (slope)", [0, 1, 2])
        ca = st.selectbox("Số mạch máu chính (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    with col2:
        st.subheader("🎯 Bắt đầu dự đoán")
        st.write("Sau khi nhập đủ thông tin, hãy nhấn nút bên dưới để xem kết quả từ AI.")
        predict_button = st.button("Thực hiện dự đoán")
        
        if predict_button:
            if model_pipeline is not None:
                input_data = pd.DataFrame([{
                    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 
                    'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 
                    'slope': slope, 'ca': ca, 'thal': thal
                }])
                
                with st.spinner("🤖 AI đang phân tích..."):
                    prediction = model_pipeline.predict(input_data)[0]
                    # Dòng xác suất đã được bỏ đi theo yêu cầu

                st.subheader("Kết quả dự đoán")
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-container positive-result">
                        <h2>⚠️ CẢNH BÁO: Nguy cơ mắc bệnh tim CAO</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-container negative-result">
                        <h2>✅ KẾT QUẢ: Nguy cơ mắc bệnh tim THẤP</h2>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Mô hình chưa được tải. Không thể dự đoán.")

# --- SIDEBAR ĐIỀU HƯỚNG ---
with st.sidebar:
    st.title("MENU")
    page_options = ["Dự đoán Bệnh tim", "Tổng quan Dataset", "Phân tích & Trực quan hóa"]
    selected_page = st.radio("Điều hướng", page_options)
    st.markdown("""
#### 👥 Thành viên nhóm:

- Huỳnh Thiện Tấn
- Trần Thanh Tú

#### 👩‍🏫 Giảng viên hướng dẫn:
- Trần Thị Thanh Thảo
""")

    st.markdown("---")
    st.info("Hệ thống này chỉ mang tính tham khảo. Luôn hỏi ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác.")


# --- HIỂN THỊ TRANG TƯƠNG ỨNG ---
if selected_page == "Tổng quan Dataset":
    show_overview_page(df)
elif selected_page == "Phân tích & Trực quan hóa":
    show_eda_page(df)
elif selected_page == "Dự đoán Bệnh tim":
    show_prediction_page(model)
