import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Phân loại khách hàng", layout="centered")

st.title("Ứng dụng phân loại khách hàng tiềm năng")
st.write("Dựa trên dữ liệu Mall Customer Segmentation")


uploaded_file = st.file_uploader("Upload dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df.head())

    required_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    if not all(col in df.columns for col in required_cols):
        st.error("File CSV phai chua cac cot: " + ", ".join(required_cols))
    else:
        X = df[required_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow Curve
        st.subheader("Phan cum bang Elbow Method")
        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia.append(km.inertia_)

        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(K_range, inertia, marker='o')
        ax_elbow.set_title("Elbow Curve - Recomend so cum")
        ax_elbow.set_xlabel("So cum K")
        ax_elbow.set_ylabel("Inertia")
        st.pyplot(fig_elbow)

        k = st.slider("Phan so cum khach hang", 2, 10, 5)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        st.success("Done")

        st.subheader("Du lieu phan loai")
        st.dataframe(df.head())

        st.subheader("Dac trung trung binh cua moi nhom khach hang")
        group_summary = df.groupby("Cluster")[required_cols].mean().round(2)
        st.dataframe(group_summary)

        st.subheader("Bieu do phan cum cua khanh hang")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            hue="Cluster",
            palette="Set2",
            data=df,
            ax=ax
        )
        plt.title("Phan nhom theo thu nhap va diem chi tieu")
        st.pyplot(fig)

        st.subheader("Bieu do ti le nhom khach hang")
        fig2, ax2 = plt.subplots()
        df["Cluster"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, ax=ax2)
        ax2.set_ylabel("")
        ax2.set_title("Ty le nhom khach hang")
        st.pyplot(fig2)

        # Download output
        st.subheader("Download output")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "segmented_customers.csv", "text/csv")
