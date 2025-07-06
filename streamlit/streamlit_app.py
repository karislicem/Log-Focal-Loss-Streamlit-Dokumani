import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("📉 Log Loss ve Focal Loss Nedir?")

st.markdown("""
Sınıflandırma problemlerinde modelin tahmin başarısını değerlendirmek için 
<span style='color:#ff4b4b'><b>Log Loss</b></span> ve 
<span style='color:#ff4b4b'><b>Focal Loss</b></span> gibi loss'lar kullanılır.  

- <b>Log Loss</b>, modelin tahmin ettiği olasılıklar üzerinden hesaplanır ve 
  yanlış ama emin tahminlere yüksek ceza verir.
- <b>Focal Loss</b> ise özellikle dengesiz veri setlerinde,
  modelin zor örneklere daha fazla odaklanmasını sağlar.

Bu uygulamada bu iki loss'un nasıl çalıştığını formüller, grafikler ve kısa açıklamalarla inceliyoruz.
""", unsafe_allow_html=True)

st.header("🔷 Log Loss")

st.latex(r"""
\text{Log Loss} = -[y \cdot \log(p) + (1 - y) \cdot \log(1 - p)]
""")

st.markdown("""
- <b>y</b>: Gerçek sınıf etiketi (0 veya 1)  
- <b>p</b>: Modelin 1 sınıfı için tahmin ettiği olasılık  
Log Loss; modelin ne kadar "emin olup hata yaptığını" ölçer.
""", unsafe_allow_html=True)

with st.expander("💡 Neden log kullanıyoruz?"):
    st.markdown("""
    - Logaritmik yapı; küçük hatalara düşük, büyük hatalara yüksek ceza verir.  
    - Model çok düşük olasılık verip yanılırsa (örneğin `y=1`, `p=0.01`), ceza çok büyür.  
    - Bu, modeli daha dikkatli tahminler yapmaya teşvik eder.  
    - Bu yüzden sınıflandırmalarda en yaygın kullanılan loss'tur.
    """)

p = np.linspace(0.001, 0.999, 100)
loss_1 = -np.log(p)
loss_0 = -np.log(1 - p)

fig1, ax1 = plt.subplots()
ax1.plot(p, loss_1, label='Gerçek: y = 1')
ax1.plot(p, loss_0, label='Gerçek: y = 0')
ax1.set_title("Log Loss")
ax1.set_xlabel("Modelin tahmin ettiği olasılık (p)")
ax1.set_ylabel("Loss")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

st.header("🔷 Focal Loss")

st.latex(r"""
\text{FL}(p_t) = -\alpha \cdot (1 - p_t)^\gamma \cdot \log(p_t)
""")

st.markdown("""
- <b>pₜ</b>: Modelin doğru sınıfa verdiği olasılık  
- <b>γ (gamma)</b>: Odaklanma parametresi (zor örneklerin ağırlığını artırır)  
- <b>α (alpha)</b>: Sınıf ağırlığı (opsiyonel)  

Focal Loss, kolay örneklerin etkisini azaltır. Böylece model, zor örnekleri daha fazla öğrenmeye çalışır.
""", unsafe_allow_html=True)

with st.expander("💡 Focal Loss ne zaman tercih edilir?"):
    st.markdown("""
    - Eğer veri setin dengesizse (%90 negatif, %10 pozitif gibi), Log Loss çoğunluk sınıfa odaklanır.  
    - Bu durumda azınlık görülen sınıflar ihmal edilebilir.  
    - Focal Loss bu azınlık sınıflara daha fazla odaklanma imkanı verir.  
    - Özellikle <b>nesne tespiti</b>, <b>fraud tespiti</b> ve <b>medikal veri</b> gibi alanlarda sık kullanılır.
    """, unsafe_allow_html=True)

gamma = st.slider("Gamma (odaklanma gücü)", 0.0, 5.0, 2.0, 0.1)
alpha = st.slider("Alpha (sınıf ağırlığı)", 0.0, 1.0, 1.0, 0.1)

pt = np.linspace(0.001, 0.999, 100)
focal_loss = -alpha * (1 - pt)**gamma * np.log(pt)

fig2, ax2 = plt.subplots()
ax2.plot(pt, focal_loss, label=f'γ = {gamma}, α = {alpha}')
ax2.set_title("Focal Loss (Gerçek sınıf = 1)")
ax2.set_xlabel("Modelin doğru sınıfa verdiği tahmin olasılığı (pₜ)")
ax2.set_ylabel("Loss")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

st.header("📌 Özet")

st.markdown("""
- <b>Log Loss</b>: Her örneği eşit önemser. Hatalı ve emin tahminlerde çok ceza verir.  
- <b>Focal Loss</b>: Kolay örneklerin etkisini azaltır, zor örnekleri öne çıkarır.  
- Dengesiz veri setlerinde Focal Loss tercih edilirse, model daha adil sonuçlar verebilir.
""", unsafe_allow_html=True)

with st.expander("📚 İlgili Makaleler ve Grafikler"):
    st.markdown("### 📘 1. *Focal Loss for Dense Object Detection*")
    st.markdown("""
**Yazarlar:** Lin, Goyal, Girshick, He, & Dollár (2017)  
**Özet:** One-stage object detection modellerinde sınıf dengesizliğini çözmek için Focal Loss önerildi.  
Yeni geliştirilen RetinaNet modeliyle başarı elde edildi.
    """)

    st.markdown("### 📘 2. *Calibrating Deep Neural Networks using Focal Loss*")
    st.markdown("""
**Yazarlar:** Mukhoti, Ghosh, Kumar, & Torr (2020)  
**Özet:** Focal Loss’un ağların güven kalibrasyonunu da iyileştirdiği gösterildi.  
Tahmin edilen olasılıklar, gerçek olasılıklara daha yakın hale geliyor.
    """)

    st.markdown("### 📘 3. *A Comprehensive Review of Loss Functions and Metrics in Deep Learning Classification*")
    st.markdown("""
**Yazarlar:** Terven & Cordova-Esparza (2021)  
**Özet:** Tüm popüler loss ve metrikler karşılaştırıldı. Log Loss ve Focal Loss'un avantajları tablolarla açıklandı.
    """)

    st.markdown("### 📊 Farklı Gamma Değerleriyle Focal Loss Grafiği")

    gamma_vals = [0, 1, 2, 5]
    fig3, ax3 = plt.subplots()
    for g in gamma_vals:
        fl = -(1 - pt)**g * np.log(pt)
        ax3.plot(pt, fl, label=f"γ={g}")
    ax3.set_title("Gamma Parametresine Göre Focal Loss Değişimi (α=1)")
    ax3.set_xlabel("pₜ")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    st.markdown("### 📖 APA Formatında Kaynaklar")
    st.markdown("""
**1.** Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).  
Focal loss for dense object detection. *ICCV*, 2980–2988. https://doi.org/10.1109/ICCV.2017.324

**2.** Mukhoti, J., Ghosh, A., Kumar, M. P., & Torr, P. H. S. (2020).  
Calibrating deep neural networks using focal loss. *NeurIPS*, 33, 15288–15299. https://arxiv.org/abs/2002.09437

**3.** Terven, J., & Cordova-Esparza, D. M. (2021).  
A comprehensive review of loss functions and metrics for deep learning classification. *Pattern Recognition*, 125, 108198. https://doi.org/10.1016/j.patcog.2021.108198
    """)

st.sidebar.markdown("👤 Cem Karışlı  \n[GitHub](https://github.com/karislicem) | [LinkedIn](https://www.linkedin.com/in/karislicem/)")
