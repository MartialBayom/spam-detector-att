# 🕵️ AT&T Détecteur de Spam par Deep Learning

> *Classifier automatiquement des SMS comme spam ou ham grâce au Deep Learning — Bi-LSTM et Transfer Learning avec DistilBERT*

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E)](https://huggingface.co/)

---

## 🎯 Objectif

AT&T reçoit des millions de SMS par jour. L'entreprise cherche un système **automatisé** pour détecter les spams dès leur réception, basé uniquement sur le contenu du message.

---

## 📊 Résultats

| Modèle | Accuracy | F1 (Spam) | Precision | Recall | AUC-ROC | Paramètres |
|---|---|---|---|---|---|---|
| Bi-LSTM | 0.9749 | 0.9091 | 0.8861 | 0.9333 | 0.9901 | ~500K |
| **DistilBERT ⭐** | **0.9875** | **0.9517** | 0.9857 | 0.9200 | **0.9986** | 66M |

> Dataset déséquilibré : **86.6% ham / 13.4% spam** → F1-score comme métrique principale.

---

## 🗂️ Structure du projet

```
spam-detector/
├── data/
│   └── spam.csv                      # 5 572 SMS labellisés (ham/spam)
├── notebooks/
│   └── spam_detector_att.ipynb       # EDA + Bi-LSTM + DistilBERT
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧠 Modèles

| Modèle | Architecture | Détail |
|---|---|---|
| **Bi-LSTM** | Embedding → LSTM bidirectionnel → Linear | Entraîné from scratch, ~500K params |
| **DistilBERT** | Transformer pré-entraîné + tête de classification | Fine-tuning avec dégel progressif |

---

## 📈 Insights clés

- **13.4%** des SMS du dataset sont des spams (86.6% ham) — dataset déséquilibré
- Les spams sont **significativement plus longs** que les ham (138.9 caractères en moyenne contre 71.0 — signal discriminant fort)
- **DistilBERT** comprend le contexte sémantique — distingue "free" dans un spam vs "feel free" dans un ham
- Sur ce dataset, **DistilBERT surpasse le Bi-LSTM** sur toutes les métriques, notamment le F1-score (0.9517 vs 0.9091)

---

## ⚙️ Installation

```bash
git clone https://github.com/MartialBayom/spam-detector-att.git
cd spam-detector-att
pip install -r requirements.txt
jupyter notebook notebooks/spam_detector_att.ipynb
```

---

## 👤 Auteur

| | Nom | Rôle |
|---|---|---|
| 🧑‍💻 | **Martial BAYOM** | Data Science |

---

## 📂 Sources

| Dataset | Lien |
|---|---|
| SMS Spam Collection | [Jedha / UCI ML Repository](https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Deep+Learning/project/spam.csv) |
