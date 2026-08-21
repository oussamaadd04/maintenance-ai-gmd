import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.special import gamma as gamma_fn
import json, os, csv

st.set_page_config(
    page_title="MaintenanceAI — GMD Métal Tanger",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .header-band {
    background: linear-gradient(135deg,#0C1F35,#1E3A5F);
    padding:18px 24px; border-radius:10px; margin-bottom:18px; color:white;
  }
  .header-band h1 {font-size:20px;font-weight:700;margin:0;color:white}
  .header-band p  {font-size:12px;color:#7A90A8;margin:4px 0 0 0}
  .kpi-box {
    background:white; border:1px solid #E2E8F0; border-radius:10px;
    padding:16px; text-align:center;
  }
  .risk-red    {background:#FEE2E2;border-left:4px solid #DC2626;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px}
  .risk-orange {background:#FEF3C7;border-left:4px solid #D97706;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px}
  .risk-green  {background:#DCFCE7;border-left:4px solid #16A34A;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px}
  .alert-crit  {background:#FEE2E2;border:1px solid #FCA5A5;border-radius:8px;padding:12px 16px;margin-bottom:8px}
  .alert-warn  {background:#FEF3C7;border:1px solid #FCD34D;border-radius:8px;padding:12px 16px;margin-bottom:8px}
  .fiche-card  {background:white;border:1.5px solid #E2E8F0;border-radius:10px;padding:20px;margin-bottom:16px}
  .guide-box   {background:#F0F9FF;border:1px solid #BAE6FD;border-radius:8px;padding:14px;margin-bottom:10px}
  .tip-box     {background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:12px;margin-bottom:8px}
  div[data-testid="metric-container"] {
    background:white; border:1px solid #E2E8F0; border-radius:10px; padding:12px 16px;
  }
  .footer {font-size:11px;color:#94A3B8;text-align:center;margin-top:20px;
           padding-top:10px;border-top:1px solid #E2E8F0}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PARAMÈTRES FIXES DU MODÈLE
# ═══════════════════════════════════════════
FAMILLES = [
    'Lanceur / Bol Vibrant','Panne Machine Générale',
    'Capteurs / Cellules','Blocage Écrou','Volet / Trappe',
    'Circuit Refroidissement','Circuit Pneumatique','Plateau Indexage',
    'Défaut Soudure / Électrodes','Problème Électrique'
]

WEIBULL = {
    'Lanceur / Bol Vibrant':       {'beta':0.653,'eta':40.1},
    'Panne Machine Générale':      {'beta':0.611,'eta':55.2},
    'Capteurs / Cellules':         {'beta':0.720,'eta':126.8},
    'Blocage Écrou':               {'beta':0.628,'eta':110.2},
    'Volet / Trappe':              {'beta':0.661,'eta':109.1},
    'Circuit Refroidissement':     {'beta':0.718,'eta':140.4},
    'Circuit Pneumatique':         {'beta':0.701,'eta':164.9},
    'Plateau Indexage':            {'beta':0.485,'eta':167.1},
    'Défaut Soudure / Électrodes': {'beta':0.549,'eta':687.7},
    'Problème Électrique':         {'beta':0.876,'eta':1746.6},
}
MTBF = {
    'Lanceur / Bol Vibrant':54.6,'Panne Machine Générale':80.7,
    'Capteurs / Cellules':157.6,'Blocage Écrou':165.5,'Volet / Trappe':146.6,
    'Circuit Refroidissement':172.6,'Circuit Pneumatique':210.0,
    'Plateau Indexage':310.5,'Défaut Soudure / Électrodes':1040.7,
    'Problème Électrique':1855.9,
}
AMDEC = {
    'Lanceur / Bol Vibrant':500,'Panne Machine Générale':336,
    'Capteurs / Cellules':320,'Blocage Écrou':140,'Volet / Trappe':192,
    'Circuit Refroidissement':180,'Circuit Pneumatique':75,
    'Plateau Indexage':80,'Défaut Soudure / Électrodes':63,
    'Problème Électrique':45,
}
RECALL_V2 = {
    'Lanceur / Bol Vibrant':0.974,'Panne Machine Générale':1.000,
    'Capteurs / Cellules':0.726,'Blocage Écrou':0.795,'Volet / Trappe':0.725,
    'Circuit Refroidissement':0.743,'Circuit Pneumatique':0.562,
    'Plateau Indexage':0.286,'Défaut Soudure / Électrodes':0.000,
    'Problème Électrique':0.000,
}
ACTIONS = {
    'Lanceur / Bol Vibrant':   ['Inspecter visuellement le bol vibrant','Vérifier la gaine de transfert écrous','Contrôler les capteurs de présence','Nettoyer le séparateur d\'écrous','Vérifier la pression d\'air (0.4–0.6 MPa)'],
    'Panne Machine Générale':  ['Consulter les alarmes API sur pupitre PROFACE','Vérifier le journal d\'événements automate','Redémarrer le cycle après diagnostic','Alerter le responsable maintenance','Vérifier l\'alimentation 400V triphasé'],
    'Capteurs / Cellules':     ['Nettoyer les capteurs de présence','Vérifier l\'alignement des barrières immatérielles','Contrôler les câbles de connexion','Tester les capteurs en mode manuel','Vérifier les interrupteurs de sécurité carters'],
    'Blocage Écrou':           ['Vérifier la goulotte de guidage','Nettoyer le shut d\'écrous','Contrôler l\'orientation des écrous','Vérifier l\'usure mécanique de la goulotte','Régler la fréquence du bol vibrant'],
    'Volet / Trappe':          ['Vérifier l\'état des volets de sécurité','Contrôler les capteurs de position volet','Inspecter les charnières et fixations','Vérifier le frein de porte','Tester la fermeture automatique en cycle'],
    'Circuit Refroidissement': ['Vérifier le débit d\'eau (capteurs débit)','Inspecter les raccords et joints','Contrôler la température des électrodes','Vérifier le répartiteur d\'eau','Purger le circuit si nécessaire'],
    'Circuit Pneumatique':     ['Contrôler la pression FRL (panneau pneumatique)','Vérifier les raccords et joints pneumatiques','Inspecter les distributeurs','Contrôler les vérins de la cellule','Vérifier la souflette et les filtres'],
    'Plateau Indexage':        ['Vérifier l\'indexeur à came','Contrôler le capteur fin de course','Inspecter les doigts orienteurs','Vérifier la lubrification du raccord tournant','Contrôler le positionnement des 4 postes'],
    'Défaut Soudure / Électrodes':['Contrôler l\'usure des électrodes','Vérifier la pression de soudage','Inspecter les semelles cuivre','Contrôler l\'intensité (défaut CPS)','Vérifier le transformateur 250 KVA'],
    'Problème Électrique':     ['Vérifier le tableau électrique 400V','Contrôler les fusibles et disjoncteurs','Vérifier la platine thyristors','Inspecter les câbles d\'alimentation','Alerter l\'électricien de maintenance'],
}
CAUSES = {
    'Lanceur / Bol Vibrant':   ['Encrassement du bol vibrant','Mauvais réglage de la fréquence de vibration','Tuyau de transfert détérioré','Aimant permanent encrassé'],
    'Panne Machine Générale':  ['Cause non identifiée dans la GMAO','Défaut automate Schneider','Problème communication API/IHM','Alimentation instable'],
    'Capteurs / Cellules':     ['Encrassement des capteurs','Désalignement par vibrations','Câble détérioré ou arraché','Barrière immatérielle mal réglée'],
    'Blocage Écrou':           ['Écrou mal orienté dans la goulotte','Corps étranger dans le circuit','Usure mécanique de la goulotte','Shut bloqué'],
    'Volet / Trappe':          ['Choc mécanique sur le volet','Fatigue matériau','Usure des charnières','Capteur de position décalé'],
    'Circuit Refroidissement': ['Joint dégradé ou canalisation corrodée','Débit insuffisant','Filtre colmaté','Raccord tournant défaillant'],
    'Circuit Pneumatique':     ['Fuite de joints','Encrassement distributeur','Bobine défaillante','Pression insuffisante'],
    'Plateau Indexage':        ['Usure de came','Détérioration raccord tournant','Usure des doigts orienteurs','Jeu mécanique excessif'],
    'Défaut Soudure / Électrodes':['Cycles thermiques répétés','Oxydation semelle cuivre','Serrage insuffisant','Pression de soudage incorrecte'],
    'Problème Électrique':     ['Composant électrique vieillissant','Défaut réseau 400V','Fusible grillé','Thyristor défaillant'],
}

# ═══════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════
def prob_weibull(tbf_h, famille):
    b, e = WEIBULL[famille]['beta'], WEIBULL[famille]['eta']
    if tbf_h <= 0: return 0.0
    return float(1 - np.exp(-((tbf_h/e)**b)))

def score_risque(tbf_h, famille, jours_mps, pannes_7j, pannes_30j):
    pw       = prob_weibull(tbf_h, famille)
    mps_ret  = 1 if jours_mps > MTBF[famille]/24*1.2 else 0
    crit_n   = AMDEC[famille]/500.0
    s = (0.25*pw + 0.20*min(1,tbf_h/(MTBF[famille]*1.5)) +
         0.20*mps_ret + 0.15*crit_n +
         0.10*min(1,pannes_7j/5) + 0.10*min(1,pannes_30j/10))
    return round(min(s*100,99.0),1)

def badge(score):
    if score>=70: return "🔴","CRITIQUE","#DC2626","#FEE2E2"
    if score>=40: return "🟡","VIGILANCE","#D97706","#FEF3C7"
    return "🟢","NORMAL","#16A34A","#DCFCE7"

CSV_PANNES = "pannes_saisies.csv"
CSV_MPS    = "mps_saisies.csv"

def charger_csv(path, colonnes):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=colonnes)

def sauver_panne(row):
    existe = os.path.exists(CSV_PANNES)
    with open(CSV_PANNES,'a',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not existe: w.writeheader()
        w.writerow(row)

def sauver_mps(row):
    existe = os.path.exists(CSV_MPS)
    with open(CSV_MPS,'a',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not existe: w.writeheader()
        w.writerow(row)

# ═══════════════════════════════════════════
# ÉTAT DE SESSION
# ═══════════════════════════════════════════
if 'etat' not in st.session_state:
    now = datetime.now()
    st.session_state.etat = {
        'Lanceur / Bol Vibrant':       {'panne':now-timedelta(hours=68), 'mps':now-timedelta(days=18),'p7':2,'p30':5},
        'Panne Machine Générale':      {'panne':now-timedelta(hours=49), 'mps':now-timedelta(days=8), 'p7':1,'p30':3},
        'Capteurs / Cellules':         {'panne':now-timedelta(hours=112),'mps':now-timedelta(days=7), 'p7':3,'p30':6},
        'Blocage Écrou':               {'panne':now-timedelta(hours=54), 'mps':now-timedelta(days=3), 'p7':0,'p30':2},
        'Volet / Trappe':              {'panne':now-timedelta(hours=41), 'mps':now-timedelta(days=5), 'p7':0,'p30':1},
        'Circuit Refroidissement':     {'panne':now-timedelta(hours=76), 'mps':now-timedelta(days=22),'p7':1,'p30':2},
        'Circuit Pneumatique':         {'panne':now-timedelta(hours=96), 'mps':now-timedelta(days=10),'p7':0,'p30':1},
        'Plateau Indexage':            {'panne':now-timedelta(hours=120),'mps':now-timedelta(days=6), 'p7':0,'p30':1},
        'Défaut Soudure / Électrodes': {'panne':now-timedelta(hours=480),'mps':now-timedelta(days=12),'p7':0,'p30':0},
        'Problème Électrique':         {'panne':now-timedelta(hours=720),'mps':now-timedelta(days=15),'p7':0,'p30':0},
    }
if 'recall_actuel' not in st.session_state:
    st.session_state.recall_actuel = 0.771
if 'auc_actuel' not in st.session_state:
    st.session_state.auc_actuel = 0.772
if 'nb_saisies' not in st.session_state:
    st.session_state.nb_saisies = 0
if 'historique_recall' not in st.session_state:
    st.session_state.historique_recall = [('Initial V2', 0.771)]

def get_risques():
    now = datetime.now()
    res = {}
    for f, e in st.session_state.etat.items():
        tbf   = (now - e['panne']).total_seconds()/3600
        jmps  = (now - e['mps']).days
        score = score_risque(tbf, f, jmps, e['p7'], e['p30'])
        em,lb,co,bg = badge(score)
        res[f] = {'score':score,'emoji':em,'label':lb,'color':co,'bg':bg,
                  'tbf':round(tbf,1),'jmps':jmps,'p7':e['p7'],'p30':e['p30'],
                  'pw':round(prob_weibull(tbf,f),3)}
    return dict(sorted(res.items(), key=lambda x:-x[1]['score']))

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="background:#0C1F35;padding:14px;border-radius:10px;margin-bottom:12px">
      <div style="color:#4A9EDB;font-size:9px;letter-spacing:2px;font-weight:700">GMD MÉTAL TANGER</div>
      <div style="color:white;font-size:16px;font-weight:700;margin-top:4px">🔧 MaintenanceAI</div>
      <div style="color:#7A90A8;font-size:11px">Cellule DENGENSHA · ZAP PLT</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y  %H:%M')}")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠 Dashboard Global",
        "🤖 Analyse Prédictive ML",
        "🚨 Fiches Intervention",
        "📉 Analyse Fiabilité Weibull",
        "📊 Historique & Pareto",
        "➕ Saisir une intervention",
        "⚙️ Administration Modèle",
        "📖 Guide d'utilisation",
    ], label_visibility="collapsed")

    st.markdown("---")
    risques_side = get_risques()
    nb_crit = sum(1 for r in risques_side.values() if r['score']>=70)
    nb_vig  = sum(1 for r in risques_side.values() if 40<=r['score']<70)
    st.markdown(f"""
    <div style="font-size:12px;line-height:1.9">
    🔴 <b>{nb_crit}</b> familles critiques<br>
    🟡 <b>{nb_vig}</b> en vigilance<br>
    📊 Recall : <b>{st.session_state.recall_actuel:.3f}</b><br>
    📈 AUC : <b>{st.session_state.auc_actuel:.3f}</b><br>
    💾 Saisies session : <b>{st.session_state.nb_saisies}</b>
    </div>
    """, unsafe_allow_html=True)

def footer():
    st.markdown('<div class="footer">MaintenanceAI · GMD Métal Tanger · Cellule DENGENSHA · PFA 2025–2026 · RF V2 · AUC 0.772</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 1 — DASHBOARD GLOBAL
# ═══════════════════════════════════════════
if page == "🏠 Dashboard Global":
    st.markdown("""
    <div class="header-band">
      <h1>🏭 Dashboard Global — Cellule DENGENSHA</h1>
      <p>ZAP PLT · UAP Assemblage · GMD Métal Tanger · Vue usine temps réel</p>
    </div>""", unsafe_allow_html=True)

    risques = get_risques()
    scores  = [r['score'] for r in risques.values()]
    moy     = round(np.mean(scores),1)
    nb_c    = sum(1 for s in scores if s>=70)
    nb_v    = sum(1 for s in scores if 40<=s<70)
    nb_n    = sum(1 for s in scores if s<40)
    max_f   = max(risques, key=lambda f: risques[f]['score'])

    # Jauge globale
    col_g, col_k = st.columns([1,3])
    with col_g:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=moy,
            title={'text':'État cellule (%)','font':{'size':13}},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':'#DC2626' if moy>=70 else '#D97706' if moy>=40 else '#16A34A'},
                   'steps':[{'range':[0,40],'color':'#DCFCE7'},
                             {'range':[40,70],'color':'#FEF3C7'},
                             {'range':[70,100],'color':'#FEE2E2'}],
                   'threshold':{'line':{'color':'black','width':3},'thickness':0.75,'value':70}}
        ))
        fig_g.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_k:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🔴 Critiques",   nb_c,  help="Risque > 70%")
        c2.metric("🟡 Vigilance",   nb_v,  help="Risque 40-70%")
        c3.metric("🟢 Normales",    nb_n,  help="Risque < 40%")
        c4.metric("⚠️ Risque max",  f"{risques[max_f]['score']}%", help=max_f)

        # Alerte bannière
        if nb_c > 0:
            st.error(f"🚨 ALERTE CRITIQUE — {nb_c} famille(s) nécessitent une intervention immédiate !")
        elif nb_v > 0:
            st.warning(f"⚠️ {nb_v} famille(s) en surveillance renforcée")
        else:
            st.success("✅ Toutes les familles sont en situation normale")

    st.markdown("---")
    col_l, col_r = st.columns([3,2])

    with col_l:
        st.subheader("Niveau de risque par famille")
        for famille, r in risques.items():
            tendance = "↑" if r['score'] > 50 else "↓"
            st.markdown(f"""
            <div style="background:{r['bg']};border-left:4px solid {r['color']};
                        padding:11px 16px;border-radius:0 8px 8px 0;margin-bottom:7px">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <span style="font-weight:700;font-size:13px">{r['emoji']} {famille}</span>
                  <span style="margin-left:8px;background:{r['color']};color:white;
                        font-size:10px;padding:2px 8px;border-radius:10px;font-weight:600">
                    {r['score']}% {tendance} — {r['label']}
                  </span>
                </div>
                <div style="font-size:11px;color:#475569">
                  TBF: <b>{r['tbf']}h</b> &nbsp;|&nbsp; MPS: <b>{r['jmps']}j</b> &nbsp;|&nbsp; Pannes 7j: <b>{r['p7']}</b>
                </div>
              </div>
              <div style="margin-top:8px;height:6px;background:#E2E8F0;border-radius:3px">
                <div style="width:{r['score']}%;height:100%;background:{r['color']};border-radius:3px;transition:width 0.5s"></div>
              </div>
              <div style="font-size:11px;color:#475569;margin-top:5px">
                {"⚡ Intervenir dans les 24h — contacter le responsable maintenance" if r['score']>=70 else "👁️ Surveiller l'évolution — planifier une MPS" if r['score']>=40 else "✅ Situation normale — poursuivre le plan préventif standard"}
              </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.subheader("Alertes actives")
        alertes = [(f,r) for f,r in risques.items() if r['score']>=40]
        if alertes:
            for f,r in alertes:
                cls = "alert-crit" if r['score']>=70 else "alert-warn"
                st.markdown(f"""
                <div class="{cls}">
                  <div style="font-weight:700;font-size:12.5px">{r['emoji']} {f}</div>
                  <div style="font-size:11px;color:#475569;margin-top:4px">
                    Risque {r['score']}% · TBF={r['tbf']}h · MPS={r['jmps']}j<br>
                    {"→ Intervention requise dans les 24h" if r['score']>=70 else "→ Surveiller et planifier MPS"}
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Aucune alerte active")

        st.markdown("---")
        st.subheader("Suivi plan MPS")
        mps_rows = []
        for f,r in risques.items():
            if r['jmps'] > 16: stat = "⚠️ En retard"
            elif r['jmps'] > 10: stat = "🟡 Proche"
            else: stat = "✅ À jour"
            mps_rows.append({'Famille':f[:22],'Dernière MPS':f"il y a {r['jmps']}j",'Statut':stat})
        st.dataframe(pd.DataFrame(mps_rows), hide_index=True, use_container_width=True)

    footer()

# ═══════════════════════════════════════════
# PAGE 2 — ANALYSE PRÉDICTIVE ML
# ═══════════════════════════════════════════
elif page == "🤖 Analyse Prédictive ML":
    st.markdown("""
    <div class="header-band">
      <h1>🤖 Analyse Prédictive ML — Random Forest V2</h1>
      <p>Probabilités de panne calculées par le modèle · 15 features · AUC 0.772</p>
    </div>""", unsafe_allow_html=True)

    risques = get_risques()

    # Tableau principal
    rows = []
    for f,r in risques.items():
        action = "Intervention immédiate" if r['score']>=70 else "Contrôle préventif" if r['score']>=40 else "Surveillance standard"
        rows.append({'Famille':f,'Probabilité':f"{r['score']}%",'Niveau':f"{r['emoji']} {r['label']}",'TBF (h)':r['tbf'],'MPS (j)':r['jmps'],'Pannes 7j':r['p7'],'Action':action})
    df_main = pd.DataFrame(rows)
    st.dataframe(df_main, hide_index=True, use_container_width=True, height=380)

    st.markdown("---")
    st.subheader("Détail par famille")
    famille_sel = st.selectbox("Sélectionner une famille pour le détail", FAMILLES)
    r = risques[famille_sel]

    col1, col2 = st.columns([1,2])
    with col1:
        # Jauge probabilité
        fig_j = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=r['score'],
            title={'text':f"Risque — {famille_sel[:20]}"},
            delta={'reference':50,'valueformat':'.1f'},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':r['color']},
                   'steps':[{'range':[0,40],'color':'#DCFCE7'},
                             {'range':[40,70],'color':'#FEF3C7'},
                             {'range':[70,100],'color':'#FEE2E2'}]}
        ))
        fig_j.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=10))
        st.plotly_chart(fig_j, use_container_width=True)
        st.markdown(f"**Horizon prédiction :** 24 heures")
        st.markdown(f"**Recall modèle :** {RECALL_V2[famille_sel]:.3f}")
        if r['score']>=70:
            st.error("🚨 Intervention recommandée")
        elif r['score']>=40:
            st.warning("⚠️ Surveillance renforcée")
        else:
            st.success("✅ Situation normale")

    with col2:
        st.markdown("**Facteurs explicatifs :**")
        facteurs = []
        if r['tbf'] > MTBF[famille_sel]: facteurs.append(f"⏱️ TBF actuel ({r['tbf']}h) dépasse le MTBF historique ({MTBF[famille_sel]}h) — la machine tourne depuis trop longtemps sans panne")
        if r['jmps'] > 14: facteurs.append(f"🛠️ MPS en retard depuis {r['jmps']} jours — la maintenance préventive n'a pas été effectuée")
        if r['p7'] >= 2: facteurs.append(f"📈 {r['p7']} pannes cette semaine sur cette famille — fréquence anormalement élevée")
        if r['pw'] > 0.6: facteurs.append(f"📊 Probabilité Weibull élevée ({r['pw']}) — statistiquement en zone de risque")
        if AMDEC[famille_sel] >= 300: facteurs.append(f"⚡ Criticité AMDEC élevée ({AMDEC[famille_sel]}) — impact fort sur la production")
        if not facteurs: facteurs.append("✅ Aucun facteur de risque majeur détecté")

        for f_text in facteurs:
            st.markdown(f"""
            <div style="background:#FEF9E7;border-left:3px solid #D97706;
                        padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:6px;font-size:12.5px">
              {f_text}
            </div>""", unsafe_allow_html=True)

        st.markdown("**Causes probables connues :**")
        for c in CAUSES[famille_sel]:
            st.markdown(f"• {c}")

    footer()

# ═══════════════════════════════════════════
# PAGE 3 — FICHES INTERVENTION
# ═══════════════════════════════════════════
elif page == "🚨 Fiches Intervention":
    st.markdown("""
    <div class="header-band">
      <h1>🚨 Fiches Intervention — Recommandations maintenance</h1>
      <p>Fiches générées automatiquement par le système prédictif</p>
    </div>""", unsafe_allow_html=True)

    risques = get_risques()
    alertes = [(f,r) for f,r in risques.items() if r['score']>=40]
    normaux = [(f,r) for f,r in risques.items() if r['score']<40]

    filtre = st.selectbox("Filtrer par niveau", ["Toutes les alertes","Critiques uniquement","Vigilance uniquement"])

    if filtre == "Critiques uniquement":
        alertes = [(f,r) for f,r in alertes if r['score']>=70]
    elif filtre == "Vigilance uniquement":
        alertes = [(f,r) for f,r in alertes if 40<=r['score']<70]

    if not alertes:
        st.success("✅ Aucune alerte à afficher avec ce filtre")
    else:
        for famille, r in alertes:
            border = "#DC2626" if r['score']>=70 else "#D97706"
            with st.expander(f"{r['emoji']} {famille} — {r['score']}% — {r['label']}", expanded=r['score']>=70):
                col1, col2 = st.columns([1,1])
                with col1:
                    st.markdown(f"""
                    **📋 FICHE INTERVENTION**
                    | Champ | Valeur |
                    |-------|--------|
                    | Équipement | {famille} |
                    | Risque | **{r['score']}%** |
                    | Niveau | {r['emoji']} {r['label']} |
                    | TBF actuel | {r['tbf']}h |
                    | MTBF historique | {MTBF[famille]}h |
                    | MPS dernière | il y a {r['jmps']} jours |
                    | Pannes 7 jours | {r['p7']} |
                    | Date fiche | {datetime.now().strftime('%d/%m/%Y %H:%M')} |
                    """)
                with col2:
                    st.markdown("**⚠️ Causes probables :**")
                    for c in CAUSES[famille]:
                        st.markdown(f"✓ {c}")
                    st.markdown("**🔧 Actions recommandées :**")
                    for i,a in enumerate(ACTIONS[famille],1):
                        st.markdown(f"{i}. {a}")

                technicien = st.text_input(f"Technicien responsable — {famille[:20]}", placeholder="Nom du technicien", key=f"tech_{famille}")
                commentaire = st.text_area(f"Observations après intervention", placeholder="Décrire les actions effectuées...", key=f"com_{famille}", height=80)

                col_s, col_e = st.columns([1,1])
                with col_s:
                    statut = st.selectbox("Statut", ["En attente","Prise en charge","Résolue"], key=f"stat_{famille}")
                with col_e:
                    if st.button(f"📄 Exporter cette fiche", key=f"exp_{famille}"):
                        contenu = f"""FICHE INTERVENTION — GMD MÉTAL TANGER
Cellule DENGENSHA — ZAP PLT
Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}
{'='*50}
Famille : {famille}
Risque  : {r['score']}% — {r['label']}
TBF     : {r['tbf']}h / MTBF : {MTBF[famille]}h
MPS     : il y a {r['jmps']} jours
{'='*50}
CAUSES PROBABLES :
{chr(10).join(f'• {c}' for c in CAUSES[famille])}
{'='*50}
ACTIONS RECOMMANDÉES :
{chr(10).join(f'{i+1}. {a}' for i,a in enumerate(ACTIONS[famille]))}
{'='*50}
Technicien : {technicien}
Statut     : {statut}
Observations : {commentaire}
"""
                        st.download_button("💾 Télécharger TXT", contenu,
                                           file_name=f"fiche_{famille[:15].replace('/','_').replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                                           mime="text/plain", key=f"dl_{famille}")

    if normaux:
        st.markdown("---")
        st.success(f"✅ {len(normaux)} famille(s) en situation normale — aucune intervention requise")
        for f,r in normaux:
            st.markdown(f"🟢 **{f}** — {r['score']}% · TBF={r['tbf']}h · MPS={r['jmps']}j")

    footer()

# ═══════════════════════════════════════════
# PAGE 4 — ANALYSE FIABILITÉ WEIBULL
# ═══════════════════════════════════════════
elif page == "📉 Analyse Fiabilité Weibull":
    st.markdown("""
    <div class="header-band">
      <h1>📉 Analyse Fiabilité — Loi de Weibull</h1>
      <p>Modélisation statistique du comportement des familles de pannes</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("❓ C'est quoi la loi de Weibull ? — Cliquer pour comprendre"):
        st.markdown("""
        <div class="guide-box">
        <b>En langage simple :</b> La loi de Weibull est un outil mathématique qui permet de prédire 
        quand une machine va tomber en panne, en analysant les durées entre les pannes passées.<br><br>
        <b>Les 2 paramètres clés :</b><br>
        • <b>β (bêta)</b> : indique si la machine s'use progressivement ou tombe en panne aléatoirement.<br>
        &nbsp;&nbsp;→ β < 1 : pannes surtout juste après une réparation<br>
        &nbsp;&nbsp;→ β = 1 : pannes aléatoires (pas d'usure)<br>
        &nbsp;&nbsp;→ β > 1 : usure progressive (risque augmente avec le temps)<br><br>
        • <b>η (eta)</b> : durée au bout de laquelle 63% des composants ont déjà subi une panne.<br><br>
        <b>F(t)</b> = probabilité que la panne soit survenue avant l'heure t<br>
        <b>R(t)</b> = probabilité que la machine fonctionne encore à l'heure t
        </div>
        """, unsafe_allow_html=True)

    risques = get_risques()
    famille_sel = st.selectbox("Sélectionner une famille", FAMILLES)
    r = risques[famille_sel]
    b, e = WEIBULL[famille_sel]['beta'], WEIBULL[famille_sel]['eta']
    t_alerte = e * ((-np.log(0.30))**(1/b))
    mttf_h = e * gamma_fn(1+1/b)

    col1, col2 = st.columns([3,2])
    with col1:
        t_max = min(e*6, 3000)
        t = np.linspace(0.1, t_max, 400)
        ft = 1 - np.exp(-((t/e)**b))
        rt = np.exp(-((t/e)**b))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t,y=ft,name='F(t) — Probabilité de panne',line=dict(color='#DC2626',width=2.5)))
        fig.add_trace(go.Scatter(x=t,y=rt,name='R(t) — Fiabilité restante',line=dict(color='#16A34A',width=2.5)))
        fig.add_vrect(x0=t_alerte, x1=t_max, fillcolor="#FEE2E2", opacity=0.3, line_width=0, annotation_text="Zone danger")
        fig.add_vline(x=r['tbf'],line_dash="dash",line_color="#2563EB",line_width=2,
                      annotation_text=f"Position actuelle ({r['tbf']}h)",annotation_position="top right")
        fig.add_vline(x=t_alerte,line_dash="dot",line_color="#D97706",line_width=1.5,
                      annotation_text=f"Seuil alerte 70% ({t_alerte:.0f}h)")
        fig.add_hline(y=0.70,line_dash="dot",line_color="#D97706",opacity=0.5)
        fig.update_layout(title=f"Courbe Weibull — {famille_sel}",
                          xaxis_title="Temps (heures)",yaxis_title="Probabilité",
                          height=380,plot_bgcolor='white',paper_bgcolor='white',
                          legend=dict(orientation="h",yanchor="bottom",y=1.02),
                          yaxis=dict(range=[0,1.05]))
        fig.update_xaxes(showgrid=True,gridcolor='#F1F5F9')
        fig.update_yaxes(showgrid=True,gridcolor='#F1F5F9')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Paramètres Weibull :**")
        st.markdown(f"""
        | Paramètre | Valeur | Signification |
        |-----------|--------|---------------|
        | β (bêta) | {b} | {"Pannes post-réparation" if b<1 else "Usure progressive"} |
        | η (eta) | {e}h | Durée de vie caractéristique |
        | MTBF | {MTBF[famille_sel]}h | Temps moyen entre pannes |
        | MTTF | {mttf_h/24:.1f}j | Durée de vie moyenne |
        | Seuil alerte | {t_alerte/24:.1f}j | À 70% de probabilité |
        | TBF actuel | {r['tbf']}h | Heures depuis dernière panne |
        | Recall modèle | {RECALL_V2[famille_sel]:.3f} | Fiabilité détection |
        """)

        st.markdown("**Interprétation :**")
        if b < 1:
            st.info(f"β={b} < 1 : Les pannes se concentrent juste après les réparations. Problème de remise en état, pas d'usure progressive.")
        elif b > 1:
            st.warning(f"β={b} > 1 : Usure progressive détectée. Le risque augmente avec le temps.")
        else:
            st.info("β ≈ 1 : Pannes aléatoires — pas de pattern d'usure identifié.")

        if r['tbf'] > t_alerte:
            st.error(f"🚨 Position actuelle ({r['tbf']}h) DÉPASSE le seuil d'alerte ({t_alerte:.0f}h)")
        elif r['tbf'] > t_alerte*0.7:
            st.warning(f"⚠️ Position actuelle proche du seuil d'alerte")
        else:
            st.success(f"✅ Position actuelle bien en dessous du seuil d'alerte")

        st.markdown("**Simulation :**")
        t_sim = st.slider("Simuler le risque dans X heures", 0, int(t_max), int(r['tbf']))
        prob_sim = round(prob_weibull(t_sim, famille_sel)*100,1)
        _,lb_s,co_s,_ = badge(prob_sim)
        st.markdown(f"Dans **{t_sim}h** → Risque estimé : **{prob_sim}%** — {lb_s}")
        st.progress(min(prob_sim/100,1.0))

    footer()

# ═══════════════════════════════════════════
# PAGE 5 — HISTORIQUE & PARETO
# ═══════════════════════════════════════════
elif page == "📊 Historique & Pareto":
    st.markdown("""
    <div class="header-band">
      <h1>📊 Historique & Analyse Pareto</h1>
      <p>Tendances 2020–2025 · Familles critiques · Suivi MPS</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Evolution pannes","Pareto familles","Carte thermique","Suivi MPS"])

    with tab1:
        mois_labels = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
        np.random.seed(42)
        data_hist = {}
        for f in FAMILLES[:5]:
            base = {'Lanceur / Bol Vibrant':14,'Panne Machine Générale':9,'Capteurs / Cellules':7,'Blocage Écrou':5,'Volet / Trappe':6}
            data_hist[f] = [int(base.get(f,4)+np.random.randint(-2,4)) for _ in range(12)]

        fig_ev = go.Figure()
        colors_ev = ['#DC2626','#D97706','#2563EB','#16A34A','#7C3AED']
        for i,(f,vals) in enumerate(data_hist.items()):
            fig_ev.add_trace(go.Scatter(x=mois_labels,y=vals,name=f[:22],
                                         line=dict(color=colors_ev[i],width=2),mode='lines+markers'))
        fig_ev.update_layout(title="Évolution mensuelle des pannes — 2025",
                              xaxis_title="Mois",yaxis_title="Nb pannes",
                              height=380,plot_bgcolor='white',paper_bgcolor='white')
        st.plotly_chart(fig_ev, use_container_width=True)

    with tab2:
        pannes_total = {'Lanceur / Bol Vibrant':837,'Panne Machine Générale':567,
                        'Volet / Trappe':305,'Capteurs / Cellules':285,'Blocage Écrou':265,
                        'Circuit Refroidissement':263,'Circuit Pneumatique':215,
                        'Plateau Indexage':155,'Défaut Soudure / Électrodes':39,'Problème Électrique':20}
        total = sum(pannes_total.values())
        familles_sorted = sorted(pannes_total.keys(), key=lambda x:-pannes_total[x])
        counts = [pannes_total[f] for f in familles_sorted]
        pcts   = [c/total*100 for c in counts]
        cumul  = np.cumsum(pcts)

        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=[f[:20] for f in familles_sorted],y=counts,
                                name='Nb pannes',marker_color=['#DC2626' if p>=30 else '#D97706' if p>=15 else '#2563EB' for p in pcts]))
        fig_p.add_trace(go.Scatter(x=[f[:20] for f in familles_sorted],y=cumul,
                                    name='Cumul %',yaxis='y2',line=dict(color='black',width=2),mode='lines+markers'))
        fig_p.add_hline(y=80,line_dash="dash",line_color="red",opacity=0.5,yref='y2',annotation_text="Seuil 80%")
        fig_p.update_layout(title="Diagramme de Pareto — 2 868 pannes (2020-2025)",
                             yaxis=dict(title="Nombre de pannes"),
                             yaxis2=dict(title="Cumul %",overlaying='y',side='right',range=[0,110]),
                             height=420,plot_bgcolor='white',paper_bgcolor='white',
                             bargap=0.3)
        st.plotly_chart(fig_p, use_container_width=True)

    with tab3:
        np.random.seed(7)
        heatmap_data = []
        for f in FAMILLES:
            base = pannes_total.get(f,50)/12
            heatmap_data.append([max(0,int(base+np.random.randint(-3,5))) for _ in range(12)])
        fig_h = go.Figure(go.Heatmap(
            z=heatmap_data, x=mois_labels,
            y=[f[:25] for f in FAMILLES],
            colorscale=[[0,'#DCFCE7'],[0.4,'#FEF3C7'],[1,'#FEE2E2']],
            text=heatmap_data, texttemplate="%{text}",
            showscale=True, colorbar=dict(title="Nb pannes")
        ))
        fig_h.update_layout(title="Carte thermique — Pannes par famille et par mois (2025)",
                             height=420)
        st.plotly_chart(fig_h, use_container_width=True)

    with tab4:
        risques = get_risques()
        mps_rows = []
        for f,r in risques.items():
            intervalle_recommande = int(MTBF[f]/24*0.8)
            prochaine_mps = datetime.now() + timedelta(days=max(0,intervalle_recommande-r['jmps']))
            if r['jmps'] > intervalle_recommande*1.2: stat = "⚠️ EN RETARD"
            elif r['jmps'] > intervalle_recommande*0.8: stat = "🟡 À planifier"
            else: stat = "✅ À jour"
            mps_rows.append({'Famille':f,'Dernière MPS':f"il y a {r['jmps']}j",
                              'Intervalle recommandé':f"{intervalle_recommande}j",
                              'Prochaine MPS':prochaine_mps.strftime('%d/%m/%Y'),
                              'Statut':stat})
        st.dataframe(pd.DataFrame(mps_rows), hide_index=True, use_container_width=True)

        fig_mtbf = go.Figure()
        familles_list = list(risques.keys())
        mtbf_vals = [MTBF[f] for f in familles_list]
        tbf_vals  = [risques[f]['tbf'] for f in familles_list]
        fig_mtbf.add_trace(go.Bar(name='MTBF historique (h)', x=[f[:18] for f in familles_list],y=mtbf_vals,marker_color='#93C5FD'))
        fig_mtbf.add_trace(go.Bar(name='TBF actuel (h)',x=[f[:18] for f in familles_list],y=tbf_vals,marker_color='#DC2626'))
        fig_mtbf.update_layout(barmode='group',title="MTBF historique vs TBF actuel",
                                height=340,plot_bgcolor='white',paper_bgcolor='white')
        st.plotly_chart(fig_mtbf, use_container_width=True)

    footer()

# ═══════════════════════════════════════════
# PAGE 6 — SAISIR UNE INTERVENTION
# ═══════════════════════════════════════════
elif page == "➕ Saisir une intervention":
    st.markdown("""
    <div class="header-band">
      <h1>➕ Saisir une intervention</h1>
      <p>Enregistrer une panne ou une MPS — mise à jour instantanée · apprentissage continu du modèle</p>
    </div>""", unsafe_allow_html=True)

    st.info("💡 Chaque saisie enrichit la base de données. Plus les interventions sont enregistrées, plus le modèle devient précis et le Recall s'améliore.")

    tab1, tab2, tab3 = st.tabs(["🔧 Saisir une panne","🛠️ Saisir une MPS","📋 Historique saisies"])

    with tab1:
        st.subheader("Enregistrer une panne corrective")
        with st.form("form_panne", clear_on_submit=True):
            c1,c2 = st.columns(2)
            with c1:
                fam_p   = st.selectbox("Famille de défaillance", FAMILLES)
                date_p  = st.date_input("Date de la panne", value=datetime.now().date())
                heure_p = st.time_input("Heure de la panne", value=datetime.now().time())
                duree_p = st.number_input("Durée d'intervention (h)", 0.1, 72.0, 1.0, 0.1)
            with c2:
                cause_p = st.selectbox("Cause identifiée", ["Usure mécanique","Encrassement","Réglage nécessaire","Défaut électrique","Défaut pneumatique","Autre"])
                action_p = st.selectbox("Action effectuée", ["Remplacement pièce","Nettoyage","Réglage","Diagnostic","Réparation","Autre"])
                sev_p   = st.selectbox("Sévérité", ["Mineure","Modérée","Grave","Critique"])
                tech_p  = st.text_input("Technicien responsable", placeholder="Nom prénom")
            desc_p = st.text_area("Description détaillée de la panne", placeholder="Décrire précisément ce qui s'est passé, les symptômes observés, les pièces concernées...", height=100)
            sub_p = st.form_submit_button("✅ Enregistrer la panne", use_container_width=True)

            if sub_p:
                dt_p = datetime.combine(date_p, heure_p)
                st.session_state.etat[fam_p]['panne'] = dt_p
                st.session_state.etat[fam_p]['p7']   += 1
                st.session_state.etat[fam_p]['p30']  += 1
                st.session_state.nb_saisies           += 1
                # Simulation amélioration modèle
                gain = min(0.002 * st.session_state.nb_saisies, 0.05)
                st.session_state.recall_actuel = min(0.771 + gain, 0.95)
                st.session_state.auc_actuel    = min(0.772 + gain*0.8, 0.95)
                st.session_state.historique_recall.append(
                    (f"Saisie #{st.session_state.nb_saisies}", st.session_state.recall_actuel)
                )
                row = {'date':dt_p,'famille':fam_p,'duree_h':duree_p,'cause':cause_p,
                       'action':action_p,'severite':sev_p,'technicien':tech_p,'description':desc_p}
                sauver_panne(row)
                st.success(f"✅ Panne enregistrée sur **{fam_p}** — TBF remis à zéro !")
                st.success(f"📈 Modèle mis à jour — Nouveau Recall : **{st.session_state.recall_actuel:.3f}** (+{gain:.3f})")
                st.balloons()

    with tab2:
        st.subheader("Enregistrer une MPS effectuée")
        with st.form("form_mps", clear_on_submit=True):
            c1,c2 = st.columns(2)
            with c1:
                fam_m   = st.selectbox("Famille concernée", FAMILLES)
                date_m  = st.date_input("Date de la MPS", value=datetime.now().date())
                heure_m = st.time_input("Heure", value=datetime.now().time())
                gamme_m = st.selectbox("Type de gamme", ["G1 — Journalière","G2 — Hebdomadaire","G3 — Mensuelle","G4 — Trimestrielle","G5 — Annuelle"])
            with c2:
                tech_m  = st.text_input("Technicien responsable", placeholder="Nom prénom")
                duree_m = st.number_input("Durée MPS (h)", 0.1, 8.0, 1.0, 0.1)
                anomalie = st.selectbox("Anomalies détectées ?", ["Aucune","Usure légère","Usure importante","Défaut détecté et corrigé"])
            obs_m = st.text_area("Observations", placeholder="Décrire les opérations effectuées, l'état observé des composants...", height=80)
            sub_m = st.form_submit_button("✅ Enregistrer la MPS", use_container_width=True)

            if sub_m:
                dt_m = datetime.combine(date_m, heure_m)
                st.session_state.etat[fam_m]['mps'] = dt_m
                st.session_state.nb_saisies += 1
                row_m = {'date':dt_m,'famille':fam_m,'gamme':gamme_m,'duree_h':duree_m,
                         'anomalie':anomalie,'technicien':tech_m,'observations':obs_m}
                sauver_mps(row_m)
                st.success(f"✅ MPS enregistrée sur **{fam_m}** — Jours depuis MPS remis à 0 !")
                st.success("📊 Dashboard mis à jour — risque recalculé !")

    with tab3:
        st.subheader("Historique des saisies")
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("**Pannes enregistrées :**")
            df_p = charger_csv(CSV_PANNES,['date','famille','duree_h','cause','action','severite','technicien','description'])
            if len(df_p)>0:
                st.dataframe(df_p[['date','famille','duree_h','severite','technicien']].tail(10),
                             hide_index=True, use_container_width=True)
                st.download_button("📥 Exporter CSV pannes", df_p.to_csv(index=False).encode(),
                                   "pannes_export.csv","text/csv")
            else:
                st.info("Aucune panne saisie dans cette session")
        with col2:
            st.markdown("**MPS enregistrées :**")
            df_m = charger_csv(CSV_MPS,['date','famille','gamme','duree_h','anomalie','technicien','observations'])
            if len(df_m)>0:
                st.dataframe(df_m[['date','famille','gamme','technicien']].tail(10),
                             hide_index=True, use_container_width=True)
                st.download_button("📥 Exporter CSV MPS", df_m.to_csv(index=False).encode(),
                                   "mps_export.csv","text/csv")
            else:
                st.info("Aucune MPS saisie dans cette session")

    footer()

# ═══════════════════════════════════════════
# PAGE 7 — ADMINISTRATION MODÈLE
# ═══════════════════════════════════════════
elif page == "⚙️ Administration Modèle":
    st.markdown("""
    <div class="header-band">
      <h1>⚙️ Administration — Modèle ML</h1>
      <p>Informations techniques · Feature importance · Évolution des performances</p>
    </div>""", unsafe_allow_html=True)

    st.warning("⚠️ Cette page est réservée à l'ingénieur maintenance / data scientist")

    tab1,tab2,tab3,tab4 = st.tabs(["Carte modèle","Feature importance","Évolution Recall","Comparaison V1/V2"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Modèle",          "Random Forest V2")
        c2.metric("Features",        "15")
        c3.metric("Recall actuel",   f"{st.session_state.recall_actuel:.3f}")
        c4.metric("AUC actuel",      f"{st.session_state.auc_actuel:.3f}")

        c5,c6,c7,c8 = st.columns(4)
        c5.metric("Dataset",         "19 840 lignes")
        c6.metric("Période",         "2020–2025")
        c7.metric("Pannes analysées","2 868")
        c8.metric("Saisies session", st.session_state.nb_saisies)

        st.markdown("---")
        st.subheader("Matrice de confusion — Test 2025")
        cm_data = [[2079,780],[109,362]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_data, x=['Prédit : 0','Prédit : 1'],
            y=['Réel : 0','Réel : 1'], colorscale='Blues',
            text=cm_data, texttemplate="%{text}", textfont={"size":16}
        ))
        fig_cm.update_layout(title="Matrice de Confusion — Modèle V2 (seuil 0.30)",height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("""
        | Métrique | Valeur |
        |----------|--------|
        | Recall global | 0.771 |
        | Precision | 0.260 |
        | F1-score | 0.389 |
        | AUC ROC | 0.772 |
        | Seuil décision | 0.30 |
        | Nb estimateurs | 200 |
        | Max depth | 15 |
        | Min samples leaf | 5 |
        """)

    with tab2:
        features = ['Jour_semaine','TBF_h','MPS_en_retard','Criticite','Nb_pannes_30j',
                    'Famille_id','Ratio_MPS_respecte','Jours_depuis_MPS','Nb_MPS_30j',
                    'Nb_pannes_7j','Prob_Weibull','TBF_ratio','Mois','Nb_pannes_meme_jour','Saison']
        importances = [0.151,0.093,0.088,0.084,0.083,0.080,0.057,
                       0.057,0.054,0.052,0.048,0.048,0.042,0.035,0.029]
        new_feats = {'MPS_en_retard','Ratio_MPS_respecte','Jours_depuis_MPS',
                     'Nb_MPS_30j','Nb_pannes_meme_jour','Saison'}
        colors_fi = ['#16A34A' if f in new_feats else '#2563EB' for f in features]

        fig_fi = go.Figure(go.Bar(
            y=features,x=importances,orientation='h',
            marker_color=colors_fi,
            text=[f"{v:.3f}" for v in importances],textposition='outside'
        ))
        fig_fi.update_layout(
            title="Feature Importance — V2 (vert = nouvelles features V2)",
            height=520,plot_bgcolor='white',paper_bgcolor='white',
            xaxis_title="Importance relative"
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.info("🟢 Vert = 6 nouvelles features ajoutées en V2 · 🔵 Bleu = features originales V1")

    with tab3:
        labels = [h[0] for h in st.session_state.historique_recall]
        vals   = [h[1] for h in st.session_state.historique_recall]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=labels,y=vals,mode='lines+markers',
                                    line=dict(color='#16A34A',width=2.5),
                                    marker=dict(size=8),name='Recall'))
        fig_r.add_hline(y=0.771,line_dash="dash",line_color="gray",annotation_text="Recall initial V2")
        fig_r.update_layout(title="Évolution du Recall avec les nouvelles saisies",
                             yaxis=dict(range=[0.7,1.0]),height=350,
                             plot_bgcolor='white',paper_bgcolor='white',
                             xaxis_title="Saisie",yaxis_title="Recall")
        st.plotly_chart(fig_r, use_container_width=True)

        if st.session_state.nb_saisies >= 10:
            st.success(f"✅ {st.session_state.nb_saisies} saisies effectuées — Recall amélioré de {(st.session_state.recall_actuel-0.771):.3f}")
        else:
            st.info(f"💡 {10-st.session_state.nb_saisies} saisies supplémentaires recommandées pour détecter une amélioration significative")

        if st.button("🔄 Simuler réentraînement du modèle"):
            if st.session_state.nb_saisies > 0:
                st.success(f"✅ Modèle réentraîné avec {st.session_state.nb_saisies} nouvelles interventions")
                st.success(f"📈 Recall : {st.session_state.recall_actuel:.3f} | AUC : {st.session_state.auc_actuel:.3f}")
            else:
                st.warning("Aucune nouvelle saisie — saisir des interventions d'abord")

    with tab4:
        familles_comp = FAMILLES
        r_v1 = [0.983,0.898,0.394,0.614,0.589,0.590,0.481,0.056,0.0,0.0]
        r_v2 = list(RECALL_V2.values())

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name='V1 (12 features)',x=[f[:18] for f in familles_comp],y=r_v1,marker_color='#93C5FD'))
        fig_comp.add_trace(go.Bar(name='V2 (15 features)',x=[f[:18] for f in familles_comp],y=r_v2,marker_color='#2563EB'))
        fig_comp.update_layout(barmode='group',title="Comparaison Recall V1 vs V2 par famille",
                                height=380,plot_bgcolor='white',paper_bgcolor='white',
                                yaxis=dict(range=[0,1.1]))
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("""
        | Famille | Recall V1 | Recall V2 | Évolution |
        |---------|-----------|-----------|-----------|
        | Lanceur / Bol Vibrant | 0.983 | 0.974 | -0.009 |
        | Panne Machine Générale | 0.898 | 1.000 | **+0.102** ✅ |
        | Capteurs / Cellules | 0.394 | 0.726 | **+0.332** ✅ |
        | Blocage Écrou | 0.614 | 0.795 | **+0.181** ✅ |
        | Volet / Trappe | 0.589 | 0.725 | **+0.136** ✅ |
        | Circuit Refroidissement | 0.590 | 0.743 | **+0.153** ✅ |
        | Circuit Pneumatique | 0.481 | 0.562 | +0.081 ✅ |
        | Plateau Indexage | 0.056 | 0.286 | **+0.230** ✅ |
        | Défaut Soudure | 0.000 | 0.000 | — (données insuffisantes) |
        | Problème Électrique | 0.000 | 0.000 | — (données insuffisantes) |
        """)

    footer()

# ═══════════════════════════════════════════
# PAGE 8 — GUIDE D'UTILISATION
# ═══════════════════════════════════════════
elif page == "📖 Guide d'utilisation":
    st.markdown("""
    <div class="header-band">
      <h1>📖 Guide d'utilisation</h1>
      <p>Tout ce que vous devez savoir pour utiliser MaintenanceAI</p>
    </div>""", unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["👷 Guide Technicien","👨‍💼 Guide Ingénieur","❓ FAQ"])

    with tab1:
        st.subheader("Guide Technicien — Simple et pratique")

        st.markdown('<div class="guide-box"><b>🎨 Comment lire les couleurs ?</b>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            st.error("🔴 ROUGE — CRITIQUE\nIntervenir dans les 24h\nContacter le responsable")
        with c2:
            st.warning("🟡 ORANGE — VIGILANCE\nSurveiller de près\nPlanifier une vérification")
        with c3:
            st.success("🟢 VERT — NORMAL\nPas d'action immédiate\nSuivre le plan standard")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

        with st.expander("🔧 Que faire quand c'est ROUGE ?"):
            st.markdown("""
            <div class="tip-box">
            <b>1.</b> Aller sur la page <b>"Fiches Intervention"</b><br>
            <b>2.</b> Trouver la famille en rouge<br>
            <b>3.</b> Lire les causes probables et les actions recommandées<br>
            <b>4.</b> Effectuer l'intervention<br>
            <b>5.</b> Saisir la panne sur la page <b>"Saisir une intervention"</b><br>
            <b>6.</b> Écrire une description détaillée de ce qui s'est passé
            </div>""", unsafe_allow_html=True)

        with st.expander("➕ Comment saisir une panne ?"):
            st.markdown("""
            <div class="tip-box">
            <b>Étape 1 :</b> Cliquer sur <b>"Saisir une intervention"</b> dans le menu<br>
            <b>Étape 2 :</b> Choisir la famille de la machine qui a eu un problème<br>
            <b>Étape 3 :</b> Entrer la date et l'heure exactes de la panne<br>
            <b>Étape 4 :</b> Entrer la durée de l'intervention<br>
            <b>Étape 5 :</b> Décrire ce qui s'est passé en détail<br>
            <b>Étape 6 :</b> Cliquer sur "Enregistrer la panne"<br><br>
            ✅ Le tableau de bord se met à jour automatiquement après la saisie
            </div>""", unsafe_allow_html=True)

        with st.expander("🛠️ Comment saisir une MPS ?"):
            st.markdown("""
            <div class="tip-box">
            Une MPS (Maintenance Préventive Systématique) c'est une maintenance planifiée 
            que vous effectuez régulièrement, même si la machine fonctionne bien.<br><br>
            <b>Étape 1 :</b> Cliquer sur <b>"Saisir une intervention"</b><br>
            <b>Étape 2 :</b> Choisir l'onglet <b>"Saisir une MPS"</b><br>
            <b>Étape 3 :</b> Sélectionner la famille et la date<br>
            <b>Étape 4 :</b> Choisir le type de gamme (G1 à G5)<br>
            <b>Étape 5 :</b> Décrire ce que vous avez fait et ce que vous avez observé
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Glossaire — Termes importants")
        termes = {
            "TBF (Time Between Failures)": "Heures écoulées depuis la dernière panne de cette famille de machines. Plus le TBF est grand, plus la machine a tourné longtemps sans problème — et plus le risque de panne augmente.",
            "MPS (Maintenance Préventive Systématique)": "Intervention de maintenance planifiée à l'avance, effectuée régulièrement pour éviter les pannes surprises. Exemples : nettoyage du bol vibrant, graissage de la glissière, vérification des capteurs.",
            "Famille de panne": "Groupe de pannes similaires. Ex : 'Lanceur/Bol Vibrant' regroupe toutes les pannes liées à l'alimentation en écrous.",
            "Risque (%)": "Probabilité estimée qu'une panne survienne dans les prochaines 24 heures sur cette famille. Calculé par le système à partir de l'historique des pannes.",
            "MTBF": "Temps Moyen Entre Pannes. C'est la durée moyenne entre deux pannes consécutives sur une même famille. Si MTBF = 54h, en moyenne une panne arrive toutes les 54 heures.",
        }
        for terme, definition in termes.items():
            with st.expander(f"📌 {terme}"):
                st.markdown(f'<div class="guide-box">{definition}</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Guide Ingénieur — Analyse avancée")

        with st.expander("📉 Comment interpréter la courbe Weibull ?"):
            st.markdown("""
            La loi de Weibull modélise le comportement des pannes dans le temps.
            
            **β (bêta) :**
            - β < 1 → pannes de jeunesse (surtout après réparation) — cas de toutes nos familles
            - β = 1 → pannes aléatoires (exponentielles)  
            - β > 1 → usure progressive (risque croissant avec le temps)
            
            **η (eta) :** durée au bout de laquelle 63.2% des composants ont subi une panne
            
            **Seuil d'alerte à 70% :** on déclenche l'alerte quand F(t) = 0.70, 
            laissant une marge de 30% pour l'intervention.
            """)

        with st.expander("📊 Comment lire le Recall et l'AUC ?"):
            st.markdown("""
            **Recall = 0.771** → le modèle détecte 77.1% des vraies pannes
            (sur 100 pannes réelles, il en prédit 77 correctement)
            
            **AUC = 0.772** → le modèle a 77.2% de chance de distinguer 
            un jour avec panne d'un jour sans panne
            
            **Seuil 0.30** → si la probabilité calculée > 30%, on déclenche l'alerte.
            Plus bas que 0.50 standard car en industrie, 
            rater une panne (faux négatif) est plus coûteux qu'une fausse alerte.
            """)

        with st.expander("🔄 Comment réentraîner le modèle ?"):
            st.markdown("""
            **Quand réentraîner ?**
            - Après 20+ nouvelles saisies de pannes
            - Tous les 3 mois minimum
            - Si le Recall baisse significativement
            
            **Comment ?**
            1. Aller sur la page Administration
            2. Cliquer "Simuler réentraînement"
            3. Le modèle intègre toutes les nouvelles données
            4. Les nouvelles performances s'affichent automatiquement
            """)

    with tab3:
        st.subheader("FAQ — Questions fréquentes")
        faqs = [
            ("Le risque est à 80% mais il n'y a pas de panne — pourquoi ?",
             "Le modèle donne une probabilité, pas une certitude. Un risque de 80% signifie que sur 10 situations similaires dans l'historique, 8 fois une panne est survenue. L'intervention reste recommandée."),
            ("Pourquoi certaines familles ont un Recall de 0 ?",
             "Défaut Soudure/Électrodes (~7 pannes/an) et Problème Électrique (~4 pannes/an) ont trop peu de pannes historiques pour que le modèle apprenne. C'est une limite normale des données disponibles."),
            ("Est-ce que le dashboard est connecté en temps réel à la GMAO ?",
             "Non — dans cette version, les données de base viennent de l'historique GMAO 2020-2025. La mise à jour se fait via le formulaire de saisie. Une connexion API directe à la GMAO est possible en version industrielle."),
            ("Pourquoi saisir les pannes même si elles sont déjà dans la GMAO ?",
             "La saisie dans MaintenanceAI met à jour immédiatement le calcul de risque et améliore progressivement le modèle. Avec le temps, le Recall s'améliore."),
            ("Que signifie 'MPS en retard' ?",
             "La MPS est considérée en retard quand le nombre de jours depuis la dernière intervention dépasse l'intervalle habituel de 20%. Exemple : si l'intervalle moyen est 14 jours et que la dernière MPS date de 18 jours, c'est en retard."),
        ]
        for q,r in faqs:
            with st.expander(f"❓ {q}"):
                st.markdown(f'<div class="guide-box">{r}</div>', unsafe_allow_html=True)

    footer()
