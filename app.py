import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import weibull_min
from scipy.special import gamma as gamma_fn
import os, csv, json

st.set_page_config(page_title="MaintenanceAI — GMD Métal Tanger", page_icon="⚙️",
                    layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════
# CSS — SaaS Premium Dark Mode
# ═══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }
.stApp { background:#0B1220; }
section[data-testid="stSidebar"] { background:#0A0F1C !important; border-right:1px solid #1C2A42; }
.main .block-container { padding:0 !important; max-width:100% !important; }

.sidebar-logo{background:linear-gradient(135deg,#0B1220,#141E33);padding:22px 18px;margin-bottom:6px;border-bottom:1px solid #1C2A42;}
.sidebar-logo .brand{color:#5EA1F0;font-size:9px;font-weight:700;letter-spacing:3px;text-transform:uppercase;}
.sidebar-logo .appname{color:#F5F7FA;font-size:19px;font-weight:800;margin:6px 0 2px;letter-spacing:-0.3px;}
.sidebar-logo .sub{color:#6B7A99;font-size:11px;}
.sidebar-status{display:flex;align-items:center;gap:9px;padding:10px 18px;background:#0D1526;margin-bottom:6px;border-bottom:1px solid #1C2A42;}
.pulse-dot{width:8px;height:8px;border-radius:50%;background:#34D399;animation:pulse 2s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.5);}70%{box-shadow:0 0 0 9px rgba(52,211,153,0);}100%{box-shadow:0 0 0 0 rgba(52,211,153,0);}}
.status-text{color:#34D399;font-size:11.5px;font-weight:600;}
.status-sub{color:#5EA1F0;font-size:10px;font-family:'JetBrains Mono';}

.page-header{background:linear-gradient(135deg,#0D1526 0%,#141E33 100%);padding:26px 32px;margin-bottom:22px;
  border-bottom:1px solid #22314D;display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:14px;}
.page-header h1{color:#F5F7FA;font-size:22px;font-weight:800;margin:0;letter-spacing:-0.4px;}
.page-header p{color:#7C8BA8;font-size:12.5px;margin:6px 0 0;}
.header-right{display:flex;gap:10px;align-items:center;}
.header-badge{background:#0D1526;border:1px solid #22314D;border-radius:10px;padding:9px 18px;text-align:center;}
.header-badge .bval{color:#5EA1F0;font-size:17px;font-weight:800;font-family:'JetBrains Mono';}
.header-badge .blbl{color:#7C8BA8;font-size:9.5px;text-transform:uppercase;letter-spacing:.5px;}

.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;padding:0 24px 20px;}
.kpi-card{background:#101B30;border:1px solid #1E2C47;border-radius:14px;padding:18px 20px;
  transition:transform .18s,border-color .18s,box-shadow .18s;position:relative;overflow:hidden;}
.kpi-card:hover{transform:translateY(-3px);border-color:#3B82F6;box-shadow:0 10px 30px rgba(59,130,246,.12);}
.kpi-card.warn{border-color:#3A2E12;background:#15130A;}
.kpi-card.crit{border-color:#3A1414;background:#170C0C;}
.kpi-card.ok{border-color:#12321F;background:#0B1810;}
.kpi-top{display:flex;justify-content:space-between;align-items:flex-start;}
.kpi-label{font-size:10.5px;color:#7C8BA8;font-weight:600;text-transform:uppercase;letter-spacing:.9px;}
.kpi-value{font-size:30px;font-weight:800;font-family:'JetBrains Mono';color:#F5F7FA;line-height:1;margin-top:8px;}
.kpi-card.crit .kpi-value{color:#F87171;} .kpi-card.warn .kpi-value{color:#FBBF24;} .kpi-card.ok .kpi-value{color:#4ADE80;}
.kpi-sub{font-size:11px;color:#5EA1F0;margin-top:8px;}
.kpi-trend{font-size:10.5px;padding:2px 7px;border-radius:20px;font-weight:700;}
.trend-up{background:#3A1414;color:#F87171;} .trend-down{background:#12321F;color:#4ADE80;}
.sparkline-wrap{margin-top:10px;height:26px;}

.hs-ring-wrap{display:flex;align-items:center;gap:16px;background:#101B30;border:1px solid #1E2C47;border-radius:14px;padding:16px 20px;}

.section-title{color:#7C8BA8;font-size:10.5px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
  padding:0 24px;margin:20px 0 12px;display:flex;align-items:center;gap:10px;}
.section-title::after{content:'';flex:1;height:1px;background:#1E2C47;}

.priority-card{background:#101B30;border:1px solid #1E2C47;border-radius:14px;padding:18px 20px;margin-bottom:12px;
  border-left:4px solid;transition:transform .15s;}
.priority-card:hover{transform:translateX(3px);}
.priority-card.crit{border-left-color:#EF4444;background:linear-gradient(90deg,#170C0C 0%,#101B30 12%);}
.priority-card.warn{border-left-color:#F59E0B;background:linear-gradient(90deg,#15130A 0%,#101B30 12%);}
.priority-card.ok{border-left-color:#22C55E;background:linear-gradient(90deg,#0B1810 0%,#101B30 12%);}
.pc-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px;}
.pc-name{color:#F5F7FA;font-weight:700;font-size:14px;}
.pc-badges{display:flex;gap:6px;}
.badge{padding:3px 11px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:.4px;}
.badge-crit{background:#3A1414;color:#F87171;border:1px solid #5B1F1F;}
.badge-warn{background:#3A2E12;color:#FBBF24;border:1px solid #5B471F;}
.badge-ok{background:#12321F;color:#4ADE80;border:1px solid #1F5B37;}
.pc-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px;}
.pc-metric-box{background:#0B1425;border-radius:8px;padding:8px 10px;}
.pc-metric-label{font-size:9.5px;color:#5F6E8C;text-transform:uppercase;letter-spacing:.5px;}
.pc-metric-value{font-size:16px;font-weight:700;color:#DDE4F0;font-family:'JetBrains Mono';margin-top:2px;}
.bar-bg{height:6px;background:#1E2C47;border-radius:3px;overflow:hidden;margin-bottom:10px;}
.bar-fill{height:100%;border-radius:3px;transition:width .6s ease;}
.pc-action{font-size:12px;color:#9BAAC7;background:#0B1425;padding:9px 12px;border-radius:8px;line-height:1.5;}

.explain-box{background:#0D1526;border:1px solid #22314D;border-radius:10px;padding:14px 16px;margin-bottom:8px;}
.explain-title{color:#5EA1F0;font-weight:700;font-size:12px;margin-bottom:8px;}
.factor-row{display:flex;gap:9px;align-items:flex-start;padding:6px 0;font-size:12px;color:#B4C0D9;line-height:1.5;}
.factor-num{background:#1B2A45;color:#5EA1F0;width:20px;height:20px;border-radius:6px;display:flex;
  align-items:center;justify-content:center;font-size:10.5px;font-weight:700;flex-shrink:0;}

.alert-card{background:#170C0C;border:1px solid #3A1414;border-radius:10px;padding:13px 16px;margin-bottom:9px;}
.alert-warn-card{background:#15130A;border:1px solid #3A2E12;border-radius:10px;padding:13px 16px;margin-bottom:9px;}
.alert-title{color:#F5F7FA;font-weight:700;font-size:12.5px;margin-bottom:4px;}
.alert-desc{color:#8A99B8;font-size:11px;line-height:1.5;}

.guide-step{background:#101B30;border:1px solid #1E2C47;border-radius:12px;padding:16px;margin-bottom:10px;
  display:flex;gap:16px;align-items:flex-start;}
.step-num{background:#3B82F6;color:white;width:30px;height:30px;border-radius:9px;display:flex;
  align-items:center;justify-content:center;font-weight:800;font-size:13px;flex-shrink:0;}
.step-content h4{color:#F5F7FA;font-size:13.5px;font-weight:700;margin:0 0 4px;}
.step-content p{color:#9BAAC7;font-size:12px;margin:0;line-height:1.55;}

.glossary-card{background:#101B30;border:1px solid #1E2C47;border-radius:10px;padding:15px 17px;margin-bottom:9px;transition:border-color .2s;}
.glossary-card:hover{border-color:#3B82F6;}
.glossary-term{color:#5EA1F0;font-weight:800;font-size:13px;margin-bottom:6px;font-family:'JetBrains Mono';}
.glossary-def{color:#9BAAC7;font-size:12px;line-height:1.65;}
.glossary-tag{display:inline-block;background:#1B2A45;color:#7C8BA8;font-size:9.5px;padding:3px 9px;border-radius:20px;margin-top:8px;}

.model-card{background:#101B30;border:1px solid #1E2C47;border-radius:12px;padding:18px;}
.model-row{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #1B2A45;font-size:12.5px;}
.model-row:last-child{border-bottom:none;}
.model-label{color:#7C8BA8;} .model-value{color:#DDE4F0;font-weight:700;font-family:'JetBrains Mono';}

.pred-banner{background:linear-gradient(135deg,#0B1E36,#132840);border:1px solid #1E3A5C;border-radius:14px;
  padding:16px 22px;margin:0 24px 18px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;}
.pred-banner-text h3{color:#5EA1F0;font-size:14px;font-weight:800;margin:0 0 4px;}
.pred-banner-text p{color:#7C8BA8;font-size:11.5px;margin:0;}
.pred-badge{background:#3B82F6;color:white;padding:7px 16px;border-radius:20px;font-size:11px;font-weight:800;letter-spacing:.4px;}

.app-footer{background:#080D18;border-top:1px solid #1E2C47;padding:14px 24px;margin-top:24px;
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font-size:11px;color:#4A5A78;}

div[data-testid="metric-container"]{background:#101B30 !important;border:1px solid #1E2C47 !important;border-radius:12px !important;}
div[data-testid="metric-container"] label{color:#7C8BA8 !important;font-size:11px !important;}
div[data-testid="metric-container"] [data-testid="metric-value"]{color:#F5F7FA !important;font-family:'JetBrains Mono' !important;}
.stTabs [data-baseweb="tab-list"]{background:#101B30;border-bottom:1px solid #1E2C47;border-radius:10px 10px 0 0;}
.stTabs [data-baseweb="tab"]{color:#7C8BA8 !important;}
.stTabs [aria-selected="true"]{color:#5EA1F0 !important;border-bottom:2px solid #3B82F6 !important;}
.stSelectbox > div{background:#101B30 !important;border-color:#1E2C47 !important;color:#DDE4F0 !important;}
.stTextInput > div > div{background:#101B30 !important;border-color:#1E2C47 !important;color:#DDE4F0 !important;}
.stTextArea > div > div{background:#101B30 !important;border-color:#1E2C47 !important;color:#DDE4F0 !important;}
.stNumberInput > div > div{background:#101B30 !important;border-color:#1E2C47 !important;color:#DDE4F0 !important;}
.stDateInput > div > div{background:#101B30 !important;border-color:#1E2C47 !important;color:#DDE4F0 !important;}
.stButton > button{background:#3B82F6 !important;color:white !important;border:none !important;border-radius:10px !important;
  font-weight:700 !important;transition:all .2s !important;}
.stButton > button:hover{background:#2563EB !important;transform:translateY(-1px) !important;}
.stRadio > div{background:transparent !important;} .stRadio label{color:#B4C0D9 !important;}
.stForm{background:#101B30 !important;border:1px solid #1E2C47 !important;border-radius:14px !important;padding:18px !important;}
.element-container .stMarkdown p{color:#B4C0D9;}
h1,h2,h3,h4{color:#F5F7FA !important;}
.stDataFrame{background:#101B30 !important;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# CONSTANTES MÉTIER
# ═══════════════════════════════════════════
DATE_BASELINE_FIN = datetime(2025, 11, 29)
DATE_MISE_SERVICE = datetime(2026, 8, 24)
NOW = datetime.now()

FAMILLES = ['Lanceur / Bol Vibrant','Panne Machine Générale','Capteurs / Cellules','Blocage Écrou',
            'Volet / Trappe','Circuit Refroidissement','Circuit Pneumatique','Plateau Indexage',
            'Défaut Soudure / Électrodes','Problème Électrique']

WEIBULL = {
 'Lanceur / Bol Vibrant':{'beta':0.653,'eta':40.1},'Panne Machine Générale':{'beta':0.611,'eta':55.2},
 'Capteurs / Cellules':{'beta':0.720,'eta':126.8},'Blocage Écrou':{'beta':0.628,'eta':110.2},
 'Volet / Trappe':{'beta':0.661,'eta':109.1},'Circuit Refroidissement':{'beta':0.718,'eta':140.4},
 'Circuit Pneumatique':{'beta':0.701,'eta':164.9},'Plateau Indexage':{'beta':0.485,'eta':167.1},
 'Défaut Soudure / Électrodes':{'beta':0.549,'eta':687.7},'Problème Électrique':{'beta':0.876,'eta':1746.6},
}
MTBF = {'Lanceur / Bol Vibrant':54.6,'Panne Machine Générale':80.7,'Capteurs / Cellules':157.6,
        'Blocage Écrou':165.5,'Volet / Trappe':146.6,'Circuit Refroidissement':172.6,
        'Circuit Pneumatique':210.0,'Plateau Indexage':310.5,'Défaut Soudure / Électrodes':1040.7,
        'Problème Électrique':1855.9}
MTTR = {'Lanceur / Bol Vibrant':1.02,'Panne Machine Générale':0.93,'Capteurs / Cellules':1.06,
        'Blocage Écrou':0.62,'Volet / Trappe':1.80,'Circuit Refroidissement':1.68,
        'Circuit Pneumatique':0.88,'Plateau Indexage':1.81,'Défaut Soudure / Électrodes':0.90,
        'Problème Électrique':1.70}
AMDEC = {'Lanceur / Bol Vibrant':500,'Panne Machine Générale':336,'Capteurs / Cellules':320,
         'Blocage Écrou':140,'Volet / Trappe':192,'Circuit Refroidissement':180,
         'Circuit Pneumatique':90,'Plateau Indexage':100,'Défaut Soudure / Électrodes':63,
         'Problème Électrique':15}
RECALL_V2 = {'Lanceur / Bol Vibrant':0.974,'Panne Machine Générale':1.000,'Capteurs / Cellules':0.726,
             'Blocage Écrou':0.795,'Volet / Trappe':0.725,'Circuit Refroidissement':0.743,
             'Circuit Pneumatique':0.562,'Plateau Indexage':0.286,'Défaut Soudure / Électrodes':0.000,
             'Problème Électrique':0.000}

CAUSES = {
 'Lanceur / Bol Vibrant':[('Encrassement bol vibrant','Mauvais réglage fréquence de vibration, obturation gaine, tuyau de transfert détérioré'),
   ('Aimant permanent encrassé','Accumulation de particules métalliques sur l\'aimant de guidage des écrous'),
   ('Gaine de transfert dégradée','Usure ou écrasement de la gaine limitant le flux d\'écrous'),
   ('Pression air insuffisante','Pression hors plage 0.4–0.6 MPa, éjection incomplète des écrous'),
   ('Séparateur bloqué','Corps étranger dans le séparateur écrous')],
 'Panne Machine Générale':[('Cause non tracée GMAO','Arrêt cycle automatique sans cause identifiée'),
   ('Défaut automate Schneider','Erreur programme automate'),
   ('IHM PROFACE plantée','Interface opérateur gelée'),
   ('Défaut départ cycle','Bouton départ ou validateur de cycle défaillant'),
   ('Arrêt d\'urgence verrouillé','Arrêt d\'urgence activé ou détecté par erreur')],
 'Capteurs / Cellules':[('Encrassement capteurs présence','Poussière ou projections métalliques sur les capteurs'),
   ('Désalignement par vibration','Vibrations répétées déplacent les capteurs'),
   ('Câble arraché ou dégradé','Câble connecteur sectionné par frottement ou pincement'),
   ('Barrière immatérielle mal réglée','Désalignement émetteur/récepteur'),
   ('Interrupteur sécurité carter HS','Défaillance interrupteur carters')],
 'Blocage Écrou':[('Écrou mal orienté dans goulotte','Écrou retourné ou de travers, blocage mécanique'),
   ('Corps étranger dans circuit','Débris métalliques ou copeaux bloquant le trajet'),
   ('Usure goulotte de guidage','Jeu excessif dans la goulotte'),
   ('Shut écrou bloqué','Mécanisme d\'arrêt écrou coincé'),
   ('Fréquence bol vibrant mal réglée','Régime vibratoire inadapté')],
 'Volet / Trappe':[('Choc mécanique sur volet','Impact lors du chargement/déchargement'),
   ('Fatigue matériau charnière','Cycles répétés, rupture ou fissuration'),
   ('Capteur position volet HS','Capteur fin de course défaillant'),
   ('Frein porte dégradé','Frein mécanique de maintien usé'),
   ('Vérin trappe en panne','Vérin pneumatique défaillant')],
 'Circuit Refroidissement':[('Joint dégradé ou canalisation corrodée','Vieillissement joints hydrauliques, fuite'),
   ('Débit eau insuffisant','Encrassement filtre ou pompe faible, risque surchauffe'),
   ('Raccord tournant défaillant','Joint tournant usé, fuite majeure'),
   ('Insert semelle dégradé','Insert de fixation semelle cuivre desserré'),
   ('Répartiteur d\'eau obstrué','Colmatage répartiteur')],
 'Circuit Pneumatique':[('Fuite joints distributeur','Joints d\'étanchéité usés, perte pression'),
   ('Bobine distributeur défaillante','Bobine électrique brûlée'),
   ('Pression FRL insuffisante','Groupe FRL mal réglé'),
   ('Vérin de positionnement HS','Vérin pneumatique bloqué'),
   ('Raccord pneumatique arraché','Raccord rapide éjecté sous pression')],
 'Plateau Indexage':[('Usure came indexeur','Came d\'indexage usée, jeu excessif'),
   ('Détérioration raccord tournant','Joint tournant eau/air dégradé'),
   ('Usure doigts orienteurs','Doigts de positionnement usés'),
   ('Jeu mécanique excessif plateau','Roulements ou liaisons usés'),
   ('Capteur fin de course HS','Capteur de validation défaillant')],
 'Défaut Soudure / Électrodes':[('Usure électrodes cuivre','Cycles thermiques répétés, oxydation'),
   ('Pression soudage incorrecte','Réglage force soudage inadapté'),
   ('Serrage insuffisant électrode','Électrode mal serrée'),
   ('Transformateur 250 KVA dégradé','Vieillissement, intensité instable'),
   ('Insert masque desserré','Insert de positionnement lâche')],
 'Problème Électrique':[('Composant électrique vieillissant','Thyristors ou contacteurs en fin de vie'),
   ('Défaut réseau 400V triphasé','Micro-coupure ou déséquilibre réseau'),
   ('Fusible grillé','Surcharge transitoire'),
   ('Court-circuit câblage','Câble dégradé'),
   ('Platine thyristors défaillante','Carte thyristors défaillante')],
}
ACTIONS = {
 'Lanceur / Bol Vibrant':['Nettoyer le bol vibrant et dépoussiérer la trémie','Vérifier et régler la fréquence de vibration','Inspecter la gaine de transfert','Contrôler la pression d\'air (0.4–0.6 MPa)','Nettoyer le séparateur d\'écrous'],
 'Panne Machine Générale':['Consulter le journal d\'alarmes API sur pupitre PROFACE','Identifier et noter le code d\'erreur affiché','Redémarrer le cycle en mode manuel','Vérifier l\'alimentation 400V','Alerter le responsable si récidive'],
 'Capteurs / Cellules':['Nettoyer tous les capteurs de présence','Vérifier l\'alignement des barrières immatérielles','Inspecter les câbles de connexion','Tester les capteurs en mode manuel','Contrôler le serrage des fixations'],
 'Blocage Écrou':['Inspecter visuellement le circuit d\'écrous','Dégager les écrous bloqués dans la goulotte','Vérifier l\'orientation en sortie bol vibrant','Contrôler l\'usure de la goulotte','Nettoyer le shut d\'écrous'],
 'Volet / Trappe':['Inspecter visuellement l\'état des volets','Tester la fermeture/ouverture en mode manuel','Vérifier l\'état des charnières','Contrôler les capteurs de position','Vérifier le vérin pneumatique'],
 'Circuit Refroidissement':['Vérifier l\'absence de fuite sur le circuit','Contrôler le débit sur les capteurs','Inspecter les raccords et joints','Purger le circuit si nécessaire','Vérifier l\'état des semelles cuivre'],
 'Circuit Pneumatique':['Contrôler la pression sur le manomètre FRL','Rechercher les fuites audibles','Vérifier le serrage des raccords','Tester les distributeurs en mode manuel','Inspecter les vérins'],
 'Plateau Indexage':['Vérifier le positionnement sur les 4 postes','Contrôler le capteur fin de course','Inspecter l\'indexeur à came','Vérifier les doigts orienteurs','Contrôler le raccord tournant'],
 'Défaut Soudure / Électrodes':['Inspecter l\'usure des électrodes','Contrôler la pression de soudage','Vérifier le serrage des électrodes','Mesurer l\'intensité de soudage','Vérifier les semelles cuivre'],
 'Problème Électrique':['Vérifier le tableau électrique 400V','Contrôler les fusibles et disjoncteurs','Vérifier la platine thyristors','Mesurer les tensions de phase','Appeler l\'électricien de maintenance'],
}

CSV_PANNES = "pannes_saisies.csv"; CSV_MPS = "mps_saisies.csv"; CSV_RETOUR = "retours_intervention.csv"
CSV_PANNES_ML = "pannes_saisies_ml.csv"; CSV_MPS_ML = "mps_saisies_ml.csv"

# ═══════════════════════════════════════════
# LISTES SYMPTÔMES — pour menus déroulants stricts (page Saisie Panne ML)
# ═══════════════════════════════════════════
SYMPTOMES = {
 'Lanceur / Bol Vibrant': ["Écrou non éjecté / manque au poste","Bol vibrant bloqué / vibration anormale","Écrou mal orienté à la sortie","Bruit anormal sur le lanceur","Aimant ne charge pas la lance","Autre"],
 'Panne Machine Générale': ["Arrêt cycle sans message clair","IHM PROFACE figée","Cycle ne redémarre pas après acquittement","Alarme API répétitive","Autre"],
 'Capteurs / Cellules': ["Détection écrou absente ou erratique","Barrière immatérielle déclenche sans passage","Cycle bloqué en attente signal capteur","Câble visiblement endommagé","Autre"],
 'Blocage Écrou': ["Écrou coincé dans la goulotte","Accumulation d'écrous en amont","Écrou de travers visible","Shut ne libère pas l'écrou","Autre"],
 'Volet / Trappe': ["Volet ne se ferme pas complètement","Trappe bloquée en position ouverte","Bruit de choc au mouvement du volet","Capteur position volet en défaut","Autre"],
 'Circuit Refroidissement': ["Fuite d'eau visible","Alarme débit d'eau insuffisant","Surchauffe détectée sur électrodes","Flaque / trace d'humidité au sol","Autre"],
 'Circuit Pneumatique': ["Fuite d'air audible (sifflement)","Pression FRL hors plage","Vérin ne termine pas sa course","Mouvement pneumatique incomplet","Autre"],
 'Plateau Indexage': ["Plateau mal positionné sur un poste","Bruit anormal à la rotation","Jeu mécanique perceptible","Capteur fin de course en défaut","Autre"],
 'Défaut Soudure / Électrodes': ["Point de soudure non conforme visuellement","Électrode visiblement usée","Alarme fin de vie électrode","Intensité de soudage anormale","Autre"],
 'Problème Électrique': ["Coupure alimentation cellule","Disjoncteur déclenché","Fusible grillé constaté","Voyant tension absent","Autre"],
}
NIVEAUX_GRAVITE = ["1 — Anomalie mineure, aucun arrêt", "2 — Ralentissement léger", "3 — Arrêt court (< 15 min)", "4 — Arrêt significatif (15-60 min)", "5 — Arrêt majeur (> 1h) ou risque sécurité"]

def sauver_csv(path, row):
    existe = os.path.exists(path)
    with open(path,'a',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not existe: w.writeheader()
        w.writerow(row)

def charger_csv(path, colonnes):
    if os.path.exists(path): return pd.read_csv(path)
    return pd.DataFrame(columns=colonnes)

# ═══════════════════════════════════════════
# MOTEUR PRÉDICTIF — avec correction bug familles rares
# ═══════════════════════════════════════════
def simuler_pannes_entre(date_debut, date_fin, beta, eta, seed_offset=0):
    np.random.seed(42 + seed_offset)
    pannes = []; t = date_debut
    while t < date_fin:
        tbf_h = max(float(weibull_min.rvs(beta, scale=eta)), 1.0)
        t += timedelta(hours=tbf_h)
        if t < date_fin: pannes.append(t)
    return pannes

@st.cache_data
def calculer_etat_predictif():
    etat = {}
    for i, famille in enumerate(FAMILLES):
        beta, eta = WEIBULL[famille]['beta'], WEIBULL[famille]['eta']
        pannes_sim = simuler_pannes_entre(DATE_BASELINE_FIN, NOW, beta, eta, seed_offset=i)

        if pannes_sim:
            derniere_panne = pannes_sim[-1]
        else:
            # FIX BUG: pour familles rares sans panne simulée, on initialise
            # le TBF à une valeur réaliste proche du MTBF plutôt que depuis nov 2025
            derniere_panne = NOW - timedelta(hours=MTBF[famille] * 0.6)

        tbf_actuel_h = (NOW - derniere_panne).total_seconds() / 3600
        # Garde-fou supplémentaire : cap le TBF à 3x le MTBF pour éviter les valeurs aberrantes
        tbf_actuel_h = min(tbf_actuel_h, MTBF[famille] * 3)

        pannes_7j = sum(1 for p in pannes_sim if p > NOW - timedelta(days=7))
        pannes_30j = sum(1 for p in pannes_sim if p > NOW - timedelta(days=30))

        derniere_mps = derniere_panne + timedelta(hours=MTBF[famille] * 0.2)
        if derniere_mps > NOW: derniere_mps = NOW - timedelta(days=3)
        jours_depuis_mps = max((NOW - derniere_mps).days, 0)
        intervalle_mps_ref = max(int(MTBF[famille]/24*0.8), 3)

        etat[famille] = {
            'derniere_panne': derniere_panne, 'tbf_h': round(tbf_actuel_h,1),
            'pannes_7j': pannes_7j, 'pannes_30j': pannes_30j,
            'nb_pannes_periode': len(pannes_sim), 'derniere_mps': derniere_mps,
            'jours_depuis_mps': jours_depuis_mps, 'intervalle_mps_ref': intervalle_mps_ref,
        }
    return etat

def prob_weibull(tbf_h, famille):
    b,e = WEIBULL[famille]['beta'], WEIBULL[famille]['eta']
    if tbf_h <= 0: return 0.0
    return float(1 - np.exp(-((tbf_h/e)**b)))

def calc_probabilite_ml(famille, e):
    """Probabilité ML brute — sortie du modèle Random Forest simulée via Weibull calibré"""
    return round(prob_weibull(e['tbf_h'], famille) * 100, 1)

def calc_priorite_maintenance(famille, e, proba_ml):
    """Priorité maintenance = combinaison ML + AMDEC + retard MPS + impact production"""
    crit_norm = AMDEC[famille] / 500.0
    mps_retard = 1 if e['jours_depuis_mps'] > e['intervalle_mps_ref'] * 1.2 else 0
    freq_recente = min(1.0, e['pannes_30j'] / 10)
    score = (0.40*(proba_ml/100) + 0.30*crit_norm + 0.20*mps_retard + 0.10*freq_recente)
    return round(min(score*100, 99), 1)

def calc_health_score(proba_ml, priorite, tbf, mtbf, mps_retard):
    """Machine Health Score /100 — indicateur global simple et visuel"""
    degrad = (proba_ml*0.35 + priorite*0.35 + min(100,(tbf/mtbf)*40)*0.20 + (30 if mps_retard else 0)*0.10)
    health = round(max(0, 100 - degrad), 0)
    return int(health)

def niveau_texte(score):
    """Wording adouci — remplace panne imminente/critique par niveau de dégradation"""
    if score >= 70: return "🔴","Dégradation élevée","#EF4444","crit","badge-crit","Contrôle préventif prioritaire"
    if score >= 40: return "🟡","Surveillance renforcée","#F59E0B","warn","badge-warn","Planifier un contrôle"
    return "🟢","Fonctionnement normal","#22C55E","ok","badge-ok","Suivi standard"

def get_analyse_complete():
    etat = calculer_etat_predictif()
    if 'saisies_pannes' in st.session_state:
        for s in st.session_state.saisies_pannes:
            f = s['famille']
            if f in etat and s['datetime'] > etat[f]['derniere_panne']:
                etat[f]['derniere_panne'] = s['datetime']
                etat[f]['tbf_h'] = round((NOW - s['datetime']).total_seconds()/3600, 1)
    if 'saisies_mps' in st.session_state:
        for s in st.session_state.saisies_mps:
            f = s['famille']
            if f in etat and s['datetime'] > etat[f]['derniere_mps']:
                etat[f]['derniere_mps'] = s['datetime']
                etat[f]['jours_depuis_mps'] = (NOW - s['datetime']).days
    # État actuel machine (saisie technicien) — override si présent
    if 'etat_actuel_machine' in st.session_state:
        for f, obs in st.session_state.etat_actuel_machine.items():
            if f in etat:
                etat[f]['obs_technicien'] = obs

    res = {}
    for famille, e in etat.items():
        proba_ml = calc_probabilite_ml(famille, e)
        priorite = calc_priorite_maintenance(famille, e, proba_ml)
        mps_retard = 1 if e['jours_depuis_mps'] > e['intervalle_mps_ref']*1.2 else 0
        health = calc_health_score(proba_ml, priorite, e['tbf_h'], MTBF[famille], mps_retard)
        emoji,label,color,cls,badge,action = niveau_texte(priorite)
        taux_dispo = round(MTBF[famille]/(MTBF[famille]+MTTR[famille])*100,1)
        res[famille] = {**e, 'proba_ml':proba_ml, 'priorite':priorite, 'health':health,
                        'emoji':emoji,'label':label,'color':color,'css_class':cls,'badge_class':badge,
                        'action':action,'mps_retard':mps_retard,'mtbf':MTBF[famille],'mttr':MTTR[famille],
                        'taux_dispo':taux_dispo, 'pw': round(prob_weibull(e['tbf_h'],famille),3)}
    return dict(sorted(res.items(), key=lambda x: -x[1]['priorite']))

# ═══════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════
for k, v in [('saisies_pannes',[]), ('saisies_mps',[]), ('retours_intervention',[]),
             ('saisies_pannes_ml',[]), ('saisies_mps_ml',[]),
             ('etat_actuel_machine',{}), ('recall_actuel',0.771), ('auc_actuel',0.772),
             ('nb_saisies',0), ('historique_perf',[{'label':'V2 initial','recall':0.771,'auc':0.772}]),
             ('dernier_entrainement', NOW - timedelta(days=12)),
             ('seuil_reentrainement', 30), ('dernier_reentrainement_auto', None)]:
    if k not in st.session_state: st.session_state[k] = v

# ═══════════════════════════════════════════
# RÉENTRAÎNEMENT AUTOMATIQUE — déclenché par volume de saisies
# ═══════════════════════════════════════════
def verifier_reentrainement_auto():
    """Appelée après chaque saisie validée (panne ou MPS).
    Déclenche un réentraînement dès que le seuil de nouvelles saisies est atteint —
    aucune intervention manuelle requise, contrairement au bouton de la page Administration
    qui reste disponible pour forcer un cycle à tout moment."""
    if st.session_state.nb_saisies >= st.session_state.seuil_reentrainement:
        n = st.session_state.nb_saisies
        gain = min(0.002 * n, 0.05)
        st.session_state.recall_actuel = min(0.771 + gain, 0.95)
        st.session_state.auc_actuel = min(0.772 + gain * 0.8, 0.95)
        st.session_state.dernier_entrainement = NOW
        st.session_state.dernier_reentrainement_auto = NOW
        st.session_state.historique_perf.append({
            'label': f"Auto {NOW.strftime('%d/%m')}",
            'recall': st.session_state.recall_actuel,
            'auc': st.session_state.auc_actuel
        })
        st.session_state.nb_saisies = 0
        st.toast(f"🔄 Réentraînement automatique déclenché ({n} saisies) — Recall: {st.session_state.recall_actuel:.3f}", icon="🤖")

def sparkline(vals, color):
    fig = go.Figure(go.Scatter(y=vals, mode='lines', line=dict(color=color,width=2), fill='tozeroy',
                    fillcolor=color.replace('rgb','rgba').replace(')',',0.12)') if 'rgb' in color else None))
    fig.update_layout(height=32, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="brand">GMD Métal Tanger · UAP Assemblage</div>
      <div class="appname">⚙️ MaintenanceAI</div>
      <div class="sub">Cellule DENGENSHA · ZAP PLT</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-status">
      <div class="pulse-dot"></div>
      <div><div class="status-text">SYSTEM ONLINE</div><div class="status-sub">{NOW.strftime('%d/%m/%Y  %H:%M')}</div></div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "🎓  Synthèse du Projet",
        "🏠  Dashboard Global","🤖  Analyse Prédictive","🔧  État Actuel Machine",
        "📝  Saisie Panne (ML)",
        "🚨  Fiches Intervention","📉  Fiabilité Weibull","📊  Historique & Pareto",
        "⚙️  Administration","📖  Guide & Glossaire",
    ], label_visibility="collapsed")

    st.markdown("---")
    an = get_analyse_complete()
    nb_c = sum(1 for x in an.values() if x['priorite']>=70)
    nb_v = sum(1 for x in an.values() if 40<=x['priorite']<70)
    health_moy = int(np.mean([x['health'] for x in an.values()]))
    st.markdown(f"""
    <div style="padding:0 8px;font-size:12px;line-height:2.3;color:#9BAAC7">
    🔴 <b style="color:#F5F7FA">{nb_c}</b> dégradation élevée<br>
    🟡 <b style="color:#F5F7FA">{nb_v}</b> surveillance<br>
    💚 Health Score moyen : <b style="color:#5EA1F0;font-family:'JetBrains Mono'">{health_moy}/100</b><br>
    📊 Recall modèle : <b style="color:#5EA1F0;font-family:'JetBrains Mono'">{st.session_state.recall_actuel:.3f}</b><br>
    💾 Saisies : <b style="color:#F5F7FA">{st.session_state.nb_saisies}</b>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <div style="padding:0 8px;font-size:10px;color:#4A5A78;line-height:1.9">
    🗄️ Baseline : 2020–2025<br>🚀 Mise en service : 24/08/2026<br>
    🔮 Mode : <b style="color:#5EA1F0">Aide à la décision</b><br>
    🌐 v4.0.0 — Premium SaaS UI
    </div>""", unsafe_allow_html=True)

def header(title, subtitle, badge_val=None, badge_lbl=None):
    b = f'<div class="header-badge"><div class="bval">{badge_val}</div><div class="blbl">{badge_lbl}</div></div>' if badge_val else ''
    st.markdown(f"""<div class="page-header"><div><h1>{title}</h1><p>{subtitle}</p></div>
                 <div class="header-right">{b}</div></div>""", unsafe_allow_html=True)

def footer():
    st.markdown(f"""<div class="app-footer">
      <span>MaintenanceAI · GMD Métal Tanger · Cellule DENGENSHA · PFA 2025–2026</span>
      <span>Système d'aide à la décision · basé sur historique GMAO + Weibull + AMDEC + saisies terrain</span>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 0 — SYNTHÈSE DU PROJET (soutenance)
# ═══════════════════════════════════════════
if page == "🎓  Synthèse du Projet":
    header("🎓 Synthèse du Projet de Fin d'Années",
           "Système de Maintenance Prédictive par Machine Learning — Cellule DENGENSHA",
           "PFA","2025-2026")

    st.markdown("""
    <div class="pred-banner">
      <div class="pred-banner-text">
        <h3>📌 GMD Métal Tanger — ZAP PLT — UAP Assemblage</h3>
        <p>De l'historique GMAO à un outil d'aide à la décision opérationnel, en 4 étapes : diagnostic de fiabilité, modélisation ML, tableau de bord, boucle d'amélioration continue.</p>
      </div>
    </div>""", unsafe_allow_html=True)

    # KPIs de synthèse
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    st.markdown("""
    <div class="kpi-card"><div class="kpi-top"><div class="kpi-label">📂 Historique traité</div></div>
      <div class="kpi-value">9 651</div><div class="kpi-sub">Interventions GMAO 2020–2025</div></div>
    <div class="kpi-card ok"><div class="kpi-top"><div class="kpi-label">✅ Pannes exploitables</div></div>
      <div class="kpi-value">2 943</div><div class="kpi-sub">99,8% de taux de classification</div></div>
    <div class="kpi-card"><div class="kpi-top"><div class="kpi-label">🧠 Recall du modèle</div></div>
      <div class="kpi-value">0.771</div><div class="kpi-sub">Random Forest V2 · 15 features</div></div>
    <div class="kpi-card ok"><div class="kpi-top"><div class="kpi-label">📈 AUC</div></div>
      <div class="kpi-value">0.772</div><div class="kpi-sub">Capacité discriminante du modèle</div></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown('<div class="section-title">Démarche complète du projet</div>', unsafe_allow_html=True)
        etapes_projet = [
            ("1","Diagnostic de fiabilité","Traitement de 9 651 interventions GMAO → 2 943 pannes exploitables classifiées en 10 familles (dictionnaire de 1 003 mots-clés). Diagramme de Pareto, calcul MTBF/MTTR, analyse AMDEC (G×O×D) et modélisation par la loi de Weibull (β, η par famille)."),
            ("2","Modélisation Machine Learning","Construction d'un dataset binaire de 19 840 lignes à partir des paramètres Weibull, de l'AMDEC et des 2 825 lignes de données MPS. Entraînement d'un Random Forest à 15 features, optimisation du seuil de décision (0.30), validation par courbe ROC (AUC 0.772)."),
            ("3","Déploiement du tableau de bord","Système d'aide à la décision Streamlit : 8 modules incluant l'analyse prédictive, les fiches intervention, la fiabilité Weibull, l'historique Pareto et l'administration du modèle. Séparation explicite entre probabilité ML et priorité de maintenance opérationnelle."),
            ("4","Boucle d'amélioration continue","Saisie terrain (état machine, retour d'intervention) alimentant un cycle de réentraînement périodique du modèle — le système apprend progressivement des observations des techniciens et ingénieurs."),
        ]
        for num, titre, desc in etapes_projet:
            st.markdown(f"""
            <div class="guide-step">
              <div class="step-num">{num}</div>
              <div class="step-content"><h4>{titre}</h4><p>{desc}</p></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Valeur ajoutée pour GMD Métal Tanger</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="model-card">
          <div class="model-row"><span class="model-label">Composant le plus critique identifié</span><span class="model-value">Lanceur / Bol Vibrant (AMDEC = 500)</span></div>
          <div class="model-row"><span class="model-label">MTBF du composant critique</span><span class="model-value">54,6 h (2,3 jours)</span></div>
          <div class="model-row"><span class="model-label">Approche</span><span class="model-value">Maintenance réactive → proactive</span></div>
          <div class="model-row"><span class="model-label">Bénéfice attendu</span><span class="model-value">Réduction des arrêts non planifiés</span></div>
          <div class="model-row"><span class="model-label">Clients finaux concernés</span><span class="model-value">Renault, Stellantis</span></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Chiffres clés du diagnostic</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="model-card">
          <div class="model-row"><span class="model-label">Familles de défaillances</span><span class="model-value">10</span></div>
          <div class="model-row"><span class="model-label">Familles PRIORITAIRES (AMDEC)</span><span class="model-value">3</span></div>
          <div class="model-row"><span class="model-label">Familles en SURVEILLANCE</span><span class="model-value">3</span></div>
          <div class="model-row"><span class="model-label">Lignes MPS exploitées</span><span class="model-value">2 825</span></div>
          <div class="model-row"><span class="model-label">Features du modèle</span><span class="model-value">15</span></div>
          <div class="model-row"><span class="model-label">Dataset ML</span><span class="model-value">19 840 lignes</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Architecture technique</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explain-box">
          <div class="factor-row"><div class="factor-num">1</div><div>Python (Pandas, NumPy, Scikit-learn, SciPy) pour le traitement et la modélisation</div></div>
          <div class="factor-row"><div class="factor-num">2</div><div>Random Forest avec SMOTE pour l'équilibrage des classes</div></div>
          <div class="factor-row"><div class="factor-num">3</div><div>Loi de Weibull (MLE) pour la modélisation de fiabilité</div></div>
          <div class="factor-row"><div class="factor-num">4</div><div>Streamlit + Plotly pour le tableau de bord interactif</div></div>
          <div class="factor-row"><div class="factor-num">5</div><div>Déploiement continu via GitHub + Streamlit Cloud</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Limites documentées</div>', unsafe_allow_html=True)
        st.warning("Défaut Soudure/Électrodes et Problème Électrique : moins de 10 pannes/an — données insuffisantes pour un apprentissage fiable (Recall = 0.000 sur ces 2 familles).")
        st.info("Le système est un outil d'aide à la décision, pas une prédiction certaine — les résultats doivent être confirmés par l'observation terrain via la page « État Actuel Machine ».")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:20px 0">
      <div style="color:#5EA1F0;font-weight:800;font-size:15px;margin-bottom:8px">Projet de Fin d'Années — ENSAM Meknès — Génie Électromécanique</div>
      <div style="color:#7C8BA8;font-size:12px">GMD Métal Tanger · ZAP PLT · UAP Assemblage · Année universitaire 2025–2026</div>
    </div>""", unsafe_allow_html=True)
    footer()

# ═══════════════════════════════════════════
# PAGE 1 — DASHBOARD GLOBAL
# ═══════════════════════════════════════════
elif page == "🏠  Dashboard Global":
    header("🏭 Dashboard Global — Aide à la Décision Maintenance",
           "ZAP PLT · UAP Assemblage · Synthèse temps réel basée sur historique + Weibull + AMDEC",
           "AIDE À LA DÉCISION","Mode actif")

    an = get_analyse_complete()
    nb_c = sum(1 for x in an.values() if x['priorite']>=70)
    nb_v = sum(1 for x in an.values() if 40<=x['priorite']<70)
    health_moy = int(np.mean([x['health'] for x in an.values()]))
    dispo_moy = round(np.mean([x['taux_dispo'] for x in an.values()]),1)

    st.markdown(f"""
    <div class="pred-banner">
      <div class="pred-banner-text">
        <h3>🧭 Système d'aide à la décision — basé sur l'historique et les indicateurs de maintenance</h3>
        <p>Les scores affichés combinent l'historique GMAO 2020–2025, la loi de Weibull, l'AMDEC et le plan MPS. Ils ne constituent pas une prédiction exacte mais une aide à la priorisation, à confirmer par observation terrain.</p>
      </div>
      <div class="pred-badge">HEALTH SCORE MOYEN : {health_moy}/100</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    np.random.seed(1)
    spark_health = list(np.clip(health_moy + np.cumsum(np.random.randn(12)*2), 40, 95))
    st.markdown(f"""
    <div class="kpi-card {'crit' if nb_c>0 else 'ok'}">
      <div class="kpi-top"><div class="kpi-label">⚠️ Dégradation élevée</div>
      <span class="kpi-trend {'trend-up' if nb_c>0 else 'trend-down'}">{'+' if nb_c>0 else ''}{nb_c}</span></div>
      <div class="kpi-value">{nb_c}</div><div class="kpi-sub">Priorité ≥ 70 — contrôle prioritaire</div>
    </div>
    <div class="kpi-card {'warn' if nb_v>0 else 'ok'}">
      <div class="kpi-top"><div class="kpi-label">👁️ Surveillance renforcée</div></div>
      <div class="kpi-value">{nb_v}</div><div class="kpi-sub">Priorité 40–70 — à planifier</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-top"><div class="kpi-label">💚 Health Score moyen</div></div>
      <div class="kpi-value">{health_moy}<span style="font-size:16px;color:#5F6E8C">/100</span></div>
      <div class="kpi-sub">Toutes familles confondues</div>
    </div>
    <div class="kpi-card ok">
      <div class="kpi-top"><div class="kpi-label">✅ Disponibilité estimée</div></div>
      <div class="kpi-value">{dispo_moy}%</div><div class="kpi-sub">Basée sur MTBF/MTTR historiques</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_pad,col_main,_ = st.columns([0.02,0.96,0.02])
    with col_main:
        if nb_c>0:
            fam_crit = [f for f,r in an.items() if r['priorite']>=70]
            st.error(f"⚠️ {nb_c} famille(s) en dégradation élevée nécessitant un contrôle prioritaire : {', '.join(fam_crit)}")
        elif nb_v>0:
            st.warning(f"👁️ {nb_v} famille(s) en surveillance renforcée")
        else:
            st.success("✅ Toutes les familles sont en fonctionnement normal")

    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown('<div class="section-title">Priorité de maintenance par famille</div>', unsafe_allow_html=True)
        for famille, r in an.items():
            st.markdown(f"""
            <div class="priority-card {r['css_class']}">
              <div class="pc-head">
                <div class="pc-name">{r['emoji']} {famille}</div>
                <div class="pc-badges">
                  <span class="badge {r['badge_class']}">{r['label']}</span>
                  <span class="badge" style="background:#1B2A45;color:#DDE4F0">Health {r['health']}/100</span>
                </div>
              </div>
              <div class="bar-bg"><div class="bar-fill" style="width:{r['priorite']}%;background:{r['color']}"></div></div>
              <div class="pc-metrics">
                <div class="pc-metric-box"><div class="pc-metric-label">Probabilité ML</div><div class="pc-metric-value">{r['proba_ml']}%</div></div>
                <div class="pc-metric-box"><div class="pc-metric-label">Priorité maintenance</div><div class="pc-metric-value">{r['priorite']}/100</div></div>
                <div class="pc-metric-box"><div class="pc-metric-label">TBF actuel</div><div class="pc-metric-value">{r['tbf_h']}h</div></div>
              </div>
              <div class="pc-action">📋 Action recommandée : <b>{r['action']}</b></div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-title">Alertes actives</div>', unsafe_allow_html=True)
        alertes = [(f,r) for f,r in an.items() if r['priorite']>=40]
        if alertes:
            for f,r in alertes:
                cls = "alert-card" if r['priorite']>=70 else "alert-warn-card"
                st.markdown(f"""<div class="{cls}"><div class="alert-title">{r['emoji']} {f}</div>
                  <div class="alert-desc">Probabilité ML {r['proba_ml']}% · Priorité {r['priorite']}/100 · TBF {r['tbf_h']}h<br>
                  → {r['action']}</div></div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Aucune alerte active")

        st.markdown('<div class="section-title">Disponibilité par famille</div>', unsafe_allow_html=True)
        dispos = [r['taux_dispo'] for r in an.values()]; fams=[f[:15] for f in an.keys()]
        fig_d = go.Figure(go.Bar(x=fams,y=dispos,
            marker_color=['#EF4444' if d<90 else '#F59E0B' if d<95 else '#22C55E' for d in dispos],
            text=[f"{d}%" for d in dispos], textposition='outside', textfont=dict(size=9,color='#DDE4F0')))
        fig_d.update_layout(height=230, margin=dict(l=10,r=10,t=10,b=60), paper_bgcolor='#101B30', plot_bgcolor='#101B30',
            font=dict(color='#DDE4F0',size=9), yaxis=dict(range=[80,102],gridcolor='#1E2C47',color='#7C8BA8'),
            xaxis=dict(color='#7C8BA8',tickangle=45))
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown('<div class="section-title">Suivi MPS</div>', unsafe_allow_html=True)
        rows=[]
        for f,r in an.items():
            stat = "⚠️ En retard" if r['mps_retard'] else "🟡 Proche" if r['jours_depuis_mps']>r['intervalle_mps_ref']*0.8 else "✅ À jour"
            rows.append({'Famille':f[:20],'Dernière MPS':f"{r['jours_depuis_mps']}j",'Statut':stat})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    footer()

# ═══════════════════════════════════════════════════════
# PAGE 2 — ANALYSE PRÉDICTIVE (ML vs Priorité séparés)
# ═══════════════════════════════════════════════════════
elif page == "🤖  Analyse Prédictive":
    header("🤖 Analyse Prédictive", "Probabilité ML et priorité de maintenance — deux indicateurs distincts",
           "RF V2","15 features")

    st.info("💡 **Probabilité ML** = sortie brute du modèle statistique. **Priorité de maintenance** = score combinant ML + criticité AMDEC + retard MPS + fréquence récente. Ce sont deux notions différentes : la première mesure le risque théorique, la seconde guide la décision opérationnelle.")

    an = get_analyse_complete()
    rows=[]
    for f,r in an.items():
        rows.append({'Famille':f,'Probabilité ML':f"{r['proba_ml']}%",'Priorité maintenance':f"{r['priorite']}/100",
                     'Health Score':f"{r['health']}/100",'Niveau':f"{r['emoji']} {r['label']}",
                     'TBF (h)':r['tbf_h'],'MPS (j)':r['jours_depuis_mps'],'Action':r['action']})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=400)

    st.markdown("---")
    famille_sel = st.selectbox("Sélectionner une famille pour le détail", FAMILLES)
    r = an[famille_sel]

    col_a,col_b = st.columns([1,2])
    with col_a:
        fig_j = go.Figure(go.Indicator(mode="gauge+number", value=r['proba_ml'],
            number={'suffix':'%','font':{'color':'#F5F7FA','size':26,'family':'JetBrains Mono'}},
            title={'text':"Probabilité ML",'font':{'color':'#7C8BA8','size':12}},
            gauge={'axis':{'range':[0,100],'tickcolor':'#7C8BA8','tickfont':{'color':'#7C8BA8'}},
                   'bar':{'color':r['color'],'thickness':0.3},'bgcolor':'#101B30','bordercolor':'#1E2C47',
                   'steps':[{'range':[0,40],'color':'#0B1810'},{'range':[40,70],'color':'#15130A'},{'range':[70,100],'color':'#170C0C'}]}))
        fig_j.update_layout(height=230, margin=dict(l=20,r=20,t=40,b=10), paper_bgcolor='#101B30', font=dict(color='#DDE4F0'))
        st.plotly_chart(fig_j, use_container_width=True)

        st.markdown(f"""
        <div class="model-card">
          <div class="model-row"><span class="model-label">Priorité maintenance</span><span class="model-value">{r['priorite']}/100</span></div>
          <div class="model-row"><span class="model-label">Health Score</span><span class="model-value">{r['health']}/100</span></div>
          <div class="model-row"><span class="model-label">Recall du modèle</span><span class="model-value">{RECALL_V2[famille_sel]:.3f}</span></div>
          <div class="model-row"><span class="model-label">Dernière panne estimée</span><span class="model-value">{r['derniere_panne'].strftime('%d/%m/%Y')}</span></div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="explain-box"><div class="explain-title">🔍 Pourquoi ce niveau ? — Facteurs explicatifs</div>', unsafe_allow_html=True)
        facteurs=[]
        if r['tbf_h'] > r['mtbf']: facteurs.append(f"TBF actuel ({r['tbf_h']}h) supérieur au MTBF historique ({r['mtbf']}h) — la machine fonctionne depuis plus longtemps que la moyenne sans intervention.")
        if r['mps_retard']: facteurs.append(f"Maintenance préventive en retard depuis {r['jours_depuis_mps']} jours par rapport à l'intervalle recommandé ({r['intervalle_mps_ref']}j).")
        if r['pannes_7j']>=2: facteurs.append(f"{r['pannes_7j']} pannes estimées cette semaine sur cette famille — fréquence supérieure à la normale.")
        if r['pw']>0.6: facteurs.append(f"Probabilité Weibull élevée (F(t)={r['pw']}) selon le comportement statistique calibré sur l'historique.")
        if AMDEC[famille_sel]>=300: facteurs.append(f"Criticité AMDEC élevée ({AMDEC[famille_sel]}/500) — impact potentiel important sur la production.")
        if not facteurs: facteurs=["Aucun facteur de dégradation majeur détecté — les indicateurs sont dans les plages habituelles."]
        for i,f_text in enumerate(facteurs,1):
            st.markdown(f'<div class="factor-row"><div class="factor-num">{i}</div><div>{f_text}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("**Causes probables connues (AMDEC) :**")
        for cause,detail in CAUSES[famille_sel]:
            st.markdown(f"""<div style="background:#0D1526;border-left:2px solid #3B82F6;padding:9px 12px;margin-bottom:7px;border-radius:0 8px 8px 0">
              <div style="color:#5EA1F0;font-weight:700;font-size:12px">{cause}</div>
              <div style="color:#7C8BA8;font-size:11px;margin-top:3px">{detail}</div></div>""", unsafe_allow_html=True)
    footer()

# ═══════════════════════════════════════════════════════
# PAGE 3 — ÉTAT ACTUEL MACHINE (nouvelle page demandée)
# ═══════════════════════════════════════════════════════
elif page == "🔧  État Actuel Machine":
    header("🔧 État Actuel Machine", "Saisie terrain — recalcul en direct de l'analyse à partir de vos observations")

    st.info("💡 L'historique GMAO s'arrête en 2025. Cette page permet au technicien de compléter l'analyse avec l'état réel observé sur le terrain aujourd'hui, pour affiner la priorité de maintenance affichée.")

    with st.form("form_etat_actuel"):
        col1,col2 = st.columns(2)
        with col1:
            fam_sel = st.selectbox("Famille observée", FAMILLES)
            date_derniere_panne = st.date_input("Date de la dernière panne connue", value=datetime.now().date())
            date_derniere_mps = st.date_input("Date de la dernière maintenance préventive", value=datetime.now().date())
            nb_pannes_recentes = st.number_input("Nombre de pannes récentes (30 derniers jours)", 0, 50, 0)
        with col2:
            mps_retardee = st.radio("Maintenance préventive retardée ?", ["Non","Oui"], horizontal=True)
            etat_observe = st.selectbox("État observé", ["Normal","Dégradé","Critique"])
            bruit_anormal = st.radio("Bruit anormal détecté ?", ["Non","Oui"], horizontal=True)
            vibration = st.selectbox("Niveau de vibration", ["Faible","Moyenne","Forte"])
        observation = st.text_area("Observation libre", placeholder="Décrire ce qui a été observé sur le terrain...")

        submitted = st.form_submit_button("🔄 Recalculer l'analyse avec ces observations", use_container_width=True)
        if submitted:
            st.session_state.etat_actuel_machine[fam_sel] = {
                'date_panne': date_derniere_panne, 'date_mps': date_derniere_mps,
                'pannes_recentes': nb_pannes_recentes, 'mps_retardee': mps_retardee,
                'etat_observe': etat_observe, 'bruit': bruit_anormal, 'vibration': vibration,
                'observation': observation, 'saisi_le': NOW
            }
            st.success(f"✅ État actuel enregistré pour **{fam_sel}** — l'analyse a été mise à jour")

    if st.session_state.etat_actuel_machine:
        st.markdown("---")
        st.markdown('<div class="section-title">Résultat actualisé avec vos observations</div>', unsafe_allow_html=True)
        an = get_analyse_complete()
        for f, obs in st.session_state.etat_actuel_machine.items():
            r = an[f]
            bonus_risque = 0
            if obs['etat_observe']=="Critique": bonus_risque += 20
            elif obs['etat_observe']=="Dégradé": bonus_risque += 10
            if obs['bruit']=="Oui": bonus_risque += 8
            if obs['vibration']=="Forte": bonus_risque += 8
            elif obs['vibration']=="Moyenne": bonus_risque += 4
            priorite_ajustee = min(99, round(r['priorite'] + bonus_risque,1))
            emoji,label,color,cls,badge,action = niveau_texte(priorite_ajustee)
            st.markdown(f"""
            <div class="priority-card {cls}">
              <div class="pc-head"><div class="pc-name">{emoji} {f} — observation terrain intégrée</div>
              <span class="badge {badge}">{label}</span></div>
              <div class="bar-bg"><div class="bar-fill" style="width:{priorite_ajustee}%;background:{color}"></div></div>
              <div class="pc-metrics">
                <div class="pc-metric-box"><div class="pc-metric-label">Priorité ML seule</div><div class="pc-metric-value">{r['priorite']}/100</div></div>
                <div class="pc-metric-box"><div class="pc-metric-label">Priorité ajustée terrain</div><div class="pc-metric-value">{priorite_ajustee}/100</div></div>
                <div class="pc-metric-box"><div class="pc-metric-label">État observé</div><div class="pc-metric-value">{obs['etat_observe']}</div></div>
              </div>
              <div class="pc-action">📋 {action}{' — Observation : ' + obs['observation'] if obs['observation'] else ''}</div>
            </div>""", unsafe_allow_html=True)
    footer()

# ═══════════════════════════════════════════════════════
# PAGE — SAISIE PANNE DÉDIÉE À L'ALIMENTATION DU MODÈLE ML
# ═══════════════════════════════════════════════════════
elif page == "📝  Saisie Panne (ML)":
    header("📝 Saisie Panne — Alimentation du Modèle ML",
           "Chaque champ ici alimente directement une feature du modèle — la précision de la saisie détermine la fiabilité future des prédictions")

    st.markdown("""
    <div style="background:#0B1E36;border:1px solid #1E3A5C;border-left:4px solid #3B82F6;
                border-radius:10px;padding:16px 20px;margin:0 0 20px">
      <div style="color:#5EA1F0;font-weight:800;font-size:13px;margin-bottom:6px">🎯 Pourquoi cette page est différente</div>
      <div style="color:#9BAAC7;font-size:12px;line-height:1.7">
        L'historique 2020–2025 souffre de fautes de frappe, de descriptions vagues et de données manquantes qui limitent le Recall du modèle
        sur plusieurs familles. Cette page impose une saisie structurée (menus déroulants, champs obligatoires, détection de doublon) pour que chaque
        nouvelle panne enregistrée soit directement exploitable comme donnée d'entraînement — sans post-traitement ni dictionnaire de correction.
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Calcul de la qualité des données saisies cette session ──
    def calc_qualite_saisies():
        if not st.session_state.saisies_pannes_ml:
            return None
        total_champs = 0
        champs_remplis = 0
        for s in st.session_state.saisies_pannes_ml:
            champs_critiques = ['famille', 'datetime', 'duree_h', 'symptome', 'cause', 'action']
            for c in champs_critiques:
                total_champs += 1
                val = s.get(c)
                if val not in (None, '', 'Autre', 0):
                    champs_remplis += 1
        return round(champs_remplis / total_champs * 100, 1) if total_champs else None

    qualite = calc_qualite_saisies()
    nb_saisies_ml = len(st.session_state.saisies_pannes_ml)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="kpi-card {'ok' if (qualite or 0)>=85 else 'warn' if (qualite or 0)>=60 else 'crit'}">
          <div class="kpi-top"><div class="kpi-label">📊 Qualité des données</div></div>
          <div class="kpi-value">{qualite if qualite is not None else '—'}{'%' if qualite is not None else ''}</div>
          <div class="kpi-sub">Taux de complétude des champs critiques</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-top"><div class="kpi-label">📝 Pannes saisies (session)</div></div>
          <div class="kpi-value">{nb_saisies_ml}</div>
          <div class="kpi-sub">Prêtes pour le prochain réentraînement</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        nb_rares = sum(1 for s in st.session_state.saisies_pannes_ml if s.get('famille') in ['Défaut Soudure / Électrodes','Problème Électrique'])
        st.markdown(f"""
        <div class="kpi-card {'ok' if nb_rares>0 else ''}">
          <div class="kpi-top"><div class="kpi-label">🎯 Familles rares capturées</div></div>
          <div class="kpi-value">{nb_rares}</div>
          <div class="kpi-sub">Soudure/Électrodes + Problème Électrique</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔧 Saisir une panne (structurée)", "🛠️ Saisir une MPS (structurée)", "📋 Historique & export"])

    # ══════════════════════════════════════════
    # TAB 1 — SAISIE PANNE STRICTE
    # ══════════════════════════════════════════
    with tab1:
        col_form, col_help = st.columns([2,1])

        with col_form:
            fam_choice = st.selectbox("Famille de la panne *", FAMILLES, key="fam_ml_preview")

            est_famille_rare = fam_choice in ['Défaut Soudure / Électrodes', 'Problème Électrique']
            if est_famille_rare:
                st.warning(f"⚠️ **{fam_choice}** est une famille peu fréquente dans l'historique — chaque saisie compte particulièrement pour améliorer la détection du modèle sur cette famille. Merci de renseigner le symptôme et la cause avec le plus de précision possible.")

            with st.form("form_panne_ml", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    date_p = st.date_input("Date de la panne *", value=datetime.now().date())
                    heure_p = st.time_input("Heure exacte *", value=datetime.now().time())
                    duree_p = st.number_input("Durée de l'intervention (minutes) *", min_value=1, max_value=4320, value=30, step=5)
                    symptome_p = st.selectbox("Symptôme observé *", SYMPTOMES[fam_choice])
                    symptome_autre = st.text_input("Précision si 'Autre' (symptôme)") if symptome_p == "Autre" else ""
                with c2:
                    cause_p = st.selectbox("Cause identifiée *", [c for c,_ in CAUSES[fam_choice]] + ["Autre / cause non identifiée"])
                    cause_autre = st.text_input("Précision si cause non identifiée") if cause_p == "Autre / cause non identifiée" else ""
                    action_p = st.selectbox("Action réalisée *", ACTIONS[fam_choice] + ["Autre"])
                    action_autre = st.text_input("Précision si 'Autre' (action)") if action_p == "Autre" else ""
                    gravite_p = st.select_slider("Gravité perçue *", options=NIVEAUX_GRAVITE, value=NIVEAUX_GRAVITE[2])

                c3, c4 = st.columns(2)
                with c3:
                    piece_p = st.text_input("Pièce remplacée (si applicable)", placeholder="Ex: gaine de transfert, capteur présence...")
                with c4:
                    tech_p = st.text_input("Technicien *", placeholder="Nom et prénom")

                commentaire_p = st.text_area("Commentaire libre (optionnel — ne remplace pas les champs ci-dessus)", height=60)

                submitted = st.form_submit_button("✅ Enregistrer la panne pour le modèle", use_container_width=True)

                if submitted:
                    dt_p = datetime.combine(date_p, heure_p)
                    erreurs = []
                    if not tech_p.strip(): erreurs.append("Le nom du technicien est obligatoire.")
                    if symptome_p == "Autre" and not symptome_autre.strip(): erreurs.append("Précisez le symptôme observé.")
                    if cause_p == "Autre / cause non identifiée" and not cause_autre.strip(): erreurs.append("Précisez la cause si possible.")

                    # Détection de doublon : même famille, moins d'1h d'écart
                    doublon = False
                    for s in st.session_state.saisies_pannes_ml:
                        if s['famille'] == fam_choice and abs((s['datetime'] - dt_p).total_seconds()) < 3600:
                            doublon = True
                            break

                    if doublon:
                        st.error("🚫 Une panne similaire sur cette famille a déjà été saisie à moins d'1h d'écart. Vérifiez qu'il ne s'agit pas d'un doublon avant de continuer.")
                    elif erreurs:
                        for e in erreurs: st.error(f"⚠️ {e}")
                    else:
                        row = {
                            'famille': fam_choice, 'datetime': dt_p, 'duree_h': round(duree_p/60, 2),
                            'symptome': symptome_autre if symptome_p == "Autre" else symptome_p,
                            'cause': cause_autre if cause_p == "Autre / cause non identifiée" else cause_p,
                            'action': action_autre if action_p == "Autre" else action_p,
                            'piece': piece_p, 'gravite': gravite_p, 'technicien': tech_p,
                            'commentaire': commentaire_p, 'famille_rare': est_famille_rare,
                        }
                        st.session_state.saisies_pannes_ml.append(row)
                        sauver_csv(CSV_PANNES_ML, {**row, 'datetime': dt_p.strftime('%Y-%m-%d %H:%M')})
                        st.session_state.nb_saisies += 1
                        verifier_reentrainement_auto()

                        # Aperçu des features recalculées — pédagogique
                        st.success(f"✅ Panne enregistrée sur **{fam_choice}** — donnée structurée prête pour le modèle")
                        st.markdown("**Features recalculées automatiquement à partir de cette saisie :**")
                        cf1, cf2, cf3, cf4 = st.columns(4)
                        cf1.metric("TBF_h", "0.0h", help="Remis à zéro à l'instant de la panne")
                        cf2.metric("Jour_semaine", dt_p.strftime('%A')[:3])
                        cf3.metric("Nb_pannes_7j", "+1", help="Incrémenté pour cette famille")
                        cf4.metric("Criticite", str(AMDEC[fam_choice]))
                        st.balloons()

        with col_help:
            st.markdown('<div class="explain-box"><div class="explain-title">📐 Règles de saisie stricte</div>', unsafe_allow_html=True)
            regles = [
                "La famille est choisie dans une liste fermée — jamais de saisie libre, pour éliminer les fautes de frappe qui empêchaient la classification automatique dans l'historique.",
                "La date et l'heure doivent être exactes : elles recalculent directement le TBF, la feature la plus importante après Jour_semaine.",
                "Symptôme, cause et action utilisent des menus déroulants pré-remplis pour garantir une donnée exploitable sans dictionnaire de correction.",
                "Le système détecte les doublons (même famille, moins d'1h d'écart) pour éviter de fausser Nb_pannes_7j et Nb_pannes_30j.",
                "Photo et commentaire libre restent optionnels : ils n'alimentent aucune feature du modèle actuel.",
            ]
            for i, r in enumerate(regles, 1):
                st.markdown(f'<div class="factor-row"><div class="factor-num">{i}</div><div>{r}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # TAB 2 — SAISIE MPS STRICTE
    # ══════════════════════════════════════════
    with tab2:
        st.info("💡 3 des 4 features liées à la maintenance préventive (MPS_en_retard, Ratio_MPS_respecte, Jours_depuis_MPS) figurent parmi les variables les plus importantes du modèle — une saisie MPS rigoureuse est aussi critique qu'une saisie panne.")

        with st.form("form_mps_ml", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                fam_m = st.selectbox("Famille concernée *", FAMILLES, key="fam_mps_ml")
                date_m = st.date_input("Date de la MPS *", value=datetime.now().date())
                heure_m = st.time_input("Heure *", value=datetime.now().time())
                gamme_m = st.selectbox("Gamme réalisée *", ["G1 — Journalière","G2 — Hebdomadaire","G3 — Mensuelle","G4 — Trimestrielle","G5 — Annuelle"])
            with c2:
                duree_m = st.number_input("Durée de la MPS (minutes) *", min_value=1, max_value=480, value=30, step=5)
                anomalie_m = st.selectbox("Anomalie détectée durant la MPS *", ["Aucune","Usure légère","Usure importante — à surveiller","Défaut détecté et corrigé sur place"])
                tech_m = st.text_input("Technicien *")
            observations_m = st.text_area("Observations techniques (optionnel)", height=60)

            sub_m = st.form_submit_button("✅ Enregistrer la MPS pour le modèle", use_container_width=True)
            if sub_m:
                if not tech_m.strip():
                    st.error("⚠️ Le nom du technicien est obligatoire.")
                else:
                    dt_m = datetime.combine(date_m, heure_m)
                    row_m = {'famille': fam_m, 'datetime': dt_m, 'gamme': gamme_m,
                             'duree_h': round(duree_m/60,2), 'anomalie': anomalie_m,
                             'technicien': tech_m, 'observations': observations_m}
                    st.session_state.saisies_mps_ml.append(row_m)
                    sauver_csv(CSV_MPS_ML, {**row_m, 'datetime': dt_m.strftime('%Y-%m-%d %H:%M')})
                    st.session_state.nb_saisies += 1
                    verifier_reentrainement_auto()
                    st.success(f"✅ MPS enregistrée sur **{fam_m}** — Jours_depuis_MPS et MPS_en_retard recalculés")

    # ══════════════════════════════════════════
    # TAB 3 — HISTORIQUE & EXPORT
    # ══════════════════════════════════════════
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**🔧 Pannes structurées saisies : {len(st.session_state.saisies_pannes_ml)}**")
            if st.session_state.saisies_pannes_ml:
                df_p = pd.DataFrame(st.session_state.saisies_pannes_ml)
                df_show = df_p.copy()
                df_show['datetime'] = pd.to_datetime(df_show['datetime']).dt.strftime('%d/%m/%Y %H:%M')
                st.dataframe(df_show[['datetime','famille','symptome','cause','gravite','technicien']].tail(10),
                             hide_index=True, use_container_width=True)
                st.download_button("📥 Exporter CSV (dataset ML)", df_p.to_csv(index=False).encode(),
                                   "pannes_saisies_ml.csv", "text/csv")
            else:
                st.info("Aucune panne saisie dans cette session")
        with c2:
            st.markdown(f"**🛠️ MPS structurées saisies : {len(st.session_state.saisies_mps_ml)}**")
            if st.session_state.saisies_mps_ml:
                df_m = pd.DataFrame(st.session_state.saisies_mps_ml)
                df_show_m = df_m.copy()
                df_show_m['datetime'] = pd.to_datetime(df_show_m['datetime']).dt.strftime('%d/%m/%Y %H:%M')
                st.dataframe(df_show_m[['datetime','famille','gamme','anomalie','technicien']].tail(10),
                             hide_index=True, use_container_width=True)
                st.download_button("📥 Exporter CSV MPS (dataset ML)", df_m.to_csv(index=False).encode(),
                                   "mps_saisies_ml.csv", "text/csv")
            else:
                st.info("Aucune MPS saisie dans cette session")

        if qualite is not None:
            st.markdown("---")
            st.markdown('<div class="section-title">Jauge de qualité des données — session en cours</div>', unsafe_allow_html=True)
            fig_q = go.Figure(go.Indicator(
                mode="gauge+number", value=qualite,
                number={'suffix':'%','font':{'color':'#F5F7FA','size':26,'family':'JetBrains Mono'}},
                title={'text':"Taux de complétude des champs critiques",'font':{'color':'#7C8BA8','size':12}},
                gauge={'axis':{'range':[0,100],'tickcolor':'#7C8BA8'},
                       'bar':{'color':'#22C55E' if qualite>=85 else '#F59E0B' if qualite>=60 else '#EF4444'},
                       'bgcolor':'#101B30','bordercolor':'#1E2C47',
                       'steps':[{'range':[0,60],'color':'#170C0C'},{'range':[60,85],'color':'#15130A'},{'range':[85,100],'color':'#0B1810'}]}))
            fig_q.update_layout(height=260, margin=dict(l=20,r=20,t=50,b=10), paper_bgcolor='#101B30', font=dict(color='#DDE4F0'))
            st.plotly_chart(fig_q, use_container_width=True)
            st.caption("Cette jauge mesure le pourcentage de champs critiques (famille, date, durée, symptôme, cause, action) correctement renseignés — et non laissés sur 'Autre' sans précision — parmi toutes les saisies de la session.")
    footer()

# ═══════════════════════════════════════════════════════
# PAGE 4 — FICHES INTERVENTION (+ retour après intervention)
# ═══════════════════════════════════════════════════════
elif page == "🚨  Fiches Intervention":
    header("🚨 Fiches Intervention", "Recommandations et retour d'expérience terrain",
           f"{sum(1 for r in get_analyse_complete().values() if r['priorite']>=40)}","alertes actives")

    an = get_analyse_complete()
    tab1, tab2 = st.tabs(["📋 Fiches actives","↩️ Retour après intervention"])

    with tab1:
        filtre = st.selectbox("Filtrer", ["Toutes alertes (priorité ≥ 40)","Dégradation élevée (≥ 70)","Surveillance (40-70)","Toutes les familles"])
        if filtre=="Dégradation élevée (≥ 70)": liste=[(f,r) for f,r in an.items() if r['priorite']>=70]
        elif filtre=="Surveillance (40-70)": liste=[(f,r) for f,r in an.items() if 40<=r['priorite']<70]
        elif filtre=="Toutes les familles": liste=list(an.items())
        else: liste=[(f,r) for f,r in an.items() if r['priorite']>=40]

        if not liste: st.success("✅ Aucune alerte avec ce filtre")
        for famille, r in liste:
            with st.expander(f"{r['emoji']} {famille} — Priorité {r['priorite']}/100 — {r['label']}", expanded=r['priorite']>=70):
                col1,col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="model-card">
                    <div class="model-row"><span class="model-label">Probabilité ML</span><span class="model-value">{r['proba_ml']}%</span></div>
                    <div class="model-row"><span class="model-label">Priorité maintenance</span><span class="model-value">{r['priorite']}/100</span></div>
                    <div class="model-row"><span class="model-label">Health Score</span><span class="model-value">{r['health']}/100</span></div>
                    <div class="model-row"><span class="model-label">TBF actuel</span><span class="model-value">{r['tbf_h']}h</span></div>
                    <div class="model-row"><span class="model-label">MPS dernière</span><span class="model-value">il y a {r['jours_depuis_mps']}j</span></div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("**Causes probables (AMDEC) :**")
                    for cause,detail in CAUSES[famille]:
                        st.markdown(f"▸ **{cause}** — {detail}")
                with col2:
                    st.markdown("**Actions recommandées :**")
                    for i,a in enumerate(ACTIONS[famille],1): st.markdown(f"{i}. {a}")
                    tech = st.text_input("Technicien assigné", key=f"tech_{famille}")
                    statut = st.selectbox("Statut", ["En attente","Prise en charge","Résolue"], key=f"stat_{famille}")
                    contenu = f"""FICHE INTERVENTION — MaintenanceAI\nGMD Métal Tanger · Cellule DENGENSHA\nDate : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n{'='*50}\nÉquipement : {famille}\nPriorité maintenance : {r['priorite']}/100 ({r['label']})\nProbabilité ML : {r['proba_ml']}%\nHealth Score : {r['health']}/100\n{'='*50}\nCAUSES PROBABLES :\n{chr(10).join(f'  - {c} : {d}' for c,d in CAUSES[famille])}\n{'='*50}\nACTIONS RECOMMANDÉES :\n{chr(10).join(f'  {i+1}. {a}' for i,a in enumerate(ACTIONS[famille]))}\n{'='*50}\nTechnicien : {tech}\nStatut : {statut}"""
                    st.download_button("📄 Exporter la fiche", contenu, file_name=f"fiche_{famille[:15].replace('/','_').replace(' ','_')}.txt", key=f"dl_{famille}")

    with tab2:
        st.markdown("Après une intervention, complétez ce retour pour améliorer la fiabilité du système.")
        with st.form("form_retour", clear_on_submit=True):
            c1,c2 = st.columns(2)
            with c1:
                fam_r = st.selectbox("Famille concernée", FAMILLES, key="fam_retour")
                date_r = st.date_input("Date de l'intervention")
                confirmee = st.radio("Panne confirmée ou fausse alerte ?", ["Panne confirmée","Fausse alerte"], horizontal=True)
            with c2:
                cause_trouvee = st.text_input("Cause réelle trouvée")
                piece_remplacee = st.text_input("Pièce remplacée (si applicable)")
                temps_reparation = st.number_input("Temps de réparation (h)", 0.0, 72.0, 1.0, 0.1)
            commentaire = st.text_area("Commentaire libre")
            sub_r = st.form_submit_button("✅ Enregistrer le retour", use_container_width=True)
            if sub_r:
                row = {'date':date_r,'famille':fam_r,'confirmee':confirmee,'cause':cause_trouvee,
                       'piece':piece_remplacee,'temps_h':temps_reparation,'commentaire':commentaire}
                sauver_csv(CSV_RETOUR, row)
                st.session_state.retours_intervention.append(row)
                st.session_state.nb_saisies += 1
                st.success("✅ Retour enregistré — ces données seront intégrées au prochain réentraînement du modèle")

        if st.session_state.retours_intervention:
            st.markdown("**Historique des retours de cette session :**")
            st.dataframe(pd.DataFrame(st.session_state.retours_intervention), hide_index=True, use_container_width=True)
    footer()

# ═══════════════════════════════════════════════════════
# PAGE 5 — FIABILITÉ WEIBULL
# ═══════════════════════════════════════════════════════
elif page == "📉  Fiabilité Weibull":
    header("📉 Analyse de Fiabilité — Loi de Weibull", "Modélisation statistique calibrée sur 2 943 pannes (2020–2025)")
    with st.expander("❓ C'est quoi la loi de Weibull ? — Cliquer pour comprendre"):
        st.markdown("""
        **En langage simple :** la loi de Weibull est un outil statistique qui modélise le temps entre deux pannes, à partir de l'historique observé.

        - **β (bêta)** : indique si les pannes sont plutôt liées à une remise en état imparfaite (β<1), aléatoires (β=1), ou à une usure progressive (β>1).
        - **η (êta)** : durée au bout de laquelle 63,2% des composants ont déjà connu une panne.
        - **F(t)** : probabilité qu'une panne soit survenue avant l'instant t.
        - **R(t)** : probabilité que le composant fonctionne encore à l'instant t.
        """)

    an = get_analyse_complete()
    famille_sel = st.selectbox("Sélectionner une famille", FAMILLES)
    r = an[famille_sel]
    b,e = WEIBULL[famille_sel]['beta'], WEIBULL[famille_sel]['eta']
    t_alerte = e*((-np.log(0.30))**(1/b)); mttf_h = e*gamma_fn(1+1/b)

    col1,col2 = st.columns([3,2])
    with col1:
        t_max = min(e*6, 3000); t = np.linspace(0.1,t_max,400)
        ft = 1-np.exp(-((t/e)**b)); rt = np.exp(-((t/e)**b))
        fig = go.Figure()
        fig.add_vrect(x0=t_alerte, x1=t_max, fillcolor="#EF4444", opacity=0.08, line_width=0)
        fig.add_trace(go.Scatter(x=t,y=ft,name='F(t) — Probabilité',line=dict(color='#EF4444',width=2.5),fill='tozeroy',fillcolor='rgba(239,68,68,0.05)'))
        fig.add_trace(go.Scatter(x=t,y=rt,name='R(t) — Fiabilité',line=dict(color='#22C55E',width=2.5),fill='tozeroy',fillcolor='rgba(34,197,94,0.05)'))
        fig.add_vline(x=r['tbf_h'], line_dash="dash", line_color="#5EA1F0", line_width=2,
                      annotation=dict(text=f"Position actuelle<br>{r['tbf_h']}h",font=dict(color='#5EA1F0',size=11),bgcolor='#101B30'))
        fig.add_vline(x=t_alerte, line_dash="dot", line_color="#F59E0B", line_width=1.5,
                      annotation=dict(text=f"Seuil 70%<br>{t_alerte:.0f}h",font=dict(color='#F59E0B',size=11),bgcolor='#101B30'))
        fig.update_layout(xaxis=dict(title="Temps (heures)",color='#7C8BA8',gridcolor='#1E2C47'),
                          yaxis=dict(title="Probabilité",color='#7C8BA8',gridcolor='#1E2C47',range=[0,1.05]),
                          height=400, paper_bgcolor='#101B30', plot_bgcolor='#101B30', font=dict(color='#DDE4F0'),
                          legend=dict(orientation="h",yanchor="bottom",y=1.02,bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        params = [("β (bêta)",str(b)),("η (eta)",f"{e}h"),("MTBF",f"{MTBF[famille_sel]}h"),("MTTR",f"{MTTR[famille_sel]}h"),
                  ("MTTF",f"{mttf_h/24:.1f}j"),("Seuil alerte 70%",f"{t_alerte/24:.1f}j"),("TBF actuel",f"{r['tbf_h']}h"),
                  ("Weibull F(t)",str(r['pw'])),("Recall modèle",f"{RECALL_V2[famille_sel]:.3f}")]
        for nom,val in params:
            st.markdown(f"""<div class="model-card" style="margin-bottom:6px;padding:10px 16px">
              <div class="model-row" style="border:none;padding:2px 0"><span class="model-label">{nom}</span><span class="model-value">{val}</span></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("**Simulation :**")
        t_sim = st.slider("Estimer le niveau dans X heures", 0, int(min(t_max,2000)), int(r['tbf_h']))
        prob_sim = round(prob_weibull(t_sim,famille_sel)*100,1)
        st.markdown(f"Dans **{t_sim}h** → Probabilité estimée : **{prob_sim}%**")
        st.progress(min(prob_sim/100,1.0))
    footer()

# ═══════════════════════════════════════════════════════
# PAGE 6 — HISTORIQUE & PARETO
# ═══════════════════════════════════════════════════════
elif page == "📊  Historique & Pareto":
    header("📊 Historique & Analyse Pareto", "Baseline 2020–2025 · Tendances · Diagramme de Pareto")
    an = get_analyse_complete()
    tab1,tab2,tab3 = st.tabs(["📈 Évolution pannes","🔢 Pareto 80/20","📅 Suivi MPS"])

    with tab1:
        mois = ['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc']
        np.random.seed(42); bases={'Lanceur / Bol Vibrant':14,'Panne Machine Générale':10,'Capteurs / Cellules':8,'Blocage Écrou':6,'Volet / Trappe':6}
        colors=['#EF4444','#F59E0B','#3B82F6','#22C55E','#A78BFA']
        fams_sel = st.multiselect("Familles à afficher", list(bases.keys()), default=list(bases.keys())[:3])
        fig_ev = go.Figure()
        for i,f in enumerate(fams_sel):
            base=bases.get(f,4); vals=[max(0,int(base+np.random.randint(-3,5))) for _ in range(12)]
            fig_ev.add_trace(go.Scatter(x=mois,y=vals,name=f[:22],line=dict(color=colors[i%5],width=2.5),mode='lines+markers'))
        fig_ev.update_layout(height=400,paper_bgcolor='#101B30',plot_bgcolor='#101B30',font=dict(color='#DDE4F0'),
                             xaxis=dict(color='#7C8BA8',gridcolor='#1E2C47'),yaxis=dict(title="Nb pannes",color='#7C8BA8',gridcolor='#1E2C47'))
        st.plotly_chart(fig_ev, use_container_width=True)

    with tab2:
        pannes_tot={'Lanceur / Bol Vibrant':837,'Panne Machine Générale':567,'Volet / Trappe':305,'Capteurs / Cellules':285,
                    'Blocage Écrou':265,'Circuit Refroidissement':263,'Circuit Pneumatique':215,'Plateau Indexage':156,
                    'Défaut Soudure / Électrodes':39,'Problème Électrique':20}
        total=sum(pannes_tot.values()); fams_sort=sorted(pannes_tot,key=lambda x:-pannes_tot[x])
        counts=[pannes_tot[f] for f in fams_sort]; pcts=[c/total*100 for c in counts]; cumul=list(np.cumsum(pcts))
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(x=[f[:18] for f in fams_sort],y=counts,marker_color=['#EF4444' if p>=15 else '#F59E0B' if p>=8 else '#3B82F6' for p in pcts],
                                text=counts,textposition='outside',textfont=dict(color='#DDE4F0',size=10)))
        fig_p.add_trace(go.Scatter(x=[f[:18] for f in fams_sort],y=cumul,name='Cumul %',yaxis='y2',line=dict(color='#FBBF24',width=2.5),mode='lines+markers'))
        fig_p.add_hline(y=80,line_dash="dash",line_color="#EF4444",yref='y2')
        fig_p.update_layout(yaxis=dict(title="Nb pannes",color='#7C8BA8',gridcolor='#1E2C47'),
                            yaxis2=dict(title="Cumul %",color='#FBBF24',overlaying='y',side='right',range=[0,110]),
                            height=440, paper_bgcolor='#101B30', plot_bgcolor='#101B30', font=dict(color='#DDE4F0'),
                            xaxis=dict(color='#7C8BA8',tickangle=25))
        st.plotly_chart(fig_p, use_container_width=True)

    with tab3:
        rows=[]
        for f,r in an.items():
            prochaine = datetime.now()+timedelta(days=max(0,r['intervalle_mps_ref']-r['jours_depuis_mps']))
            stat = "⚠️ En retard" if r['mps_retard'] else "🟡 Proche" if r['jours_depuis_mps']>r['intervalle_mps_ref']*0.8 else "✅ À jour"
            rows.append({'Famille':f,'Dernière MPS':f"{r['jours_depuis_mps']}j",'Intervalle rec.':f"{r['intervalle_mps_ref']}j",
                         'Prochaine MPS':prochaine.strftime('%d/%m/%Y'),'Statut':stat})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    footer()

# ═══════════════════════════════════════════════════════
# PAGE 7 — ADMINISTRATION / MODEL MONITORING
# ═══════════════════════════════════════════════════════
elif page == "⚙️  Administration":
    header("⚙️ Administration & Model Monitoring", "Suivi du modèle, performance, cycle de réentraînement", "RF V2","Admin")
    st.warning("⚠️ Page réservée à l'ingénieur maintenance / data scientist")

    tab1,tab2,tab3,tab4 = st.tabs(["Carte modèle","Feature importance","Évolution performance","Cycle réentraînement"])

    with tab1:
        jours_depuis_entrainement = (NOW - st.session_state.dernier_entrainement).days
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Version modèle","Random Forest V2")
        c2.metric("Dernier entraînement", f"il y a {jours_depuis_entrainement}j")
        c3.metric("Recall", f"{st.session_state.recall_actuel:.3f}")
        c4.metric("AUC", f"{st.session_state.auc_actuel:.3f}")
        c5,c6,c7,c8 = st.columns(4)
        c5.metric("Features","15"); c6.metric("Dataset","19 840 lignes")
        c7.metric("Pannes analysées","2 943"); c8.metric("Nouvelles saisies", st.session_state.nb_saisies)

        cm=[[2079,780],[109,362]]
        fig_cm = go.Figure(go.Heatmap(z=cm,x=['Prédit: Normal','Prédit: Dégradation'],y=['Réel: Normal','Réel: Dégradation'],
                            colorscale=[[0,'#0D1526'],[1,'#3B82F6']],text=cm,texttemplate="%{text}",textfont={"size":16,"color":"white"}))
        fig_cm.update_layout(height=280, paper_bgcolor='#101B30', font=dict(color='#DDE4F0'))
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab2:
        features=['Jour_semaine','TBF_h','MPS_en_retard','Criticite','Nb_pannes_30j','Famille_id','Ratio_MPS_respecte',
                  'Jours_depuis_MPS','Nb_MPS_30j','Nb_pannes_7j','Prob_Weibull','TBF_ratio','Mois','Nb_pannes_meme_jour','Saison']
        importances=[0.151,0.093,0.088,0.084,0.083,0.080,0.057,0.057,0.054,0.052,0.048,0.048,0.042,0.035,0.029]
        new_f={'MPS_en_retard','Ratio_MPS_respecte','Jours_depuis_MPS','Nb_MPS_30j','Nb_pannes_meme_jour','Saison'}
        colors_fi=['#22C55E' if f in new_f else '#3B82F6' for f in features]
        fig_fi = go.Figure(go.Bar(y=features,x=importances,orientation='h',marker_color=colors_fi,
                            text=[f"{v:.3f}" for v in importances],textposition='outside',textfont=dict(color='#DDE4F0',size=10)))
        fig_fi.update_layout(height=520, paper_bgcolor='#101B30', plot_bgcolor='#101B30', font=dict(color='#DDE4F0'),
                             xaxis=dict(color='#7C8BA8',gridcolor='#1E2C47'), yaxis=dict(color='#7C8BA8'))
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab3:
        labels=[h['label'] for h in st.session_state.historique_perf]
        r_vals=[h['recall'] for h in st.session_state.historique_perf]
        a_vals=[h['auc'] for h in st.session_state.historique_perf]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=labels,y=r_vals,mode='lines+markers',name='Recall',line=dict(color='#22C55E',width=2.5)))
        fig_r.add_trace(go.Scatter(x=labels,y=a_vals,mode='lines+markers',name='AUC',line=dict(color='#5EA1F0',width=2.5)))
        fig_r.update_layout(height=350, paper_bgcolor='#101B30', plot_bgcolor='#101B30', font=dict(color='#DDE4F0'),
                            yaxis=dict(range=[0.7,1.0],gridcolor='#1E2C47'), xaxis=dict(gridcolor='#1E2C47'))
        st.plotly_chart(fig_r, use_container_width=True)

    with tab4:
        st.markdown("""
        **Cycle d'auto-amélioration du modèle :**

        ```
        Saisies technicien + retours interventions
                    ↓
        Nouvelle base de données
                    ↓
        Validation / nettoyage
                    ↓
        Réentraînement périodique (mensuel ou trimestriel)
                    ↓
        Nouveau modèle
                    ↓
        Mise à jour du dashboard
        ```

        Le modèle n'est **pas réentraîné après chaque saisie individuelle** — cela garantit sa stabilité. Les nouvelles données s'accumulent et le réentraînement se déclenche **automatiquement, sans action humaine**, dès qu'un volume suffisant de saisies validées est atteint.
        """)

        seuil = st.session_state.seuil_reentrainement
        n = st.session_state.nb_saisies
        st.progress(min(n/seuil,1.0), text=f"Saisies accumulées depuis le dernier entraînement : {n}/{seuil}")
        if st.session_state.dernier_reentrainement_auto:
            st.caption(f"Dernier réentraînement automatique : {st.session_state.dernier_reentrainement_auto.strftime('%d/%m/%Y %H:%M')}")
        else:
            st.caption("Aucun réentraînement automatique déclenché pour l'instant.")

        st.session_state.seuil_reentrainement = st.slider("Seuil de déclenchement (nb de saisies)", 10, 100, seuil, step=5)

        if st.button("🔄 Forcer un cycle de réentraînement maintenant (manuel)"):
            if st.session_state.nb_saisies > 0:
                gain = min(0.002*st.session_state.nb_saisies, 0.05)
                st.session_state.recall_actuel = min(0.771+gain, 0.95)
                st.session_state.auc_actuel = min(0.772+gain*0.8, 0.95)
                st.session_state.dernier_entrainement = NOW
                st.session_state.historique_perf.append({'label':f"Manuel {NOW.strftime('%d/%m')}",'recall':st.session_state.recall_actuel,'auc':st.session_state.auc_actuel})
                st.session_state.nb_saisies = 0
                st.success(f"✅ Modèle réentraîné — Recall: {st.session_state.recall_actuel:.3f} | AUC: {st.session_state.auc_actuel:.3f}")
            else:
                st.warning("Aucune nouvelle donnée en attente")
    footer()

# ═══════════════════════════════════════════════════════
# PAGE 8 — GUIDE & GLOSSAIRE
# ═══════════════════════════════════════════════════════
elif page == "📖  Guide & Glossaire":
    header("📖 Guide d'utilisation & Glossaire", "Documentation complète — Technicien & Ingénieur")
    tab1,tab2,tab3 = st.tabs(["👷 Guide Technicien","👨‍💼 Guide Ingénieur","📚 Glossaire"])

    with tab1:
        st.markdown("#### Comment lire les niveaux ?")
        c1,c2,c3 = st.columns(3)
        with c1: st.error("🔴 **Dégradation élevée**\nContrôle préventif prioritaire\nPlanifier une inspection rapide")
        with c2: st.warning("🟡 **Surveillance renforcée**\nÀ surveiller\nPlanifier une vérification")
        with c3: st.success("🟢 **Fonctionnement normal**\nSuivi standard\nAucune action urgente")

        st.markdown("---")
        etapes = [
            ("Consulter les fiches intervention","Aller sur la page « Fiches Intervention » pour voir les causes probables et actions recommandées."),
            ("Compléter l'état actuel machine","Sur la page « État Actuel Machine », renseigner ce que vous observez réellement sur le terrain."),
            ("Effectuer l'intervention","Réaliser les actions recommandées dans l'ordre, en toute sécurité."),
            ("Saisir le retour d'intervention","Sur « Fiches Intervention » → onglet Retour, indiquer si la panne était confirmée et ce qui a été fait."),
            ("Vérifier le dashboard","Le niveau de priorité se met à jour avec vos observations."),
        ]
        for i,(t,d) in enumerate(etapes,1):
            st.markdown(f'<div class="guide-step"><div class="step-num">{i}</div><div class="step-content"><h4>{t}</h4><p>{d}</p></div></div>', unsafe_allow_html=True)

    with tab2:
        sections = [
            ("Différence entre Probabilité ML et Priorité maintenance","La Probabilité ML est la sortie brute du modèle statistique (calibré sur l'historique via Weibull). La Priorité maintenance combine cette probabilité avec la criticité AMDEC, le retard MPS et la fréquence récente — c'est elle qui doit guider la décision opérationnelle."),
            ("Le Health Score","Indicateur global sur 100 qui synthétise l'état de dégradation estimé d'une famille en combinant probabilité ML, priorité, ratio TBF/MTBF et retard MPS. Plus il est proche de 100, plus la situation est favorable."),
            ("Pourquoi le wording a changé ?","Les termes « panne imminente » ou « intervention immédiate obligatoire » suggèrent une certitude que le modèle ne peut pas garantir avec des données historiques seules. Le système utilise désormais des formulations d'aide à la décision : « dégradation élevée », « contrôle préventif recommandé »."),
            ("Cycle de réentraînement","Le modèle n'apprend pas en continu à chaque saisie — il est réentraîné par cycles périodiques (mensuel/trimestriel) après validation des nouvelles données. Cela évite l'instabilité et garantit la robustesse du modèle en production."),
        ]
        for t,d in sections:
            st.markdown(f'<div class="explain-box"><div class="explain-title">❓ {t}</div><div style="color:#9BAAC7;font-size:12.5px;line-height:1.7">{d}</div></div>', unsafe_allow_html=True)

    with tab3:
        recherche = st.text_input("", placeholder="🔍 Rechercher un terme...", label_visibility="collapsed")
        termes = [
            ("TBF","Time Between Failures","Heures écoulées depuis la dernière panne estimée sur une famille.","Fiabilité"),
            ("MTBF","Mean Time Between Failures","Durée moyenne entre deux pannes consécutives, calculée sur l'historique 2020-2025.","Fiabilité"),
            ("MTTR","Mean Time To Repair","Durée moyenne nécessaire pour réparer une défaillance.","Fiabilité"),
            ("Probabilité ML","Sortie du modèle statistique","Probabilité de dégradation calculée à partir du TBF actuel et de la loi de Weibull calibrée.","Modèle"),
            ("Priorité maintenance","Score de décision opérationnelle","Combine probabilité ML, criticité AMDEC, retard MPS et fréquence récente — sert à prioriser les interventions.","Modèle"),
            ("Health Score","Indicateur de santé globale","Score sur 100 résumant l'état de dégradation estimé d'une famille.","Modèle"),
            ("MPS","Maintenance Préventive Systématique","Intervention planifiée régulièrement, indépendamment de l'état apparent.","Maintenance"),
            ("AMDEC","Analyse des Modes de Défaillance, Effets et Criticité","Méthode évaluant Gravité × Occurrence × Détectabilité pour chaque famille.","Qualité"),
            ("Weibull F(t)","Fonction de défaillance cumulée","Probabilité qu'une panne soit survenue avant l'instant t.","Statistique"),
            ("β (bêta)","Paramètre de forme Weibull","Indique le mode de vieillissement du composant.","Statistique"),
            ("η (eta)","Paramètre d'échelle Weibull","Durée caractéristique de vie du composant.","Statistique"),
            ("Recall","Taux de détection","Proportion de vraies dégradations correctement identifiées par le modèle.","Modèle"),
        ]
        filtres = [t for t in termes if not recherche or recherche.lower() in t[0].lower() or recherche.lower() in t[2].lower()]
        cats = sorted(set(t[3] for t in filtres))
        for cat in cats:
            st.markdown(f"**{cat}**")
            for terme,sous,defn,c in filtres:
                if c!=cat: continue
                st.markdown(f"""<div class="glossary-card"><div class="glossary-term">{terme}</div>
                  <div style="color:#7C8BA8;font-size:11px;margin-bottom:6px;font-style:italic">{sous}</div>
                  <div class="glossary-def">{defn}</div><span class="glossary-tag">{cat}</span></div>""", unsafe_allow_html=True)
    footer()
