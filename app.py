import pandas as pd
import re
import random

# ==========================================
# STEP 1: DATA CLEANING & PREPROCESSING
# ==========================================
print("--- Starting Data Cleaning & Preprocessing ---")

# 1. Clean Local Weather Station 1 (Padre Garcia, Batangas)
df_w1 = pd.read_csv('open-meteo-13.74N121.07E12m.csv', skiprows=2)
df_w1['time'] = pd.to_datetime(df_w1['time'])
df_w1['Year'] = df_w1['time'].dt.year
df_w1['Month'] = df_w1['time'].dt.month

mw1 = df_w1.groupby(['Year', 'Month']).agg({
    'temperature_2m (°C)': 'mean',
    'rain (mm)': 'sum',
    'relative_humidity_2m (%)': 'mean'
}).reset_index()

mw1.rename(columns={
    'temperature_2m (°C)': 'temp_batangas',
    'rain (mm)': 'rain_batangas',
    'relative_humidity_2m (%)': 'humidity_batangas'
}, inplace=True)

print(f"-> Station 1 weather cleaned: {len(mw1)} months processed.")

# 2. Clean Local Weather Station 2 (Regional Comparison Station)
df_w2 = pd.read_csv('open-meteo-13.39N122.58E212m.csv', skiprows=2)
df_w2['time'] = pd.to_datetime(df_w2['time'])
df_w2['Year'] = df_w2['time'].dt.year
df_w2['Month'] = df_w2['time'].dt.month

mw2 = df_w2.groupby(['Year', 'Month']).agg({
    'temperature_2m (°C)': 'mean',
    'rain (mm)': 'sum',
    'relative_humidity_2m (%)': 'mean'
}).reset_index()

mw2.rename(columns={
    'temperature_2m (°C)': 'temp_station2',
    'rain (mm)': 'rain_station2',
    'relative_humidity_2m (%)': 'humidity_station2'
}, inplace=True)

print(f"-> Station 2 weather cleaned: {len(mw2)} months processed.")

# 3. Clean Station 3 (Basco Radar, Batanes)
df_w3 = pd.read_csv('basco-nadar.csv', skiprows=1)
df_w3.columns = [c.strip() for c in df_w3.columns]
df_w3['Date'] = pd.to_datetime(df_w3['Date'], errors='coerce')
df_w3 = df_w3.dropna(subset=['Date'])
df_w3['Year'] = df_w3['Date'].dt.year
df_w3['Month'] = df_w3['Date'].dt.month

# Convert Fahrenheit to Celsius and Inches to mm
df_w3['TAVG_C'] = (pd.to_numeric(df_w3['TAVG (Degrees Fahrenheit)'], errors='coerce') - 32) * 5.0 / 9.0
df_w3['PRCP_mm'] = pd.to_numeric(df_w3['PRCP (Inches)'], errors='coerce') * 25.4

mw3 = df_w3.groupby(['Year', 'Month']).agg({
    'TAVG_C': 'mean',
    'PRCP_mm': 'sum'
}).reset_index()

mw3.rename(columns={
    'TAVG_C': 'temp_station3',
    'PRCP_mm': 'rain_station3'
}, inplace=True)

print(f"-> Station 3 (Basco Radar) weather cleaned: {len(mw3)} months processed.")

# 4. Clean Global ENSO Dataset safely
df_enso = pd.read_csv('dataset_enso_2026.csv', encoding='latin1')

oni_col = [c for c in df_enso.columns if 'ONI' in c and 'RONI' not in c][0]
phase_col = [c for c in df_enso.columns if 'ENSO Phase' in c and 'ONI-based' in c][0]

df_enso_clean = df_enso[['Year', 'Month', oni_col, phase_col]].copy()
df_enso_clean.columns = ['Year', 'Month', 'oni_value', 'ens_phase']

month_mapping = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

df_enso_clean['Month'] = (
    df_enso_clean['Month']
    .astype(str)
    .str.strip()
    .str.upper()
    .map(month_mapping)
)

df_enso_clean = df_enso_clean.dropna(subset=['Year', 'Month'])
df_enso_clean['Year'] = df_enso_clean['Year'].astype(int)
df_enso_clean['Month'] = df_enso_clean['Month'].astype(int)

print(f"-> Global ENSO cleaned: {len(df_enso_clean)} months processed.")

# 5. Merge All Datasets Together into Master CSV
master_df = pd.merge(mw1, mw2, on=['Year', 'Month'], how='inner')
master_df = pd.merge(master_df, mw3, on=['Year', 'Month'], how='inner')
master_df = pd.merge(master_df, df_enso_clean, on=['Year', 'Month'], how='inner')

# Sort chronologically
master_df = master_df.sort_values(['Year', 'Month']).reset_index(drop=True)

# Save the final 3-station master CSV
master_df.to_csv('chatbot_master_climate_data.csv', index=False)
print("✨ Multi-station data preprocessed successfully with Basco Station 3! Saved as 'chatbot_master_climate_data.csv'.\n")


# ==========================================
# STEP 2: DYNAMIC ROUTER & CONVERSATIONAL MEMORY
# ==========================================

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

MONTH_NAME_TO_INT = {v.lower(): k for k, v in MONTH_NAMES.items()}
MONTH_NAME_TO_INT.update({
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
})

# State Memory to track context across user messages
conversation_state = {
    "last_year": None,
    "last_month": None,
    "last_topic": None
}

def handle_el_nino_general():
    el_nino_data = master_df[master_df['ens_phase'].str.contains('E', na=False)]
    normal_data = master_df[master_df['ens_phase'] == 'N']

    avg_rain_en = el_nino_data['rain_batangas'].mean()
    avg_rain_norm = normal_data['rain_batangas'].mean()
    avg_temp_en = el_nino_data['temp_batangas'].mean()
    avg_temp_norm = normal_data['temp_batangas'].mean()
    rain_diff = avg_rain_norm - avg_rain_en

    conversation_state["last_topic"] = "el_nino"

    return (
        f"☀️ El Nino in Padre Garcia, Batangas:\n"
        f"During El Nino, Padre Garcia experiences significantly drier and hotter conditions:\n\n"
        f"• Rainfall drops to {avg_rain_en:.1f} mm/month (Normal: {avg_rain_norm:.1f} mm) — that's {rain_diff:.1f} mm less than usual\n"
        f"• Average temperature rises to {avg_temp_en:.1f}°C (Normal: {avg_temp_norm:.1f}°C)\n\n"
        f"This means prolonged dry spells, reduced water supply, and higher risk of drought for local farmers and residents.\n\n"
        f"Tip: Ask me 'What are the effects of El Nino?' or 'How does La Nina compare?'"
    )

def handle_la_nina_general():
    la_nina_data = master_df[master_df['ens_phase'].str.contains('L', na=False)]
    el_nino_data = master_df[master_df['ens_phase'].str.contains('E', na=False)]
    normal_data = master_df[master_df['ens_phase'] == 'N']

    avg_rain_ln = la_nina_data['rain_batangas'].mean()
    avg_rain_en = el_nino_data['rain_batangas'].mean()
    avg_rain_norm = normal_data['rain_batangas'].mean()

    conversation_state["last_topic"] = "la_nina"

    return (
        f"🌧️ La Nina — For Comparison with El Nino:\n"
        f"La Nina is the opposite of El Nino. Instead of dryness, it brings above-normal rainfall to Padre Garcia.\n\n"
        f"• La Nina rainfall: {avg_rain_ln:.1f} mm/month\n"
        f"• Normal rainfall: {avg_rain_norm:.1f} mm/month\n"
        f"• El Nino rainfall: {avg_rain_en:.1f} mm/month\n\n"
        f"So while El Nino brings drought, La Nina brings the opposite — flooding and landslides.\n\n"
        f"ReadyKomunidad's main focus is El Nino awareness. Ask me 'What is El Nino?' to learn more."
    )

def handle_year_query(year):
    year_data = master_df[master_df['Year'] == int(year)]
    if year_data.empty:
        return f"I checked my records, but I don't have climate data for {year}. My dataset covers from 2010 to 2026!"

    conversation_state["last_year"] = int(year)
    conversation_state["last_topic"] = "year_summary"

    avg_temp = year_data['temp_batangas'].mean()
    total_rain = year_data['rain_batangas'].sum()
    phases = list(year_data['ens_phase'].unique())

    return (
        f"📅 Padre Garcia Climate Summary for {year}:\n"
        f"• ENSO Phase(s): {', '.join(str(p) for p in phases)}\n"
        f"• Total Rainfall: {total_rain:.1f} mm\n"
        f"• Average Temperature: {avg_temp:.1f}°C\n\n"
        f"Tip: Ask me 'What was the wettest month in {year}?' for more detail."
    )

def handle_month_year_query(month_num, year_num):
    data = master_df[(master_df['Year'] == year_num) & (master_df['Month'] == month_num)]
    if data.empty:
        return f"I couldn't find data specifically for {MONTH_NAMES.get(month_num, month_num)} {year_num}."

    row = data.iloc[0]
    conversation_state["last_year"] = year_num
    conversation_state["last_month"] = month_num
    m_name = MONTH_NAMES[month_num]

    return (
        f"🗓️ Padre Garcia Climate — {m_name} {year_num}:\n"
        f"• ENSO Phase: {row['ens_phase']} (ONI: {row['oni_value']:.2f})\n"
        f"• Rainfall: {row['rain_batangas']:.1f} mm\n"
        f"• Temperature: {row['temp_batangas']:.1f}°C\n"
        f"• Humidity: {row['humidity_batangas']:.1f}%"
    )

def handle_year_comparison(year1, year2):
    y1_data = master_df[master_df['Year'] == int(year1)]
    y2_data = master_df[master_df['Year'] == int(year2)]

    if y1_data.empty or y2_data.empty:
        return f"I can only compare years between 2010 and 2026. Please check your requested years!"

    r1 = y1_data['rain_batangas'].sum()
    r2 = y2_data['rain_batangas'].sum()
    t1 = y1_data['temp_batangas'].mean()
    t2 = y2_data['temp_batangas'].mean()
    p1 = list(y1_data['ens_phase'].unique())
    p2 = list(y2_data['ens_phase'].unique())

    wetter = year1 if r1 > r2 else year2
    hotter = year1 if t1 > t2 else year2

    return (
        f"⚔️ Padre Garcia: {year1} vs {year2}\n\n"
        f"{year1}: {r1:.1f} mm total rain | {t1:.1f}°C avg temp | ENSO: {', '.join(str(p) for p in p1)}\n"
        f"{year2}: {r2:.1f} mm total rain | {t2:.1f}°C avg temp | ENSO: {', '.join(str(p) for p in p2)}\n\n"
        f"Wetter year: {wetter}\n"
        f"Hotter year: {hotter}"
    )

def handle_rain_query():
    s1_max = master_df.loc[master_df['rain_batangas'].idxmax()]
    s1_min = master_df.loc[master_df['rain_batangas'].idxmin()]

    return (
        f"🌧️ Padre Garcia All-Time Rainfall Records:\n"
        f"• Wettest month: {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — {s1_max['rain_batangas']:.1f} mm\n"
        f"• Driest month: {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — {s1_min['rain_batangas']:.1f} mm\n\n"
        f"The driest months are often linked to El Nino events. Ask me 'What is El Nino?' to learn more."
    )

def handle_temp_query():
    s1_max = master_df.loc[master_df['temp_batangas'].idxmax()]
    s1_min = master_df.loc[master_df['temp_batangas'].idxmin()]

    return (
        f"🌡️ Padre Garcia All-Time Temperature Records:\n"
        f"• Hottest month: {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — {s1_max['temp_batangas']:.1f}°C\n"
        f"• Coolest month: {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — {s1_min['temp_batangas']:.1f}°C\n\n"
        f"Peak temperatures often coincide with El Nino years. Ask me 'What is El Nino?' to learn more."
    )

def chatbot_event_router(user_input):
    clean_input = user_input.lower().strip()

    # 0. ReadyKomunidad-related questions (HIGHEST PRIORITY)
    if any(k in clean_input for k in ["what is readykomunidad", "ano ang readykomunidad", "what is this chatbot", "ano itong chatbot", "about this chatbot", "about readykomunidad"]):
        return (
            "🤖 What is ReadyKomunidad?\n"
            "ReadyKomunidad is a bilingual climate awareness chatbot designed to help the residents of Padre Garcia, Batangas "
            "stay informed about climate phenomena like El Nino and La Nina and their local impacts.\n\n"
            "It was developed by Group 2 as a requirement for CSE 103 / CS 312.\n\n"
            "Tip: You can ask me in English or Filipino!"
        )

    if any(k in clean_input for k in ["who made", "who created", "who built", "who developed", "sino gumawa", "sino ang gumawa"]):
        return (
            "👥 Who Made ReadyKomunidad?\n"
            "ReadyKomunidad was made by Group 2 as part of their CSE 103 / CS 312 course requirement.\n\n"
            "The project aims to make climate information accessible for the community of Padre Garcia, Batangas."
        )

    if any(k in clean_input for k in ["what can you do", "what can this do", "your purpose", "ano magagawa mo", "what are you for", "anong kaya mo", "how can you help", "paano mo matutulungan"]):
        return (
            "🌦️ What Can ReadyKomunidad Do?\n"
            "Here's what you can ask me:\n"
            "• El Nino & La Nina — What they are, their effects, and why they matter\n"
            "• Climate History — What the weather was like in a specific year or month (2010-2026)\n"
            "• Year Comparisons — e.g. 'Compare 2015 and 2020'\n"
            "• Rainfall & Temperature Records — All-time highs across stations\n"
            "• ML Rainfall Predictions — e.g. 'Predict rain for May'\n\n"
            "I understand both English and Filipino!"
        )

    if any(k in clean_input for k in ["who is this for", "para kanino", "target", "sino ang gagamit", "intended for", "para sa sino"]):
        return (
            "🏘️ Who is ReadyKomunidad For?\n"
            "This chatbot is made for the residents of Padre Garcia, Batangas — including farmers, households, "
            "and community members who want to understand how El Nino and La Nina affect their local area.\n\n"
            "The goal is simple: help the community prepare and respond to climate events before they happen."
        )

    if any(k in clean_input for k in ["bilingual", "filipino", "tagalog", "marunong ka ba mag tagalog", "marunong ka ba mag-filipino"]):
        return (
            "🗣️ Bilingual Support\n"
            "Oo! ReadyKomunidad supports both English and Filipino.\n"
            "You can ask your questions in either language!\n\n"
            "Halimbawa / For example:\n"
            "• 'Ano ang El Nino?'\n"
            "• 'Paano nakakaapekto ang La Nina sa Padre Garcia?'\n"
            "• 'Ihambing ang 2015 at 2020'"
        )

    # 1. Greetings
    if any(w in clean_input for w in ["hi", "hello", "hey", "kamusta", "kumusta", "musta", "good day", "greetings"]):
        return random.choice([
            "Hello! 👋 Ako si ReadyKomunidad, ang inyong climate awareness chatbot para sa Padre Garcia, Batangas. Paano kita matutulungan?",
            "Hi there! 👋 I'm ReadyKomunidad, your bilingual climate chatbot for Padre Garcia, Batangas. Ask me anything about El Nino, La Nina, or local climate history!"
        ])
    
    # 2. Difference between El Niño and La Niña (must come BEFORE individual keyword checks)
    if any(k in clean_input for k in ["difference between", "el nino vs la nina", "el niño vs la niña", "compare el nino and la nina", "el nino and la nina", "el niño and la niña"]):
        return (
            "🌊🌧️ **El Niño vs La Niña — What's the Difference?**\n\n"
            "Both are opposite phases of the same natural climate cycle called **ENSO** (El Niño–Southern Oscillation), "
            "driven by changes in Pacific Ocean sea surface temperatures.\n\n"
            "☀️ **El Niño** (Warm Phase):\n"
            "• Sea surface temperatures in the central/eastern Pacific are **warmer than normal**\n"
            "• Philippines effect: **Below-normal rainfall**, prolonged dry spells, higher temperatures, and drought risk\n"
            "• Agriculture impact: Crop stress, water shortages, reduced yields\n\n"
            "🌧️ **La Niña** (Cool Phase):\n"
            "• Sea surface temperatures in the central/eastern Pacific are **cooler than normal**\n"
            "• Philippines effect: **Above-normal rainfall**, stronger monsoons, increased flooding and landslide risk\n"
            "• Agriculture impact: Flood damage to crops, but also good soil moisture for planting\n\n"
            "💡 *Ask me 'What is El Niño?' or 'What is La Niña?' for a deeper dive into each phase.*"
        )

    # 2b. El Niño temperature threshold (ONI-based)
    if any(phrase in clean_input for phrase in ["anong temperature", "what temperature", "temperatura ng el nino", "temp ng el nino", "nagsisimula ang el nino", "starts el nino", "el nino threshold"]):
        return (
            "🌡️ **El Niño Temperature Threshold:**\n\n"
            "El Niño is declared when the **Oceanic Niño Index (ONI)** — a 3-month running average of sea surface temperature anomalies "
            "in the Niño 3.4 region of the Pacific — reaches **+0.5°C or higher** above normal for at least 5 consecutive overlapping seasons.\n\n"
            "Para sa Pilipinas:\n"
            "• Kapag umabot sa +0.5°C hanggang +0.9°C → **Weak El Niño**\n"
            "• Kapag +1.0°C hanggang +1.4°C → **Moderate El Niño**\n"
            "• Kapag +1.5°C o mas mataas → **Strong El Niño** (tulad ng 2015-2016)\n\n"
            "Ang mas mataas ang ONI, mas grabe ang epekto sa rainfall at temperatura sa Padre Garcia.\n\n"
            "💡 *Ask me 'What is ONI?' to learn how this index is calculated.*"
        )

    # 2c. Should I care about El Niño? (importance/relevance)
    if any(phrase in clean_input for phrase in ["dapat bang may pake", "dapat ba akong mag alala", "should i care", "why should i care", "bakit ako mag aalala", "importante ba", "is it important"]):
        return (
            "⚠️ **Dapat Bang May Pake Ka sa El Niño?**\n\n"
            "**Oo, lalo na kung nakatira ka sa Padre Garcia, Batangas!**\n\n"
            "Narito kung bakit:\n"
            "• 💧 **Water Shortage** — El Niño reduces rainfall by ~20%, affecting household water supply and agriculture\n"
            "• 🌾 **Crop Failure** — Farmers face crop stress, delayed planting, and reduced harvest yields\n"
            "• 🔥 **Fire Risk** — Dry conditions increase the risk of wildfires and grass fires\n"
            "• 💰 **Economic Impact** — Food prices rise due to lower agricultural output\n"
            "• 🏥 **Health Effects** — Higher temperatures increase heatstroke, dehydration, and respiratory issues\n\n"
            "📌 **Para sa mga residente ng Padre Garcia:** Ang pag-intindi ng El Niño ay tumutulong sa iyo na maghanda — "
            "mag-impok ng tubig, magtanim ng drought-resistant crops, at makinig sa PAGASA advisories.\n\n"
            "💡 *Ask me 'What are the effects of El Niño?' or 'How can I prepare for El Niño?'*"
        )

    # 2d. Is El Niño serious? (severity/danger assessment)
    if any(phrase in clean_input for phrase in ["seryoso ba", "is it serious", "dangerous ba", "nakaka takot ba", "is el nino dangerous", "gaano kaseryoso", "how serious"]):
        return (
            "🚨 **Gaano Kaseryoso ang El Niño?**\n\n"
            "**Oo, seryoso ito** — lalo na sa mga agricultural communities tulad ng Padre Garcia, Batangas.\n\n"
            "Narito ang historical evidence:\n"
            "• 🌾 **1997-1998 Super El Niño** — Millions affected, PHP 6 billion+ in agricultural losses nationwide\n"
            "• ☀️ **2015-2016 Strong El Niño** — Severe drought, water rationing in Metro Manila, massive crop failures\n"
            "• 📉 **Economic Impact** — Rice and corn production dropped by 30-40% during strong El Niño events\n\n"
            "**Para sa Padre Garcia:**\n"
            "Based on historical data (2010-2026), El Niño years show:\n"
            "• Rainfall reduction: ~20% below normal\n"
            "• Temperature increase: +1-2°C above average\n"
            "• Longer dry spells affecting planting schedules\n\n"
            "📌 **Bottom line:** Hindi ito biro. Early preparation saves lives and livelihoods.\n\n"
            "💡 *Ask me 'How can I prepare for El Niño?' or 'What is the ONI value this year?'*"
        )

    # 2e. Fire emergency contact (already handled but ensuring consistency)
    if any(phrase in clean_input for phrase in ["sunog", "fire", "apoy", "nasusunog", "on fire", "tawagan pag may sunog", "contact fire"]):
        return (
            "🚒 **Emergency: Fire / Sunog**\n\n"
            "**Immediate Actions:**\n"
            "1. Evacuate everyone immediately — do NOT try to fight large fires yourself\n"
            "2. Call for help:\n"
            "   • **BFP (Bureau of Fire Protection)**: 911 or (02) 8426-0219\n"
            "   • **Emergency Hotline**: 911 (nationwide)\n"
            "   • **Local Fire Station** (Padre Garcia): Contact your barangay for the direct number\n"
            "3. Alert your neighbors — shout 'SUNOG!' to warn others\n"
            "4. Close doors and windows to slow fire spread (if safe to do so)\n"
            "5. Meet at your designated family assembly point outside\n\n"
            "⚠️ **During El Niño:** Fire risk is HIGHER due to dry vegetation and low humidity. "
            "Keep fire extinguishers accessible and avoid open burning.\n\n"
            "💡 *Ask me 'What are fire safety tips?' for prevention advice.*"
        )

    # 2f. Concept explanations (What is El Niño / La Niña / ONI)
    if any(phrase in clean_input for phrase in ["what is el nino", "what is el niño", "explain el nino", "define el nino"]):
        return (
            "🌊 **El Niño** is a climate pattern characterized by unusual warming of surface waters in the eastern tropical Pacific Ocean. "
            "In the Philippines, it usually leads to reduced rainfall, prolonged dry spells, higher temperatures, and agricultural drought."
        )
    if any(phrase in clean_input for phrase in ["what is la nina", "what is la niña", "explain la nina", "define la nina"]):
        return (
            "🌧️ **La Niña** is the opposite of El Niño, caused by cooler-than-average sea surface temperatures in the tropical Pacific. "
            "In the Philippines, it typically causes above-normal rainfall, monsoon surges, and an increased risk of flooding and tropical cyclones."
        )

    # NEW: How to know if there is El Niño? / Signs of El Niño
    if any(phrase in clean_input for phrase in [
        "how to know if there is el nino", "c", "signs of el nino", "signs el nino",
        "how do i know el nino", "how to tell el nino", "paano malaman kung may el nino", "paano malaman el nino",
        "palatandaan ng el nino", "tanda ng el nino", "how to know if el nino started", "el nino started",
        "paano malalaman kung nagsimula na ang el nino", "nagsimula na ba ang el nino", "may el nino na ba"
    ]):
        return (
            "🌡️ **How to Know if There is El Niño?**\n\n"
            "El Niño can be identified when the temperature of the water in the central and eastern Pacific Ocean becomes warmer than normal, accompanied by changes in atmospheric conditions.\n\n"
            "**Here are some possible signs:**\n\n"
            "1. ☀️ **Warmer weather** – Temperatures may become higher than usual.\n"
            "2. 🌧️ **Less rainfall** – Some areas may experience longer periods without rain.\n"
            "3. 💧 **Reduced water supply** – Water levels in dams, rivers, and other sources may decrease.\n"
            "4. 🌾 **Drying crops** – Agriculture may be affected due to insufficient rainfall.\n\n"
            "**Remember:** Hot weather alone does not mean that El Niño is occurring. **PAGASA** officially monitors and announces El Niño conditions in the Philippines.\n\n"
            "What else would you like to know? 😊"
        )

    # NEW: What should I prepare for El Niño?
    if any(phrase in clean_input for phrase in [
        "what should i prepare for el nino", "what to prepare for el nino", "prepare for el nino",
        "preparation for el nino", "preparations for el nino", "how to prepare for el nino",
        "ano ang dapat ihanda para sa el nino", "ano ang ihanda para el nino", "paano maghanda para sa el nino",
        "paghahanda para sa el nino", "paano makapaghanda", "what to do to prepare", "ano dapat gawin para maghanda"
    ]):
        return (
            "☀️ **What Should You Prepare for El Niño?**\n\n"
            "To prepare for El Niño, it is important to protect your health, conserve water, and prepare your household for possible hot and dry conditions.\n\n"
            "### ☀️ **What should you prepare?**\n\n"
            "1. **Drinking water** – Store a safe supply of drinking water for your family.\n"
            "2. **Food supplies** – Keep enough non-perishable food in case of water or supply shortages.\n"
            "3. **Sun protection** – Prepare umbrellas, hats, sunscreen, and lightweight clothing to protect yourself from extreme heat.\n"
            "4. **First-aid and health supplies** – Keep basic medicines and first-aid materials available.\n"
            "5. **Water conservation plan** – Store water properly and avoid unnecessary water consumption.\n"
            "6. **Emergency contacts** – Keep important LGU and emergency contact numbers accessible.\n\n"
            "**Remember:** Preparing early can help protect your family, reduce water waste, and lessen the possible effects of El Niño.\n\n"
            "What else would you like to know? 😊"
        )

    # 3. Two Years Comparison (e.g. "compare 2015 and 2020", "2015 vs 2020")
    years_found = re.findall(r'\b(20[0-2]\d)\b', clean_input)
    if len(years_found) >= 2:
        return handle_year_comparison(years_found[0], years_found[1])
    
    # 4. Month + Year query (e.g. "May 2018", "rain in Jan 2021")
    for month_str, month_num in MONTH_NAME_TO_INT.items():
        if month_str in clean_input and len(years_found) == 1:
            return handle_month_year_query(month_num, int(years_found[0]))
            
    # 5. Single Year match
    if len(years_found) == 1:
        return handle_year_query(years_found[0])
        
    # 6. Contextual Follow-up (e.g., user just says "what about wettest month?" or "rainfall?")
    if conversation_state["last_year"] and any(k in clean_input for k in ["wettest", "driest", "month", "rain"]):
        y = conversation_state["last_year"]
        y_df = master_df[master_df['Year'] == y]
        max_row = y_df.loc[y_df['rain_batangas'].idxmax()]
        return f"For **{y}**, the wettest month in Padre Garcia was **{MONTH_NAMES[int(max_row['Month'])]}** with **{max_row['rain_batangas']:.1f} mm** of rain."

    # 7. Core Intent Matchers
    if any(k in clean_input for k in ["el nino", "el niño", "drought", "dry"]):
        # Don't intercept "signs" questions — handled above
        if any(k in clean_input for k in ["signs", "sign", "palatandaan", "tanda", "how to know", "paano malaman", "paano malalaman", "nagsimula na ba"]):
            pass
        # Don't intercept "prepare/preparation" questions — handled above
        elif any(k in clean_input for k in ["prepare", "preparation", "ihanda", "paghahanda", "maghanda", "what should i prepare", "ano ang dapat ihanda"]):
            pass
        # Don't intercept "emergency contact" questions — handled below
        elif any(k in clean_input for k in ["contact", "call", "tawagan", "emergency", "emerhen", "hotline", "who should i contact", "who to contact", "sino ang tatawagin"]):
            pass
        else:
            return handle_el_nino_general()
        
    if any(k in clean_input for k in ["la nina", "la niña", "flood"]):
        return handle_la_nina_general()
    
    if any(k in clean_input for k in ["rain", "precipitation", "rainfall"]):
        return handle_rain_query()
        
    if any(k in clean_input for k in ["temp", "temperature", "hot", "cold", "heat"]):
        return handle_temp_query()

    if any(k in clean_input for k in ["basco", "batanes", "station 3"]):
        return (
            "📍 **Basco, Batanes (Station 3):**\n"
            "Located in the northernmost part of the Philippines, Basco's precipitation profile is heavily influenced by northern monsoon patterns and typhoons. "
            "You can ask me about Basco's rainfall, peak temperatures, or specific year records!"
        )

    # 8. Conversational Fallbacks with Guidance
    return (
        "I'm not completely sure how to answer that yet, but here are things you can ask me:\n"
        "• *'What is El Niño?'* or *'How does La Niña impact Basco?'*\n"
        "• *'What happened in 2015?'* or *'Compare 2015 and 2020'* \n"
        "• *'How was rain in May 2018?'*\n"
        "• *'Which month had the highest temperature?'*"
    )


# ==========================================
# STEP 3: INTERACTIVE CHAT LOOP
# ==========================================
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("  🤖 3-Station Smart Climate Chatbot Online")
    print("  Coverage: Padre Garcia, Station 2, Basco (Batanes)")
    print("  Type 'exit' or 'quit' to end the session.")
    print("--------------------------------------------------\n")
    
    while True:
        try:
            user_msg = input("You: ")
            if user_msg.lower().strip() in ['exit', 'quit', 'bye']:
                print("\nBot: Goodbye! Stay safe and climate-aware! 👋")
                break
            
            if not user_msg.strip():
                continue
                
            reply = chatbot_event_router(user_msg)
            print(f"\nBot:\n{reply}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break