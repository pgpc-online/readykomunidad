from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import re
import random
import os
import uuid
import joblib
import db

app = Flask(__name__)

# Initialise SQLite storage (chatlogs + knowledge base) on startup.
db.init_db()

DATA_FILE = 'chatbot_master_climate_data.csv'
MODEL_FILE = 'climate_rain_model.pkl'

if os.path.exists(DATA_FILE):
    master_df = pd.read_csv(DATA_FILE)
    print(f"Loaded master climate dataset with {len(master_df)} rows.")
else:
    master_df = None
    print(f"Warning: {DATA_FILE} not found. Run preprocessing first.")

if os.path.exists(MODEL_FILE):
    rain_model = joblib.load(MODEL_FILE)
    print("Loaded local ML rain forecasting model successfully.")
else:
    rain_model = None
    print("Warning: 'climate_rain_model.pkl' not found. Run 'train_model.py' first.")

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

# ENSO phase code translations
ENSO_LABELS = {
    'N':   'Neutral 🟡',
    'WE':  'Weak El Niño ☀️',
    'SE':  'Strong El Niño ☀️☀️',
    'VSE': 'Very Strong El Niño ☀️☀️☀️',
    'WL':  'Weak La Niña 🌧️',
    'ML':  'Moderate La Niña 🌧️🌧️',
    'SL':  'Strong La Niña 🌧️🌧️🌧️',
}

def label_phase(code):
    """Convert raw ENSO code to human-readable label."""
    if pd.isna(code):
        return 'Unknown'
    return ENSO_LABELS.get(str(code).strip(), str(code))

def label_oni(value):
    """Return a human-readable interpretation of an ONI value."""
    try:
        v = float(value)
    except:
        return str(value)
    if v >= 2.0:   return f"{v:.2f} — Very Strong El Niño ☀️☀️☀️"
    elif v >= 1.5: return f"{v:.2f} — Strong El Niño ☀️☀️"
    elif v >= 0.5: return f"{v:.2f} — Weak/Moderate El Niño ☀️"
    elif v >= -0.4: return f"{v:.2f} — Neutral 🟡"
    elif v >= -1.0: return f"{v:.2f} — Weak La Niña 🌧️"
    elif v >= -1.5: return f"{v:.2f} — Moderate La Niña 🌧️🌧️"
    else:           return f"{v:.2f} — Strong La Niña 🌧️🌧️🌧️"

conversation_state = {
    "last_topic": None,
    "last_year": None,
    "last_month": None,
    "lang": "en",
    "station_mode": False  # Default: Padre Garcia only!
}

# Common Tagalog words/particles that signal the user is speaking Filipino
TAGALOG_MARKERS = [
    "ano", "ang", "sa", "na", "ay", "mga", "ng", "paano", "bakit", "sino",
    "kailan", "magkano", "ilan", "hindi", "oo", "hulaan", "prediksyon",
    "tungkol", "gumawa", "nagawa", "marunong", "magtagalog", "magfilipino",
    "ihambing", "pagkukumpara", "ulan", "pag-ulan", "temperatura", "init",
    "maghanda", "gagawin", "epekto", "mahalaga", "kailangan", "bakit",
    "taga", "komunidad", "para", "tulong", "sagot", "tanong", "alamin",
    "kamusta", "kumusta", "musta", "salamat", "opo", "oho", "wala",
    "meron", "mayroon", "kapag", "pag", "nga", "din", "rin", "ba", "po",
    "namin", "amin", "kayo", "tayo", "siya", "ito", "iyan", "iyon",
    "dahil", "kung", "para", "upang", "pwede", "maaari", "ginawa", "gagawin", "ipredict", "icompare", "i-predict", "i-compare",
    "paalam", "hanggang", "maraming", "walang", "anuman", "mag-ingat", "ingat",
    # Greeting words
    "magandang", "umaga", "tanghali", "hapon", "gabi", "mabuhay"
]

def detect_language(text):
    """Detect whether the user is speaking Tagalog/Filipino or English."""
    words = set(text.lower().split())
    tagalog_count = sum(1 for w in words if w in TAGALOG_MARKERS)
    # If at least 1 distinct Tagalog marker is found, treat as Tagalog
    if tagalog_count >= 1:
        return "tl"
    return "en"

def L(en, tl):
    """Return the text in the current conversation language."""
    return tl if conversation_state.get("lang") == "tl" else en

# ==========================================
# MODERN COMPARISON CHART & CARD GENERATOR
# ==========================================

def render_comparison_chart_card(year1, year2, y1_data, y2_data, is_tl=False, incomplete_note=""):
    """
    Renders an interactive, side-by-side comparison bar chart card for two years.
    Uses blue bars for Year 1 and amber/orange bars for Year 2.
    """
    months_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    vals1 = []
    vals2 = []
    for m in range(1, 13):
        r1 = y1_data[y1_data['Month'] == m]
        r2 = y2_data[y2_data['Month'] == m]
        vals1.append(float(r1['rain_batangas'].values[0]) if not r1.empty and pd.notna(r1['rain_batangas'].values[0]) else 0.0)
        vals2.append(float(r2['rain_batangas'].values[0]) if not r2.empty and pd.notna(r2['rain_batangas'].values[0]) else 0.0)

    total1 = sum(vals1)
    total2 = sum(vals2)
    avg_temp1 = y1_data['temp_batangas'].mean()
    avg_temp2 = y2_data['temp_batangas'].mean()

    # ENSO phase labels
    p1 = list(y1_data['ens_phase'].dropna().unique())
    p2 = list(y2_data['ens_phase'].dropna().unique())
    p1_labels = ', '.join(label_phase(p) for p in p1) if p1 else "Neutral 🟡"
    p2_labels = ', '.join(label_phase(p) for p in p2) if p2 else "Neutral 🟡"

    # Difference calculation
    diff = abs(total1 - total2)
    wetter_year = year1 if total1 > total2 else year2
    drier_year = year2 if total1 > total2 else year1
    diff_pct = (diff / min(total1, total2) * 100) if min(total1, total2) > 0 else 0

    # Chart Y-axis scaling
    peak = max(max(vals1), max(vals2), 10.0)
    top_limit = peak * 1.15
    if top_limit <= 100:
        step = 25
    elif top_limit <= 200:
        step = 45
    elif top_limit <= 400:
        step = 80
    else:
        step = int(top_limit / 4 / 25) * 25 or 50

    top_y = step * 4
    if top_y < peak:
        top_y = step * 5

    y_levels = [top_y, step * 3, step * 2, step * 1, 0]

    # SVG layout specs
    chart_height = 210
    chart_width = 560
    margin_left = 60
    margin_right = 20
    plot_width = chart_width - margin_left - margin_right
    slot_width = plot_width / 12
    bar_width = 9  # Two bars side by side per month

    bars_svg = []
    labels_svg = []

    for i, (v1, v2, abbr) in enumerate(zip(vals1, vals2, months_abbr)):
        x_slot = margin_left + (i * slot_width)
        
        # Bar 1 (Year 1 - Blue)
        b1_x = x_slot + (slot_width - (bar_width * 2 + 3)) / 2
        b1_h = (v1 / top_y) * (chart_height - 45) if top_y > 0 else 0
        b1_y = (chart_height - 30) - b1_h

        # Bar 2 (Year 2 - Amber)
        b2_x = b1_x + bar_width + 3
        b2_h = (v2 / top_y) * (chart_height - 45) if top_y > 0 else 0
        b2_y = (chart_height - 30) - b2_h

        tooltip_x = b1_x + bar_width
        tooltip_y = max(min(b1_y, b2_y) - 34, 4)

        bars_svg.append(f"""
            <g class="bar-group group cursor-pointer">
                <!-- Bar 1 ({year1}) -->
                <rect x="{b1_x:.1f}" y="{b1_y:.1f}" width="{bar_width}" height="{max(b1_h, 3):.1f}" rx="3"
                      class="fill-blue-500 hover:fill-blue-600 dark:fill-blue-400 transition-all duration-150" />
                <!-- Bar 2 ({year2}) -->
                <rect x="{b2_x:.1f}" y="{b2_y:.1f}" width="{bar_width}" height="{max(b2_h, 3):.1f}" rx="3"
                      class="fill-amber-500 hover:fill-amber-600 dark:fill-amber-400 transition-all duration-150" />
                <!-- Hover Dual Tooltip -->
                <g class="bar-tooltip opacity-0 transition-all duration-150 pointer-events-none">
                    <rect x="{tooltip_x - 45:.1f}" y="{tooltip_y:.1f}" width="90" height="28" rx="6"
                          class="fill-slate-900 dark:fill-slate-100 shadow-md" />
                    <text x="{tooltip_x:.1f}" y="{tooltip_y + 12:.1f}" text-anchor="middle"
                          class="fill-blue-300 dark:fill-blue-600 text-[9px] font-semibold">{year1}: {v1:.0f} mm</text>
                    <text x="{tooltip_x:.1f}" y="{tooltip_y + 23:.1f}" text-anchor="middle"
                          class="fill-amber-300 dark:fill-amber-600 text-[9px] font-semibold">{year2}: {v2:.0f} mm</text>
                </g>
            </g>
        """)

        labels_svg.append(f"""
            <text x="{b1_x + bar_width + 1.5:.1f}" y="{chart_height - 10}" text-anchor="middle"
                  class="fill-slate-500 dark:fill-slate-400 text-[10px] font-medium">{abbr}</text>
        """)

    grid_svg = []
    for lvl in y_levels:
        y_pos = 14 + (chart_height - 44) * (1.0 - (lvl / top_y)) if top_y > 0 else 14
        grid_svg.append(f"""
            <line x1="{margin_left}" y1="{y_pos:.1f}" x2="{chart_width - margin_right}" y2="{y_pos:.1f}"
                  stroke="currentColor" stroke-dasharray="3,3" class="stroke-slate-200 dark:stroke-slate-800" stroke-width="1"/>
            <text x="{margin_left - 8}" y="{y_pos + 3.5:.1f}" text-anchor="end"
                  class="fill-slate-400 dark:fill-slate-500 text-[10px]">{int(lvl)} mm</text>
        """)

    # Localized text strings
    badge_text = "Historical Comparison · Padre Garcia, Batangas"
    header_title = f"Rainfall Comparison — {year1} vs {year2}" if not is_tl else f"Paghahambing ng Ulan — {year1} vs {year2}"
    chart_title = f"Monthly Rainfall Comparison ({year1} vs {year2})" if not is_tl else f"Buwanang Paghahambing ng Ulan ({year1} vs {year2})"
    chart_subtitle = "Hover over bars to see monthly measurements side-by-side." if not is_tl else "Itapat ang cursor sa bar upang makita ang buwanang sukat."

    summary_headline = (
        f"In Padre Garcia, <strong>{wetter_year}</strong> recorded <strong>{diff:.1f} mm</strong> (+{diff_pct:.0f}%) more rainfall than {drier_year}."
        if not is_tl else
        f"Sa Padre Garcia, ang taong <strong>{wetter_year}</strong> ay nagtala ng <strong>{diff:.1f} mm</strong> (+{diff_pct:.0f}%) mas maraming ulan kaysa sa {drier_year}."
    )

    verdict_badge = (
        f"📈 {wetter_year} was wetter by {diff:.0f} mm" if not is_tl else f"📈 Mas maulan noong {wetter_year} ng {diff:.0f} mm"
    )

    note_html = f"<div class='p-2.5 rounded-lg bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 text-xs mb-2 border border-amber-200/60 dark:border-amber-800/60'>{incomplete_note}</div>" if incomplete_note else ""

    return f"""<!-- RAINFALL_CARD -->
<div class="w-full my-2 space-y-4 animate-fade-in font-sans">
    {note_html}
    <!-- Header -->
    <div>
        <div class="flex items-center gap-2 mb-1.5 flex-wrap">
            <!-- <span class="text-base">⚖️</span> -->
            <span class="font-semibold text-slate-900 dark:text-white text-base">{header_title}</span>
            <span class="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-100 text-blue-800 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200/60 dark:border-blue-800/60">
                {badge_text}
            </span>
        </div>
        <p class="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{summary_headline}</p>
    </div>

    <!-- Chart Container Card -->
    <div class="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 shadow-xs">
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
            <div>
                <h4 class="font-semibold text-slate-900 dark:text-white text-sm sm:text-base">{chart_title}</h4>
                <p class="text-[11px] sm:text-xs text-slate-400 dark:text-slate-500 mt-0.5">{chart_subtitle}</p>
            </div>
            <!-- Legend -->
            <div class="flex items-center gap-3 text-xs self-start sm:self-auto pt-1">
                <span class="inline-flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
                    <span class="w-3 h-3 rounded-xs bg-blue-500 inline-block"></span>
                    <span>{year1}</span>
                </span>
                <span class="inline-flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
                    <span class="w-3 h-3 rounded-xs bg-amber-500 inline-block"></span>
                    <span>{year2}</span>
                </span>
            </div>
        </div>

        <!-- SVG Grouped Bar Chart -->
        <div class="w-full overflow-x-auto">
            <svg viewBox="0 0 {chart_width} {chart_height}" class="w-full h-52 sm:h-60 select-none">
                {''.join(grid_svg)}
                {''.join(bars_svg)}
                {''.join(labels_svg)}
            </svg>
        </div>
    </div>

    <!-- 2-Column Side-by-Side Metric Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        <!-- Year 1 Card -->
        <div class="p-4 rounded-xl bg-white dark:bg-slate-900/90 border border-blue-200/80 dark:border-blue-900/50 shadow-xs">
            <div class="flex items-center justify-between mb-2">
                <span class="font-semibold text-slate-900 dark:text-white text-sm sm:text-base flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                    <span>{year1} Summary</span>
                </span>
                <span class="text-[11px] px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">{p1_labels}</span>
            </div>
            <div class="space-y-1 text-xs text-slate-600 dark:text-slate-300">
                <div class="flex justify-between py-0.5 border-b border-slate-100 dark:border-slate-800/60">
                    <span class="text-slate-400">Total Rainfall:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{total1:.1f} mm</strong>
                </div>
                <div class="flex justify-between py-0.5 border-b border-slate-100 dark:border-slate-800/60">
                    <span class="text-slate-400">Monthly Average:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{(total1 / 12.0):.1f} mm</strong>
                </div>
                <div class="flex justify-between py-0.5">
                    <span class="text-slate-400">Average Temp:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{avg_temp1:.1f}°C</strong>
                </div>
            </div>
        </div>

        <!-- Year 2 Card -->
        <div class="p-4 rounded-xl bg-white dark:bg-slate-900/90 border border-amber-200/80 dark:border-amber-900/50 shadow-xs">
            <div class="flex items-center justify-between mb-2">
                <span class="font-semibold text-slate-900 dark:text-white text-sm sm:text-base flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                    <span>{year2} Summary</span>
                </span>
                <span class="text-[11px] px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">{p2_labels}</span>
            </div>
            <div class="space-y-1 text-xs text-slate-600 dark:text-slate-300">
                <div class="flex justify-between py-0.5 border-b border-slate-100 dark:border-slate-800/60">
                    <span class="text-slate-400">Total Rainfall:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{total2:.1f} mm</strong>
                </div>
                <div class="flex justify-between py-0.5 border-b border-slate-100 dark:border-slate-800/60">
                    <span class="text-slate-400">Monthly Average:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{(total2 / 12.0):.1f} mm</strong>
                </div>
                <div class="flex justify-between py-0.5">
                    <span class="text-slate-400">Average Temp:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{avg_temp2:.1f}°C</strong>
                </div>
            </div>
        </div>
    </div>

    <!-- Bottom Comparative Verdict -->
    <div class="p-3 rounded-xl bg-slate-100/80 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/60 flex items-center justify-between text-xs flex-wrap gap-2">
        <span class="font-medium text-slate-700 dark:text-slate-200">{verdict_badge}</span>
        <span class="text-[11px] text-slate-500 dark:text-slate-400">Difference: <strong>{diff:.1f} mm</strong></span>
    </div>
</div>
"""

# ==========================================
# HANDLER FUNCTIONS
# ==========================================
# ==========================================
# HEALTH, FIRST AID & EMERGENCY RESPONSE HANDLERS
# (Consistent with DOH & Philippine Red Cross guidelines)
# ==========================================

def handle_heatstroke_emergency():
    conversation_state["last_topic"] = "heatstroke_emergency"
    return L(
        # "🚨 **CRITICAL MEDICAL EMERGENCY: Heatstroke**\n\n"
        "**Heatstroke is life-threatening.** Do not treat this as simple tiredness or attempt home rest. Immediate emergency response is required.\n\n"
        "⚠️ **Warning Signs:**\n"
        "• **Confusion, altered mental state, or strange behavior**\n"
        "• **Seizures or muscle twitching**\n"
        "• **Loss of consciousness or fainting**\n"
        "• **Very hot skin** (Note: *The person may still be sweating* — sweating does NOT rule out heatstroke)\n"
        "• Rapid breathing and fast heart rate\n\n"
        "🚑 **Immediate Emergency Response:**\n"
        "1. **CALL FOR HELP IMMEDIATELY:**\n"
        "   • **National Emergency:** 911\n"
        "   • **Padre Garcia MDRRMO:** (043) 515-7424 | 0977-801-1557\n"
        "   • **Padre Garcia RHU:** (043) 515-9209\n"
        "2. **Move the person to a cool, shaded area** or indoors with air conditioning.\n"
        "3. **Remove excess clothing.**\n"
        "4. **Cool them rapidly:** Sponge the body with cool water, apply wet cloths to neck, armpits, and groin, and fan vigorously. Use safe cold-water immersion if available.\n"
        "5. ⛔ **DO NOT GIVE FLUIDS OR FOOD** if the person is confused, unconscious, having seizures, or unable to swallow safely (severe choking risk).\n"
        "6. Continue active cooling until professional medical responders arrive.\n\n"
        "⚠️ *Medical Disclaimer: ReadyKomunidad provides first-aid awareness and does not replace professional emergency medical care.*",

        "🚨 **KRITIKAL NA MEDICAL EMERGENCY: Heatstroke**\n\n"
        "**Nakalalason at banta sa buhay ang heatstroke.** Huwag itong ituring na karaniwang pagod o hayaang magpahinga lang sa bahay. Kailangan ng agarang medikal na tulong.\n\n"
        "⚠️ **Mga Babalang Senyales:**\n"
        "• **Pagkalito, pagka-lito sa sarili, o kakaibang pag-uugali**\n"
        "• **Kombulsyon (seizures) o panginginig**\n"
        "• **Pagkawala ng malay o pagkahimatay**\n"
        "• **Napakainit na balat** (Tandaan: *Maaaring nagpapawis pa rin ang pasyente* — ang pagpapawis ay HINDI patunay na walang heatstroke)\n"
        "• Mabilis na paghinga at mabilis na tibok ng puso\n\n"
        "🚑 **Mga Hakbang sa Agarang First Aid:**\n"
        "1. **TUMAWAG AGAD NG TULONG:**\n"
        "   • **National Emergency Hotline:** 911\n"
        "   • **Padre Garcia MDRRMO:** (043) 515-7424 | 0977-801-1557\n"
        "   • **Padre Garcia RHU:** (043) 515-9209\n"
        "2. **Ilipat agad ang tao sa malamig na lugar** o may lilim/aircon.\n"
        "3. **Tanggalin ang makakapal na damit.**\n"
        "4. **Palamigin agad ang katawan:** Punasan ng malamig na tubig, lagyan ng basang tela sa leeg, kili-kili, at singit, at paypayan nang mabilis.\n"
        "5. ⛔ **HUWAG MAGBIGAY NG TUBIG O PAGKAIN** kung ang tao ay nalilito, walang malay, nangingisay, o hindi kayang lumunok (panganib ng mabulunan).\n"
        "6. Ipagpatuloy ang pagpapalamig habang hinihintay ang mga rumespondeng doktor o ambulansya.\n\n"
        "⚠️ *Paalalang Medikal: Ang ReadyKomunidad ay nagbibigay ng first-aid awareness at hindi pamalit sa propesyonal na serbisyong medikal.*"
    )

def handle_heat_exhaustion_first_aid():
    conversation_state["last_topic"] = "heat_exhaustion_action"
    return L(
        "🌡️ **Heat Exhaustion: Warning Signs & First Aid**\n\n"
        "Heat exhaustion is a serious heat-related illness. If not managed quickly, it can rapidly escalate into life-threatening **heatstroke**.\n\n"
        "⚠️ **Possible Signs:**\n"
        "• Heavy, excessive sweating\n"
        "• Weakness, tiredness, and dizziness\n"
        "• Headache\n"
        "• Nausea or vomiting\n"
        "• Muscle cramps\n"
        "• Cool, pale, or clammy skin\n\n"
        "🩹 **Basic First Aid Steps:**\n"
        "1. **Move to shade:** Bring the person to a cool, shaded area or an air-conditioned room.\n"
        "2. **Rest & loosen clothes:** Have them lie down and loosen tight or unnecessary clothing.\n"
        "3. **Active cooling:** Apply cool, wet cloths to the skin, spray cool water, and use a fan.\n"
        "4. **Rehydrate carefully:** If the person is **fully awake and able to swallow**, let them slowly sip cool water or Oral Rehydration Solution (ORS).\n"
        "5. ⛔ **Never give liquids** if the person is confused, vomiting, or losing consciousness.\n\n"
        "🚨 **When to Seek Immediate Medical Help:**\n"
        "• Symptoms do not improve after **30 minutes**\n"
        "• The person starts vomiting repeatedly\n"
        "• The person becomes confused, develops seizures, or loses consciousness (this indicates **heatstroke** — call 911 immediately)\n\n"
        "⚠️ *Medical Disclaimer: ReadyKomunidad first-aid steps do not replace professional medical diagnosis and care.*",

        "🌡️ **Heat Exhaustion: Mga Senyales at First Aid**\n\n"
        "Ang heat exhaustion ay isang seryosong kondisyon dulot ng matinding init. Kapag napabayaan, maaari itong mabilis na lumala at maging nakamamatay na **heatstroke**.\n\n"
        "⚠️ **Mga Posibleng Senyales:**\n"
        "• Malakas at sobrang pagpapawis\n"
        "• Panghihina, labis na pagod, at pagkahilo\n"
        "• Pananakit ng ulo\n"
        "• Pagduduwal o pagsusuka\n"
        "• Pamumulikat ng kalamnan (muscle cramps)\n"
        "• Malamig, maputla, o malagkit na balat\n\n"
        "🩹 **Mga Hakbang sa First Aid:**\n"
        "1. **Pumunta sa lilim:** Ilipat ang tao sa malamig na lugar, may lilim, o may electric fan.\n"
        "2. **Pahinga at luwagan ang damit:** Pahigain at luwagan ang mahigpit o makapal na kasuotan.\n"
        "3. **Palamigin ang katawan:** Maglagay ng basang tela sa noo, leeg, at kilikili, at paypayan.\n"
        "4. **Painumin nang dahan-dahan:** Kung ang tao ay **gising na gising at kayang lumunok**, painumin ng pakonti-konting malamig na tubig o Oral Rehydration Solution (ORS).\n"
        "5. ⛔ **Huwag piliting painumin** kung ang tao ay nalilito, nagsusuka, o nawawalan ng malay.\n\n"
        "🚨 **Kailan Dapat Magpatingin sa Doktor:**\n"
        "• Hindi bumubuti ang pakiramdam pagkatapos ng **30 minuto**\n"
        "• Patuloy ang pagsusuka\n"
        "• Nagsimulang maguluhan sa pag-iisip o nawalan ng malay (senyales na ito ng **heatstroke** — tumawag agad sa 911 o RHU)\n\n"
        "⚠️ *Paalalang Medikal: Ang gabay na ito ay first-aid awareness lamang at hindi pamalit sa pagsusuri ng doktor.*"
    )

def handle_dehydration_first_aid():
    conversation_state["last_topic"] = "dehydration_first_aid"
    return L(
        "💧 **Dehydration: Warning Signs & First Aid**\n\n"
        "Dehydration occurs when the body loses more fluids than it takes in, commonly triggered by extreme heat, heavy sweating, vomiting, or diarrhea during El Niño months.\n\n"
        "⚠️ **Possible Signs:**\n"
        "• Extreme thirst and dry mouth\n"
        "• Dark yellow, strong-smelling urine\n"
        "• Reduced urination (fewer trips to the bathroom)\n"
        "• Fatigue, dry skin, and dizziness upon standing\n\n"
        "🩹 **Basic First Aid Steps:**\n"
        "1. **Rest away from heat:** Move the person out of direct sunlight and let them rest in a cool area.\n"
        "2. **Drink in small sips:** Give small, frequent sips of clean drinking water.\n"
        "3. **Use ORS (Oral Rehydration Salts):** If fluid was lost through heavy sweating, vomiting, or diarrhea, give properly prepared ORS to replace vital electrolytes.\n"
        "4. **Monitor recovery:** Check urine color and frequency (pale or clear urine means rehydration is succeeding).\n"
        "5. ⛔ **Never give drinks** to anyone who is unconscious, drowsy, or having difficulty swallowing.\n\n"
        "🚨 **When to Seek Immediate Medical Help:**\n"
        "• Little or no urine for more than 8 hours\n"
        "• Inability to keep fluids down due to continuous vomiting\n"
        "• Lethargy, confusion, fainting, or rapid pulse\n"
        "• Children under 5 or elderly individuals showing sunken eyes or extreme weakness\n\n"
        "⚠️ *Medical Disclaimer: ReadyKomunidad guidance is for informational first aid and does not substitute for clinical medical evaluation.*",

        "💧 **Dehydration: Mga Senyales at First Aid**\n\n"
        "Ang dehydration ay kakulangan ng tubig sa katawan. Madalas itong mangyari tuwing tagtuyot o El Niño dahil sa sobrang init, labis na pagpapawis, pagtatae, o pagsusuka.\n\n"
        "⚠️ **Mga Posibleng Senyales:**\n"
        "• Matinding pagkauhaw at tuyong bibig\n"
        "• Matingkad na dilaw na ihi na may matapang na amoy\n"
        "• Madalang na pag-ihi (o hindi umiihi nang maraming oras)\n"
        "• Panghihina, tuyong balat, at pagkahilo kapag tumatayo\n\n"
        "🩹 **Mga Hakbang sa First Aid:**\n"
        "1. **Lumayo sa init:** Dalhin ang tao sa malilim at malamig na lugar.\n"
        "2. **Painumin nang paunti-unti:** Magbigay ng pakonti-konti ngunit madalas na higop ng malinis na tubig.\n"
        "3. **Gumamit ng ORS (Oral Rehydration Solution):** Kung marami ang pawis o nagtae, magtimpla ng ORS upang mapalitan ang nawalang electrolytes at asin sa katawan.\n"
        "4. **Bantayan ang paggaling:** Tignan ang kulay ng ihi (maliwanag na dilaw o malinaw na ihi ay patunay na nagiging maayos na ang hydration).\n"
        "5. ⛔ **Huwag magpainom** kung ang tao ay tuliro, walang malay, o hindi makalunok nang maayos.\n\n"
        "🚨 **Kailan Dapat Dalhin sa Ospital o RHU:**\n"
        "• Halos walang ihi sa loob ng 8 oras\n"
        "• Hindi makainom dahil patuloy na nagsusuka\n"
        "• Pagkahimatay, pagkalito, o mabilis na tibok ng puso\n"
        "• Mga bata o matatanda na lumulubog ang mata at labis na nanghihina\n\n"
        "⚠️ *Paalalang Medikal: Ang first-aid na ito ay gabay lamang at hindi pamalit sa lunas ng lisensyadong manggagamot.*"
    )

def handle_stations_overview():
    conversation_state["last_topic"] = "stations_overview"
    return L(
        "📍 **The 3 Monitored Weather Stations:**\n"
        "1. **Padre Garcia, Batangas** — Primary local station for community and agricultural climate tracking.\n"
        "2. **Quezon / Bicol Region** — Regional comparison station providing broader Calabarzon/Bicol patterns.\n"
        "3. **Basco, Batanes** — Northernmost reference station capturing northern monsoon and typhoon dynamics.",
        "📍 **Ang 3 Monitored na Weather Stations:**\n"
        "1. **Padre Garcia, Batangas** — Primary local station para sa climate tracking ng community at agrikultura.\n"
        "2. **Quezon / Bicol Region** — Regional comparison station na nagbibigay ng mas malawak na Calabarzon/Bicol patterns.\n"
        "3. **Basco, Batanes** — Northernmost reference station na sumusukat sa northern monsoon at typhoon dynamics."
    )

def handle_el_nino_general():
    if master_df is None: return "Dataset not loaded."
    el_nino_data = master_df[master_df['ens_phase'].astype(str).str.contains('E', na=False)]
    normal_data = master_df[master_df['ens_phase'] == 'N']

    avg_rain_s1_en = el_nino_data['rain_batangas'].mean()
    avg_rain_s1_norm = normal_data['rain_batangas'].mean()
    avg_temp_s1_en = el_nino_data['temp_batangas'].mean()

    avg_rain_s2_en = el_nino_data['rain_station2'].mean()
    avg_rain_s2_norm = normal_data['rain_station2'].mean()
    avg_temp_s2_en = el_nino_data['temp_station2'].mean()

    r3_col = 'rain_basco' if 'rain_basco' in master_df.columns else 'rain_station3'
    t3_col = 'temp_basco' if 'temp_basco' in master_df.columns else 'temp_station3'
    avg_rain_s3_en = el_nino_data[r3_col].mean()
    avg_rain_s3_norm = normal_data[r3_col].mean()
    avg_temp_s3_en = el_nino_data[t3_col].mean()

    conversation_state["last_topic"] = "el_nino"

    if conversation_state.get("lang") == "tl":
        return (
            f"📊 **Epekto ng El Niño sa 3 Stations:**\n"
            f"Kapag dumating ang El Niño, nagbabago ang conditions sa lahat ng monitored na stations:\n\n"
            f"• **Padre Garcia, Batangas:** Bumababa ang rainfall sa **{avg_rain_s1_en:.1f} mm** (Normal: {avg_rain_s1_norm:.1f} mm) | Temp: **{avg_temp_s1_en:.1f}°C**\n"
            f"• **Quezon / Bicol Region:** Bumababa ang rainfall sa **{avg_rain_s2_en:.1f} mm** (Normal: {avg_rain_s2_norm:.1f} mm) | Temp: **{avg_temp_s2_en:.1f}°C**\n"
            f"• **Basco, Batanes:** Average na rainfall **{avg_rain_s3_en:.1f} mm** (Normal: {avg_rain_s3_norm:.1f} mm) | Temp: **{avg_temp_s3_en:.1f}°C**"
        )
    return (
        f"📊 **3-Station El Niño Impact:**\n"
        f"When El Niño strikes, conditions shift across all monitored stations:\n\n"
        f"• **Padre Garcia, Batangas:** Rain drops to **{avg_rain_s1_en:.1f} mm** (Normal: {avg_rain_s1_norm:.1f} mm) | Temp: **{avg_temp_s1_en:.1f}°C**\n"
        f"• **Quezon / Bicol Region:** Rain drops to **{avg_rain_s2_en:.1f} mm** (Normal: {avg_rain_s2_norm:.1f} mm) | Temp: **{avg_temp_s2_en:.1f}°C**\n"
        f"• **Basco, Batanes:** Rain averages **{avg_rain_s3_en:.1f} mm** (Normal: {avg_rain_s3_norm:.1f} mm) | Temp: **{avg_temp_s3_en:.1f}°C**"
    )

def handle_el_nino_padre_garcia():
    if master_df is None: return "Dataset not loaded."
    el_nino_data = master_df[master_df['ens_phase'].astype(str).str.contains('E', na=False)]
    normal_data = master_df[master_df['ens_phase'] == 'N']

    avg_rain_en = el_nino_data['rain_batangas'].mean()
    avg_rain_norm = normal_data['rain_batangas'].mean()
    avg_temp_en = el_nino_data['temp_batangas'].mean()
    avg_temp_norm = normal_data['temp_batangas'].mean()
    rain_diff = avg_rain_norm - avg_rain_en

    conversation_state["last_topic"] = "el_nino"

    if conversation_state.get("lang") == "tl":
        return (
            f"☀️ **El Niño sa Padre Garcia, Batangas:**\n"
            f"• **Bumababa ang rainfall sa {avg_rain_en:.1f} mm/month** (Normal: {avg_rain_norm:.1f} mm) — **{rain_diff:.1f} mm mas kaunti** kaysa sa usual\n"
            f"• **Tumataas ang average temperature sa {avg_temp_en:.1f}°C** (Normal: {avg_temp_norm:.1f}°C)\n\n"
            f"⚠️ Mahahabang dry spells, nababawasang water supply, at mas mataas na drought risk para sa mga local farmers at residents."
        )
    return (
        f"☀️ **El Niño in Padre Garcia, Batangas:**\n"
        f"• **Rainfall drops to {avg_rain_en:.1f} mm/month** (Normal: {avg_rain_norm:.1f} mm) — **{rain_diff:.1f} mm less** than usual\n"
        f"• **Average temperature rises to {avg_temp_en:.1f}°C** (Normal: {avg_temp_norm:.1f}°C)\n\n"
        f"⚠️ Prolonged dry spells, reduced water supply, and higher drought risk for local farmers and residents."
    )

def handle_la_nina_general():
    if master_df is None: return "Dataset not loaded."
    la_nina_data = master_df[master_df['ens_phase'].astype(str).str.contains('L', na=False)]
    el_nino_data = master_df[master_df['ens_phase'].astype(str).str.contains('E', na=False)]
    normal_data = master_df[master_df['ens_phase'] == 'N']

    avg_rain_s1_ln = la_nina_data['rain_batangas'].mean()
    avg_rain_s1_en = el_nino_data['rain_batangas'].mean()
    avg_rain_s1_norm = normal_data['rain_batangas'].mean()
    avg_temp_s1_ln = la_nina_data['temp_batangas'].mean()

    avg_rain_s2_ln = la_nina_data['rain_station2'].mean()
    avg_rain_s2_norm = normal_data['rain_station2'].mean()
    avg_temp_s2_ln = la_nina_data['temp_station2'].mean()

    r3_col = 'rain_basco' if 'rain_basco' in master_df.columns else 'rain_station3'
    t3_col = 'temp_basco' if 'temp_basco' in master_df.columns else 'temp_station3'
    avg_rain_s3_ln = la_nina_data[r3_col].mean()
    avg_rain_s3_norm = normal_data[r3_col].mean()
    avg_temp_s3_ln = la_nina_data[t3_col].mean()

    conversation_state["last_topic"] = "la_nina"

    if conversation_state.get("lang") == "tl":
        return (
            f"🌧️ **Ano ang La Niña?**\n"
            f"Ang La Niña ay isang natural na climate phenomenon kung saan ang sea surface temperature ng Pacific Ocean ay **mas malamig kaysa sa normal**, "
            f"na nagdudulot ng kabaligtaran ng epekto ng El Niño. Imbes na drought, nagdudulot ito ng **above-normal rainfall**, mas malakas na monsoons, "
            f"at mas malaking flood at landslide risk sa buong Pilipinas.\n\n"
            f"📊 **La Niña Data sa 3 Stations:**\n\n"
            f"• **Padre Garcia, Batangas:** Tumataas ang rainfall sa **{avg_rain_s1_ln:.1f} mm/month** (Normal: {avg_rain_s1_norm:.1f} mm | El Niño: {avg_rain_s1_en:.1f} mm) | Temp: **{avg_temp_s1_ln:.1f}°C**\n"
            f"• **Quezon / Bicol Region:** Tumataas ang rainfall sa **{avg_rain_s2_ln:.1f} mm/month** (Normal: {avg_rain_s2_norm:.1f} mm) | Temp: **{avg_temp_s2_ln:.1f}°C**\n"
            f"• **Basco, Batanes:** Tumataas ang rainfall sa **{avg_rain_s3_ln:.1f} mm/month** (Normal: {avg_rain_s3_norm:.1f} mm) | Temp: **{avg_temp_s3_ln:.1f}°C**\n\n"
            f"⚠️ **Local Impacts Tuwing La Niña:**\n"
            f"• Flooding sa mga low-lying areas at bukid\n"
            f"• Landslide risk sa mga elevated na communities\n"
            f"• Mas malakas at mas madalas na typhoons\n"
            f"• Soil erosion at crop damage dahil sa sobrang rainfall"
        )
    return (
        f"🌧️ **What is La Niña?**\n"
        f"La Niña is a natural climate phenomenon where sea surface temperatures in the Pacific Ocean are **cooler than normal**, "
        f"causing the opposite effects of El Niño. Instead of drought, it brings **above-normal rainfall**, stronger monsoons, "
        f"and increased risk of flooding and landslides across the Philippines.\n\n"
        f"📊 **3-Station La Niña Data:**\n\n"
        f"• **Padre Garcia, Batangas:** Rain surges to **{avg_rain_s1_ln:.1f} mm/mo** (Normal: {avg_rain_s1_norm:.1f} mm | El Niño: {avg_rain_s1_en:.1f} mm) | Temp: **{avg_temp_s1_ln:.1f}°C**\n"
        f"• **Quezon / Bicol Region:** Rain surges to **{avg_rain_s2_ln:.1f} mm/mo** (Normal: {avg_rain_s2_norm:.1f} mm) | Temp: **{avg_temp_s2_ln:.1f}°C**\n"
        f"• **Basco, Batanes:** Rain surges to **{avg_rain_s3_ln:.1f} mm/mo** (Normal: {avg_rain_s3_norm:.1f} mm) | Temp: **{avg_temp_s3_ln:.1f}°C**\n\n"
        f"⚠️ **Local Impacts during La Niña:**\n"
        f"• Flooding of low-lying areas and farm fields\n"
        f"• Landslide risk in elevated communities\n"
        f"• Stronger and more frequent typhoons\n"
        f"• Soil erosion and crop damage from excessive rain"
    )

def handle_ml_forecast_query(month_num, target_year=None):
    if not rain_model:
        return "⚠️ Machine learning model file (`climate_rain_model.pkl`) is not loaded yet. Run `train_model.py` first!"

    month_data = master_df[master_df['Month'] == month_num]
    if month_data.empty:
        return f"Could not find baseline data for month {month_num}."

    year_str = f" for {target_year}" if target_year else ""
    avg_temp = month_data['temp_batangas'].mean()
    avg_rain_s2 = month_data['rain_station2'].mean()
    avg_oni = month_data['oni_value'].mean()
    predicted_rain = rain_model.predict([[month_num, avg_temp, avg_rain_s2, avg_oni]])[0]
    m_name = MONTH_NAMES[month_num]

    conversation_state["last_topic"] = "prediction"

    future_note = ""
    if target_year and int(target_year) > int(master_df['Year'].max()):
        if conversation_state.get("lang") == "tl":
            future_note = f"\n📅 *{target_year} ay isang future year — ito ay ML forecast, hindi historical data.*"
        else:
            future_note = f"\n📅 *{target_year} is a future year — this is an ML forecast, not historical data.*"

    if conversation_state.get("lang") == "tl":
        return (
            f"🤖 **AI Rainfall Prediction{year_str}:**\n"
            f"Batay sa mga historical patterns para sa **{m_name}**:\n"
            f"• **Predicted Rainfall para sa Padre Garcia:** **{predicted_rain:.1f} mm**\n"
            f"*(Gumagamit ng regional station data at ONI indicators bilang inputs)*"
            f"{future_note}"
        )
    return (
        f"🤖 **AI Rainfall Prediction{year_str}:**\n\n"
        f"Based on historical patterns for **{m_name}**:\n\n"
        f"• **Predicted Rainfall for Padre Garcia:** **{predicted_rain:.1f} mm**\n"
        f"*(Uses regional station data and ONI indicators as inputs)*"
        f"{future_note}"
    )


    #EMOJI MIRRORING FUNCTIONS
import unicodedata

def extract_user_emojis(text):
    """Finds all emojis used in the user's input."""
    emojis = []
    for char in text:
        # Check if character is an emoji or symbol
        category = unicodedata.category(char)
        if category in ('So', 'Sk') or ord(char) > 0x1F300:
            if char not in emojis:
                emojis.append(char)
    return emojis

def apply_emoji_mirroring(reply, user_message, is_tl=False):
    """
    If the user used emojis, mirror them back and optionally add
    an empathetic emotional touch to match their mood.
    """
    emojis = extract_user_emojis(user_message)
    if not emojis:
        return reply

    emoji_str = " ".join(emojis[:3])  # up to 3 emojis to keep it clean

    # Detect emotional context based on common emojis
    worried_emojis = {'😢', '😭', '😰', '😨', '😱', '😥', '🥺', '😔', '😟'}
    heat_emojis = {'🥵', '🔥', '☀️', '🌡️'}
    happy_emojis = {'😊', '😄', '😀', '❤️', '🙌', '👍', '🙏', '✨'}

    opener = ""
    # 1. Worried / distressed
    if any(e in worried_emojis for e in emojis):
        opener = (
            f"Naiintindihan ko ang iyong pag-aalala {emoji_str}. " 
            if is_tl else 
            f"I hear your concern {emoji_str}. "
        )
    # 2. Feeling the heat / exhausted
    elif any(e in heat_emojis for e in emojis):
        opener = (
            f"Grabe, ramdam na ramdam ang tindi ng init {emoji_str}! " 
            if is_tl else 
            f"I feel you with that heat {emoji_str}! "
        )
    # 3. Friendly / grateful / positive
    elif any(e in happy_emojis for e in emojis):
        opener = (
            f"Maraming salamat din {emoji_str}! " 
            if is_tl else 
            f"Glad to help {emoji_str}! "
        )
    # 4. Any other emoji: simply mirror it warmly at the beginning
    else:
        opener = f"{emoji_str} "

    # If the response is a visual card (e.g. rainfall card), place opener smoothly
    if "<!-- RAINFALL_CARD -->" in reply:
        return f"<p class='text-sm font-medium text-slate-700 dark:text-slate-200 mb-2'>{opener}</p>" + reply

    return f"{opener}\n\n{reply}"

# ==========================================
# MODERN RAINFALL CARD & CHART GENERATOR
# ==========================================

def render_rainfall_chart_card(year_num, year_data, is_tl=False, enso_phase_label="Neutral 🟡", avg_temp=None):
    """
    Renders an interactive, modern visual card matching the reference design:
    - Title with rain emoji + Year
    - Tag badge: 'Historical Weather Records · Padre Garcia, Batangas'
    - Responsive SVG Bar Chart with 12 monthly bars & hover tooltips
    - 2x2 metric cards (Annual total, Monthly average, Wettest month, Driest month)
    - Simple community observation & El Niño preparedness section
    """
    months_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_vals = []

    for m in range(1, 13):
        m_row = year_data[year_data['Month'] == m]
        if not m_row.empty and pd.notna(m_row['rain_batangas'].values[0]):
            val = float(m_row['rain_batangas'].values[0])
        else:
            val = 0.0
        monthly_vals.append(val)

    annual_total = sum(monthly_vals)
    monthly_avg = annual_total / 12.0

    # Wettest & driest months
    s_vals = pd.Series(monthly_vals)
    max_idx = int(s_vals.idxmax())
    max_val = monthly_vals[max_idx]
    wettest_name = MONTH_NAMES.get(max_idx + 1, months_abbr[max_idx])

    non_zero = s_vals[s_vals > 0]
    min_idx = int(non_zero.idxmin()) if not non_zero.empty else int(s_vals.idxmin())
    min_val = monthly_vals[min_idx]
    driest_name = MONTH_NAMES.get(min_idx + 1, months_abbr[min_idx])

    # Y-axis bounds
    peak = max(max_val, 10.0)
    top_limit = peak * 1.15
    if top_limit <= 100:
        step = 25
    elif top_limit <= 200:
        step = 45
    elif top_limit <= 400:
        step = 80
    else:
        step = int(top_limit / 4 / 25) * 25 or 50

    top_y = step * 4
    if top_y < peak:
        top_y = step * 5

    y_levels = [top_y, step * 3, step * 2, step * 1, 0]

    # SVG layout
    chart_height = 200
    chart_width = 540
    margin_left = 65
    margin_right = 20
    plot_width = chart_width - margin_left - margin_right
    bar_width = 24
    slot_width = plot_width / 12

    bars_svg = []
    labels_svg = []

    for i, (val, abbr) in enumerate(zip(monthly_vals, months_abbr)):
        x_slot = margin_left + (i * slot_width)
        bar_x = x_slot + (slot_width - bar_width) / 2
        bar_h = (val / top_y) * (chart_height - 40) if top_y > 0 else 0
        bar_y = (chart_height - 30) - bar_h

        tooltip_x = bar_x + (bar_width / 2)
        tooltip_y = max(bar_y - 28, 6)

        bars_svg.append(f"""
            <g class="bar-group group cursor-pointer">
                <rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width}" height="{max(bar_h, 3):.1f}" rx="6"
                      class="fill-blue-500 hover:fill-blue-600 dark:fill-blue-400 dark:hover:fill-blue-300 transition-all duration-150" />
                <g class="bar-tooltip opacity-0 transition-all duration-150 pointer-events-none">
                    <rect x="{tooltip_x - 30:.1f}" y="{tooltip_y:.1f}" width="60" height="22" rx="4"
                          class="fill-slate-900 dark:fill-slate-100 shadow-md" />
                    <text x="{tooltip_x:.1f}" y="{tooltip_y + 14:.1f}" text-anchor="middle"
                          class="fill-white dark:fill-slate-900 text-[10px] font-semibold">{val:.1f} mm</text>
                </g>
            </g>
        """)

        labels_svg.append(f"""
            <text x="{bar_x + (bar_width / 2):.1f}" y="{chart_height - 10}" text-anchor="middle"
                  class="fill-slate-500 dark:fill-slate-400 text-[11px] font-medium">{abbr}</text>
        """)

    grid_svg = []
    for lvl in y_levels:
        y_pos = 12 + (chart_height - 42) * (1.0 - (lvl / top_y)) if top_y > 0 else 12
        grid_svg.append(f"""
            <line x1="{margin_left}" y1="{y_pos:.1f}" x2="{chart_width - margin_right}" y2="{y_pos:.1f}"
                  stroke="currentColor" stroke-dasharray="3,3" class="stroke-slate-200 dark:stroke-slate-800" stroke-width="1"/>
            <text x="{margin_left - 8}" y="{y_pos + 3.5:.1f}" text-anchor="end"
                  class="fill-slate-400 dark:fill-slate-500 text-[10px]">{int(lvl)} mm</text>
        """)

    badge_text = "Historical Weather Records · Padre Garcia, Batangas"
    intro_text = (
        f"Here is an illustrative monthly rainfall record for <strong>{year_num}</strong> in Padre Garcia, Batangas."
        if not is_tl else
        f"Narito ang buwanang ulat ng ulan para sa taong <strong>{year_num}</strong> sa Padre Garcia, Batangas."
    )
    chart_title = f"Monthly Rainfall in {year_num}" if not is_tl else f"Buwanang Pag-ulan noong {year_num}"
    chart_subtitle = "Rainfall measurements in millimeters (mm)." if not is_tl else "Sukat ng ulan sa millimeters (mm)."
    summary_title = "Annual rainfall summary" if not is_tl else "Taunang Buod ng Pag-ulan"

    lbl_annual_total = "Annual total" if not is_tl else "Kabuuang Ulan"
    lbl_monthly_avg = "Monthly average" if not is_tl else "Buwanang Karaniwan"
    lbl_wettest = "Wettest month" if not is_tl else "Pinakabasang Buwan"
    lbl_driest = "Driest month" if not is_tl else "Pinakatuyong Buwan"

    # Context tag
    enso_block = ""
    if enso_phase_label:
        lbl_climate = "Climate Context:" if not is_tl else "Konteksto ng Klima:"
        lbl_temp = f" • Avg Temp: <strong>{avg_temp:.1f}°C</strong>" if avg_temp else ""
        enso_block = f"""
        <div class="mt-2.5 pt-2 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
            <span>{lbl_climate} <strong>{enso_phase_label}</strong>{lbl_temp}</span>
            <span class="text-[11px] text-slate-400">Padre Garcia Station</span>
        </div>
        """


    if not is_tl:
        observation_block = f"""
        <!-- Simple Community Observation & Preparedness -->
        <div class="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800 space-y-3 text-xs leading-relaxed">
            <div>
                <h6 class="font-semibold text-slate-900 dark:text-white flex items-center gap-1.5 text-xs sm:text-sm mb-1">
                    <span>🔍</span><span>What can we observe?</span>
                </h6>
                <p class="text-slate-600 dark:text-slate-300">
                    Based on the historical records for {year_num}, <strong>{driest_name}</strong> had the lowest rainfall ({min_val:.0f} mm), while <strong>{wettest_name}</strong> recorded the highest ({max_val:.0f} mm). Rainfall varied throughout the year, with drier conditions in the earlier months and heavier rainfall during the middle and latter parts of the year.
                </p>
            </div>

            <div>
                <h6 class="font-semibold text-slate-900 dark:text-white flex items-center gap-1.5 text-xs sm:text-sm mb-1">
                    <span>☀️</span><span>What does this mean for El Niño preparedness?</span>
                </h6>
                <p class="text-slate-600 dark:text-slate-300">
                    Months with lower rainfall remind us of the importance of conserving water and preparing for possible dry conditions. However, these records alone describe historical rainfall patterns and do not confirm that El Niño occurred or caused the changes in rainfall.
                </p>
            </div>

            <div class="pt-2 border-t border-slate-200/70 dark:border-slate-800">
                <span class="font-semibold text-slate-800 dark:text-slate-200 block mb-1">💡 Stay prepared in Padre Garcia:</span>
                <ul class="space-y-0.5 text-slate-600 dark:text-slate-300 text-[11px]">
                    <li>• Save and store clean water whenever possible.</li>
                    <li>• Avoid unnecessary water consumption and fix leaking pipes.</li>
                    <li>• Follow official PAGASA weather updates and local MDRRMO advisories.</li>
                </ul>
            </div>
        </div>
        """
    else:
        observation_block = f"""
        <!-- Simple Community Observation & Preparedness -->
        <div class="p-4 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800 space-y-3 text-xs leading-relaxed">
            <div>
                <h6 class="font-semibold text-slate-900 dark:text-white flex items-center gap-1.5 text-xs sm:text-sm mb-1">
                    <span>🔍</span><span>Ano ang mapapansin natin?</span>
                </h6>
                <p class="text-slate-600 dark:text-slate-300">
                    Batay sa mga tala noong {year_num}, ang buwan ng <strong>{driest_name}</strong> ang may pinakamababang ulan ({min_val:.0f} mm), habang ang <strong>{wettest_name}</strong> naman ang nagtala ng pinakamataas ({max_val:.0f} mm). Nag-iiba-iba ang ulan sa buong taon, kung saan mas tuyo ang mga unang buwan at mas marami ang ulan bandang kalagitnaan hanggang huling bahagi ng taon.
                </p>
            </div>

            <div>
                <h6 class="font-semibold text-slate-900 dark:text-white flex items-center gap-1.5 text-xs sm:text-sm mb-1">
                    <span>☀️</span><span>Ano ang ibig sabihin nito para sa paghahanda sa El Niño?</span>
                </h6>
                <p class="text-slate-600 dark:text-slate-300">
                    Ang mga buwan na may mababang ulan ay nagpapaalala sa atin ng kahalagahan ng pagtitipid ng tubig at paghahanda para sa posibleng tagtuyot. Gayunpaman, ang mga rekord na ito ay naglalarawan lamang ng ulan at hindi awtomatikong nagpapatunay na El Niño ang sanhi.
                </p>
            </div>

            <div class="pt-2 border-t border-slate-200/70 dark:border-slate-800">
                <span class="font-semibold text-slate-800 dark:text-slate-200 block mb-1">💡 Maging laging handa sa Padre Garcia:</span>
                <ul class="space-y-0.5 text-slate-600 dark:text-slate-300 text-[11px]">
                    <li>• Mag-ipon at magtipid ng malinis na tubig sa ligtas at may takip na lalagyan.</li>
                    <li>• Iwasan ang pag-aaksaya ng tubig at ayusin ang tumutulong gripo.</li>
                    <li>• Subaybayan ang mga opisyal na weather advisory ng PAGASA at lokal na MDRRMO.</li>
                </ul>
            </div>
        </div>
        """

    return f"""<!-- RAINFALL_CARD -->
<div class="w-full my-2 space-y-4 animate-fade-in font-sans">
    <!-- Header -->
    <div>
        <div class="flex items-center gap-2 mb-1.5 flex-wrap">
            <span class="text-base">🌧️</span>
            <span class="font-semibold text-slate-900 dark:text-white text-base">Historical Rainfall – {year_num}</span>
            <span class="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300 border border-amber-300/40 dark:border-amber-700/50">
                {badge_text}
            </span>
        </div>
        <p class="text-xs sm:text-sm text-slate-600 dark:text-slate-300">{intro_text}</p>
    </div>

    <!-- Chart Container Card -->
    <div class="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 shadow-xs">
        <div class="flex items-start justify-between mb-4">
            <div>
                <h4 class="font-semibold text-slate-900 dark:text-white text-sm sm:text-base">{chart_title}</h4>
                <p class="text-[11px] sm:text-xs text-slate-400 dark:text-slate-500 mt-0.5">{chart_subtitle}</p>
            </div>
            <div class="flex items-center gap-1.5 text-slate-400">
                <span class="p-1 hover:text-slate-600 dark:hover:text-slate-200" title="Rainfall Graph">
                    <i class="fa-solid fa-chart-column text-xs"></i>
                </span>
            </div>
        </div>

        <!-- Responsive SVG Bar Chart -->
        <div class="w-full overflow-x-auto">
            <svg viewBox="0 0 {chart_width} {chart_height}" class="w-full h-48 sm:h-56 select-none">
                {''.join(grid_svg)}
                {''.join(bars_svg)}
                {''.join(labels_svg)}
            </svg>
        </div>
    </div>

    <!-- 2x2 Metric Summary Grid -->
    <div>
        <h5 class="text-xs sm:text-sm font-semibold text-slate-800 dark:text-slate-200 mb-2.5">{summary_title}</h5>
        <div class="grid grid-cols-2 gap-2.5">
            <div class="p-3.5 rounded-xl bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 shadow-xs">
                <span class="text-[11px] text-slate-400 dark:text-slate-500 block mb-1 font-medium">{lbl_annual_total}</span>
                <span class="text-base sm:text-lg font-bold text-slate-900 dark:text-white">{annual_total:.0f} mm</span>
            </div>
            <div class="p-3.5 rounded-xl bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 shadow-xs">
                <span class="text-[11px] text-slate-400 dark:text-slate-500 block mb-1 font-medium">{lbl_monthly_avg}</span>
                <span class="text-base sm:text-lg font-bold text-slate-900 dark:text-white">{monthly_avg:.1f} mm</span>
            </div>
            <div class="p-3.5 rounded-xl bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 shadow-xs">
                <span class="text-[11px] text-slate-400 dark:text-slate-500 block mb-1 font-medium">{lbl_wettest}</span>
                <span class="text-xs sm:text-sm font-semibold text-slate-900 dark:text-white block">{wettest_name}</span>
                <span class="text-xs font-medium text-blue-600 dark:text-blue-400">{max_val:.0f} mm</span>
            </div>
            <div class="p-3.5 rounded-xl bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 shadow-xs">
                <span class="text-[11px] text-slate-400 dark:text-slate-500 block mb-1 font-medium">{lbl_driest}</span>
                <span class="text-xs sm:text-sm font-semibold text-slate-900 dark:text-white block">{driest_name}</span>
                <span class="text-xs font-medium text-amber-600 dark:text-amber-400">{min_val:.0f} mm</span>
            </div>
        </div>
        {enso_block}
    </div>

    <!-- Community Observation & Preparedness Box -->
    {observation_block}
</div>
"""

# ==========================================
# FULL HANDLE_YEAR_QUERY IMPLEMENTATION
# ==========================================

def handle_year_query(year, padre_garcia_only=False, is_rain_specific=False):
    """
    Handles year-level climate and rainfall inquiries:
    - If year exists in master dataset (2010–2026), renders the sleek rainfall chart card & metric summary.
    - If year is in the future, generates 12-month machine learning predictions.
    - If year is outside supported bounds, gracefully guides the user.
    """
    if master_df is None:
        return "⚠️ Dataset not loaded. Please verify chatbot_master_climate_data.csv exists."

    try:
        target_year = int(year)
    except (ValueError, TypeError):
        return L(f"Invalid year: {year}", f"Hindi wastong taon: {year}")

    year_data = master_df[master_df['Year'] == target_year]

    # Handle years not found in the historical dataset
    if year_data.empty:
        max_year = int(master_df['Year'].max())
        min_year = int(master_df['Year'].min())

        if target_year > max_year:
            # Future year: generate ML-based forecast for all 12 months
            if rain_model:
                monthly_preds = []
                total_pred = 0.0
                for m in range(1, 13):
                    month_data = master_df[master_df['Month'] == m]
                    if month_data.empty:
                        continue
                    avg_temp = month_data['temp_batangas'].mean()
                    avg_rain_s2 = month_data['rain_station2'].mean()
                    avg_oni = month_data['oni_value'].mean()
                    pred = rain_model.predict([[m, avg_temp, avg_rain_s2, avg_oni]])[0]
                    total_pred += pred
                    monthly_preds.append(f"• **{MONTH_NAMES[m]}:** {pred:.1f} mm")

                conversation_state["last_topic"] = "prediction"
                conversation_state["last_year"] = target_year

                if conversation_state.get("lang") == "tl":
                    return (
                        f"🔮 **AI Rainfall Forecast para sa {target_year}:**\n"
                        f"*{target_year} ay isang future year — wala pang historical records. "
                        f"Narito ang ML-based na monthly rainfall predictions para sa Padre Garcia, Batangas:*\n\n"
                        + "\n".join(monthly_preds) +
                        f"\n\n**Total Predicted Rainfall:** ~{total_pred:.1f} mm\n\n"
                        f"**Disclaimer**: *Ang mga ito ay machine-learning-based estimates na ginawa gamit ang historical weather data. Maaaring mag-iba ang aktwal na ulan mula sa predicted values dahil sa pagbabago ng panahon at klima. Ang mga estimate na ito ay hindi opisyal na PAGASA forecasts.*\n\n"
                        f"\n\n💡 *Para sa isang partikular na buwan, subukan: 'i-predict ang ulan para sa May {target_year}'*"
                    )
                return (
                    f"🔮 **AI Rainfall Forecast for {target_year}:**\n"
                    f"*{target_year} is a future year — no historical records exist yet. "
                    f"Here are ML-based monthly rainfall predictions for Padre Garcia, Batangas:*\n\n"
                    + "\n".join(monthly_preds) +
                    f"\n\n **Total Predicted Rainfall:** ~{total_pred:.1f} mm\n\n"
                    f"**Disclaimer**: *These are machine-learning-based estimates generated using historical weather data. Actual rainfall may differ from the predicted values due to changing weather and climate conditions. These estimates are not official PAGASA forecasts.*\n\n"

                    f"\n\n💡 *For a specific month, try: 'Predict rain for May {target_year}'*"
                )

            return L(
                f"I don't have climate records for **{target_year}**. I cover {min_year} through {max_year}. The ML prediction model is also unavailable.",
                f"Wala akong climate records para sa **{target_year}**. Sakop ko ang {min_year} hanggang {max_year}. Hindi rin available ang ML prediction model."
            )

        return L(
            f"I don't have climate records for **{target_year}**. I cover {min_year} through {max_year}.",
            f"Wala akong climate records para sa **{target_year}**. Sakop ko ang {min_year} hanggang {max_year}."
        )

    # Historical data exists: save conversational state
    conversation_state["last_year"] = target_year
    conversation_state["last_topic"] = "year_summary"
    is_tl = conversation_state.get("lang") == "tl"

    # Extract climate metadata
    phases = list(year_data['ens_phase'].dropna().unique())
    phase_labels = ', '.join(label_phase(p) for p in phases) if phases else "Neutral 🟡"
    avg_temp_s1 = year_data['temp_batangas'].mean()

    # Always render the modern visual card for rainfall responses
    return render_rainfall_chart_card(
        year_num=target_year,
        year_data=year_data,
        is_tl=is_tl,
        enso_phase_label=phase_labels,
        avg_temp=avg_temp_s1
    )

def handle_month_comparison(month1_num, month2_num, padre_garcia_only=False):
    """Compare average historical climate data between two calendar months across all years."""
    if master_df is None: return "Dataset not loaded."

    m1_name = MONTH_NAMES[month1_num]
    m2_name = MONTH_NAMES[month2_num]
    r3_col = 'rain_basco' if 'rain_basco' in master_df.columns else 'rain_station3'
    t3_col = 'temp_basco' if 'temp_basco' in master_df.columns else 'temp_station3'

    m1_data = master_df[master_df['Month'] == month1_num]
    m2_data = master_df[master_df['Month'] == month2_num]

    if m1_data.empty or m2_data.empty:
        return L("Not enough data to compare those months.", "Hindi sapat ang data para ihambing ang mga buwang iyon.")

    # Padre Garcia averages
    m1_rain_s1 = m1_data['rain_batangas'].mean()
    m2_rain_s1 = m2_data['rain_batangas'].mean()
    m1_temp_s1 = m1_data['temp_batangas'].mean()
    m2_temp_s1 = m2_data['temp_batangas'].mean()

    # Regional averages
    m1_rain_s2 = m1_data['rain_station2'].mean()
    m2_rain_s2 = m2_data['rain_station2'].mean()
    m1_rain_s3 = m1_data[r3_col].mean()
    m2_rain_s3 = m2_data[r3_col].mean()

    # Wettest single month on record for each
    m1_max = m1_data.loc[m1_data['rain_batangas'].idxmax()]
    m2_max = m2_data.loc[m2_data['rain_batangas'].idxmax()]
    m1_min = m1_data.loc[m1_data['rain_batangas'].idxmin()]
    m2_min = m2_data.loc[m2_data['rain_batangas'].idxmin()]

    wetter = m1_name if m1_rain_s1 > m2_rain_s1 else m2_name
    rain_diff = abs(m1_rain_s1 - m2_rain_s1)

    conversation_state["last_topic"] = "month_comparison"

    if padre_garcia_only:
        if conversation_state.get("lang") == "tl":
            return (
                f"⚔️ **Padre Garcia — {m1_name} vs {m2_name} (Historical Average):**\n\n"
                f"📊 **Average Monthly Rainfall (2010–2026):**\n"
                f"• **{m1_name}:** {m1_rain_s1:.1f} mm | {m1_temp_s1:.1f}°C avg temp\n"
                f"• **{m2_name}:** {m2_rain_s1:.1f} mm | {m2_temp_s1:.1f}°C avg temp\n\n"
                f"📈 **Trend:** Karaniwang mas malakas ang ulan sa **{wetter}** ng {rain_diff:.1f} mm sa average.\n\n"
                f"🏆 **Record highs sa Padre Garcia:**\n"
                f"• {m1_name}: {m1_max['rain_batangas']:.1f} mm ({int(m1_max['Year'])})\n"
                f"• {m2_name}: {m2_max['rain_batangas']:.1f} mm ({int(m2_max['Year'])})\n\n"
                f"📉 **Record lows sa Padre Garcia:**\n"
                f"• {m1_name}: {m1_min['rain_batangas']:.1f} mm ({int(m1_min['Year'])})\n"
                f"• {m2_name}: {m2_min['rain_batangas']:.1f} mm ({int(m2_min['Year'])})\n\n"
                f"💡 *Para sa isang tukoy na taon, subukan: '{m1_name} 2015' o '{m2_name} 2020'*"
            )
        return (
            f"⚔️ **Padre Garcia — {m1_name} vs {m2_name} (Historical Average):**\n\n"
            f"📊 **Average Monthly Rainfall (2010–2026):**\n"
            f"• **{m1_name}:** {m1_rain_s1:.1f} mm | {m1_temp_s1:.1f}°C avg temp\n"
            f"• **{m2_name}:** {m2_rain_s1:.1f} mm | {m2_temp_s1:.1f}°C avg temp\n\n"
            f"📈 **Trend:** On average, **{wetter}** is wetter by {rain_diff:.1f} mm.\n\n"
            f"🏆 **Record highs in Padre Garcia:**\n"
            f"• {m1_name}: {m1_max['rain_batangas']:.1f} mm ({int(m1_max['Year'])})\n"
            f"• {m2_name}: {m2_max['rain_batangas']:.1f} mm ({int(m2_max['Year'])})\n\n"
            f"📉 **Record lows in Padre Garcia:**\n"
            f"• {m1_name}: {m1_min['rain_batangas']:.1f} mm ({int(m1_min['Year'])})\n"
            f"• {m2_name}: {m2_min['rain_batangas']:.1f} mm ({int(m2_min['Year'])})\n\n"
            f"💡 *For a specific year, try: '{m1_name} 2015' or '{m2_name} 2020'*"
        )

    if conversation_state.get("lang") == "tl":
        return (
            f"⚔️ **Month Comparison: {m1_name} vs {m2_name} (Historical Average, 2010–2026)**\n\n"
            f"📍 **Padre Garcia, Batangas:**\n"
            f"• {m1_name}: **{m1_rain_s1:.1f} mm** avg rainfall | {m1_temp_s1:.1f}°C avg temp\n"
            f"• {m2_name}: **{m2_rain_s1:.1f} mm** avg rainfall | {m2_temp_s1:.1f}°C avg temp\n"
            f"• Mas malakas ang ulan sa **{wetter}** ng {rain_diff:.1f} mm sa average\n\n"
            f"📍 **Quezon / Bicol Region:**\n"
            f"• {m1_name}: {m1_rain_s2:.1f} mm avg | {m2_name}: {m2_rain_s2:.1f} mm avg\n\n"
            f"📍 **Basco, Batanes:**\n"
            f"• {m1_name}: {m1_rain_s3:.1f} mm avg | {m2_name}: {m2_rain_s3:.1f} mm avg\n\n"
            f"🏆 **Record highs sa Padre Garcia:**\n"
            f"• {m1_name}: {m1_max['rain_batangas']:.1f} mm ({int(m1_max['Year'])})\n"
            f"• {m2_name}: {m2_max['rain_batangas']:.1f} mm ({int(m2_max['Year'])})\n\n"
            f"💡 *Para sa isang tukoy na taon, subukan: '{m1_name} 2015' o '{m2_name} 2020'*"
        )
    return (
        f"⚔️ **Month Comparison: {m1_name} vs {m2_name} (Historical Average, 2010–2026)**\n\n"
        f"📍 **Padre Garcia, Batangas:**\n"
        f"• {m1_name}: **{m1_rain_s1:.1f} mm** avg rainfall | {m1_temp_s1:.1f}°C avg temp\n"
        f"• {m2_name}: **{m2_rain_s1:.1f} mm** avg rainfall | {m2_temp_s1:.1f}°C avg temp\n"
        f"• On average, **{wetter}** is wetter by {rain_diff:.1f} mm\n\n"
        f"📍 **Quezon / Bicol Region:**\n"
        f"• {m1_name}: {m1_rain_s2:.1f} mm avg | {m2_name}: {m2_rain_s2:.1f} mm avg\n\n"
        f"📍 **Basco, Batanes:**\n"
        f"• {m1_name}: {m1_rain_s3:.1f} mm avg | {m2_name}: {m2_rain_s3:.1f} mm avg\n\n"
        f"🏆 **Record highs in Padre Garcia:**\n"
        f"• {m1_name}: {m1_max['rain_batangas']:.1f} mm ({int(m1_max['Year'])})\n"
        f"• {m2_name}: {m2_max['rain_batangas']:.1f} mm ({int(m2_max['Year'])})\n\n"
        f"💡 *For a specific year, try: '{m1_name} 2015' or '{m2_name} 2020'*"
    )


def handle_month_year_query(month_num, year_num, padre_garcia_only=False):
    if master_df is None: return "Dataset not loaded."
    data = master_df[(master_df['Year'] == year_num) & (master_df['Month'] == month_num)]
    if data.empty:
        max_year = int(master_df['Year'].max())
        min_year = int(master_df['Year'].min())
        if year_num > max_year:
            # Future year — redirect to ML rainfall prediction
            if rain_model:
                return handle_ml_forecast_query(month_num, year_num)
            return L(
                f"I couldn't find data for **{MONTH_NAMES.get(month_num, month_num)} {year_num}**. I cover {min_year} through {max_year}, and the ML prediction model is unavailable.",
                f"Hindi ko mahanap ang data para sa **{MONTH_NAMES.get(month_num, month_num)} {year_num}**. Sakop ko ang {min_year} hanggang {max_year}, at hindi available ang ML prediction model."
            )
        return L(
            f"I couldn't find data for **{MONTH_NAMES.get(month_num, month_num)} {year_num}**.",
            f"Hindi ko mahanap ang data para sa **{MONTH_NAMES.get(month_num, month_num)} {year_num}**."
        )

    row = data.iloc[0]
    conversation_state["last_year"] = year_num
    conversation_state["last_month"] = month_num
    conversation_state["last_topic"] = "month_snapshot"
    m_name = MONTH_NAMES[month_num]
    t3_col = 'temp_basco' if 'temp_basco' in master_df.columns else 'temp_station3'
    r3_col = 'rain_basco' if 'rain_basco' in master_df.columns else 'rain_station3'

    # Monthly averages for context
    month_avg_rain = master_df[master_df['Month'] == month_num]['rain_batangas'].mean()
    month_avg_temp = master_df[master_df['Month'] == month_num]['temp_batangas'].mean()
    rain_s1 = row['rain_batangas']
    temp_s1 = row['temp_batangas']
    rain_label_en = "above average 📈" if rain_s1 > month_avg_rain else "below average 📉"
    rain_label_tl = "above average 📈" if rain_s1 > month_avg_rain else "below average 📉"
    temp_label_en = "above average 🌡️" if temp_s1 > month_avg_temp else "below average"
    temp_label_tl = "above average 🌡️" if temp_s1 > month_avg_temp else "below average"

    if padre_garcia_only:
        if conversation_state.get("lang") == "tl":
            return (
                f"🗓️ **Padre Garcia, Batangas — {m_name} {year_num}:**\n"
                f"• **ONI:** {label_oni(row['oni_value'])}\n"
                f"• **Rainfall:** {rain_s1:.1f} mm ({rain_label_tl}, avg: {month_avg_rain:.1f} mm)\n"
                f"• **Temperature:** {temp_s1:.1f}°C ({temp_label_tl}, avg: {month_avg_temp:.1f}°C)\n"
                f"• **Humidity:** {row['humidity_batangas']:.1f}%"
            )
        return (
            f"🗓️ **Padre Garcia, Batangas — {m_name} {year_num}:**\n"
            f"• **ONI:** {label_oni(row['oni_value'])}\n"
            f"• **Rainfall:** {rain_s1:.1f} mm ({rain_label_en}, avg: {month_avg_rain:.1f} mm)\n"
            f"• **Temperature:** {temp_s1:.1f}°C ({temp_label_en}, avg: {month_avg_temp:.1f}°C)\n"
            f"• **Humidity:** {row['humidity_batangas']:.1f}%"
        )

    if conversation_state.get("lang") == "tl":
        return (
            f"🗓️ **Climate Snapshot — {m_name} {year_num}:**\n"
            f"• **ONI:** {label_oni(row['oni_value'])}\n\n"
            f"📍 **Padre Garcia, Batangas:** {rain_s1:.1f} mm ({rain_label_tl}, avg: {month_avg_rain:.1f} mm) | {temp_s1:.1f}°C\n"
            f"📍 **Quezon / Bicol Region:** {row['rain_station2']:.1f} mm | {row['temp_station2']:.1f}°C\n"
            f"📍 **Basco, Batanes:** {row[r3_col]:.1f} mm | {row[t3_col]:.1f}°C"
        )
    return (
        f"🗓️ **Climate Snapshot — {m_name} {year_num}:**\n"
        f"• **ONI:** {label_oni(row['oni_value'])}\n\n"
        f"📍 **Padre Garcia, Batangas:** {rain_s1:.1f} mm ({rain_label_en}, avg: {month_avg_rain:.1f} mm) | {temp_s1:.1f}°C\n"
        f"📍 **Quezon / Bicol Region:** {row['rain_station2']:.1f} mm | {row['temp_station2']:.1f}°C\n"
        f"📍 **Basco, Batanes:** {row[r3_col]:.1f} mm | {row[t3_col]:.1f}°C"
    )
# ==========================================
# MODERN COMPARISON CHART & CARD GENERATOR
# ==========================================

def render_comparison_chart_card(year1, year2, y1_data, y2_data, is_tl=False, incomplete_note=""):
    """
    Renders an interactive, side-by-side comparison bar chart card for two years.
    Uses blue bars for Year 1 and amber/orange bars for Year 2.
    """
    months_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    vals1 = []
    vals2 = []
    for m in range(1, 13):
        r1 = y1_data[y1_data['Month'] == m]
        r2 = y2_data[y2_data['Month'] == m]
        vals1.append(float(r1['rain_batangas'].values[0]) if not r1.empty and pd.notna(r1['rain_batangas'].values[0]) else 0.0)
        vals2.append(float(r2['rain_batangas'].values[0]) if not r2.empty and pd.notna(r2['rain_batangas'].values[0]) else 0.0)

    total1 = sum(vals1)
    total2 = sum(vals2)
    avg_temp1 = y1_data['temp_batangas'].mean()
    avg_temp2 = y2_data['temp_batangas'].mean()

    # ENSO phase labels
    p1 = list(y1_data['ens_phase'].dropna().unique())
    p2 = list(y2_data['ens_phase'].dropna().unique())
    p1_labels = ', '.join(label_phase(p) for p in p1) if p1 else "Neutral 🟡"
    p2_labels = ', '.join(label_phase(p) for p in p2) if p2 else "Neutral 🟡"

    # Difference calculation
    diff = abs(total1 - total2)
    wetter_year = year1 if total1 > total2 else year2
    drier_year = year2 if total1 > total2 else year1
    diff_pct = (diff / min(total1, total2) * 100) if min(total1, total2) > 0 else 0

    # Chart Y-axis scaling
    peak = max(max(vals1), max(vals2), 10.0)
    top_limit = peak * 1.15
    if top_limit <= 100:
        step = 25
    elif top_limit <= 200:
        step = 45
    elif top_limit <= 400:
        step = 80
    else:
        step = int(top_limit / 4 / 25) * 25 or 50

    top_y = step * 4
    if top_y < peak:
        top_y = step * 5

    y_levels = [top_y, step * 3, step * 2, step * 1, 0]

    # SVG layout specs
    chart_height = 210
    chart_width = 560
    margin_left = 60
    margin_right = 20
    plot_width = chart_width - margin_left - margin_right
    slot_width = plot_width / 12
    bar_width = 9  # Two bars side by side per month

    bars_svg = []
    labels_svg = []

    for i, (v1, v2, abbr) in enumerate(zip(vals1, vals2, months_abbr)):
        x_slot = margin_left + (i * slot_width)
        
        # Bar 1 (Year 1 - Blue)
        b1_x = x_slot + (slot_width - (bar_width * 2 + 3)) / 2
        b1_h = (v1 / top_y) * (chart_height - 45) if top_y > 0 else 0
        b1_y = (chart_height - 30) - b1_h

        # Bar 2 (Year 2 - Amber)
        b2_x = b1_x + bar_width + 3
        b2_h = (v2 / top_y) * (chart_height - 45) if top_y > 0 else 0
        b2_y = (chart_height - 30) - b2_h

        tooltip_x = b1_x + bar_width
        tooltip_y = max(min(b1_y, b2_y) - 34, 4)

        bars_svg.append(f"""
            <g class="bar-group group cursor-pointer">
                <!-- Bar 1 ({year1}) -->
                <rect x="{b1_x:.1f}" y="{b1_y:.1f}" width="{bar_width}" height="{max(b1_h, 3):.1f}" rx="3"
                      class="fill-blue-500 hover:fill-blue-600 dark:fill-blue-400 transition-all duration-150" />
                <!-- Bar 2 ({year2}) -->
                <rect x="{b2_x:.1f}" y="{b2_y:.1f}" width="{bar_width}" height="{max(b2_h, 3):.1f}" rx="3"
                      class="fill-amber-500 hover:fill-amber-600 dark:fill-amber-400 transition-all duration-150" />
                <!-- Hover Dual Tooltip -->
                <g class="bar-tooltip opacity-0 transition-all duration-150 pointer-events-none">
                    <rect x="{tooltip_x - 45:.1f}" y="{tooltip_y:.1f}" width="90" height="28" rx="6"
                          class="fill-slate-900 dark:fill-slate-100 shadow-md" />
                    <text x="{tooltip_x:.1f}" y="{tooltip_y + 12:.1f}" text-anchor="middle"
                          class="fill-blue-300 dark:fill-blue-600 text-[9px] font-semibold">{year1}: {v1:.0f} mm</text>
                    <text x="{tooltip_x:.1f}" y="{tooltip_y + 23:.1f}" text-anchor="middle"
                          class="fill-amber-300 dark:fill-amber-600 text-[9px] font-semibold">{year2}: {v2:.0f} mm</text>
                </g>
            </g>
        """)

        labels_svg.append(f"""
            <text x="{b1_x + bar_width + 1.5:.1f}" y="{chart_height - 10}" text-anchor="middle"
                  class="fill-slate-500 dark:fill-slate-400 text-[10px] font-medium">{abbr}</text>
        """)

    grid_svg = []
    for lvl in y_levels:
        y_pos = 14 + (chart_height - 44) * (1.0 - (lvl / top_y)) if top_y > 0 else 14
        grid_svg.append(f"""
            <line x1="{margin_left}" y1="{y_pos:.1f}" x2="{chart_width - margin_right}" y2="{y_pos:.1f}"
                  stroke="currentColor" stroke-dasharray="3,3" class="stroke-slate-200 dark:stroke-slate-800" stroke-width="1"/>
            <text x="{margin_left - 8}" y="{y_pos + 3.5:.1f}" text-anchor="end"
                  class="fill-slate-400 dark:fill-slate-500 text-[10px]">{int(lvl)} mm</text>
        """)

    # Localized text strings
    badge_text = "Historical Comparison · Padre Garcia, Batangas"
    header_title = f"Rainfall Comparison — {year1} vs {year2}" if not is_tl else f"Paghahambing ng Ulan — {year1} vs {year2}"
    chart_title = f"Monthly Rainfall Comparison ({year1} vs {year2})" if not is_tl else f"Buwanang Paghahambing ng Ulan ({year1} vs {year2})"
    chart_subtitle = "Hover over bars to see monthly measurements side-by-side." if not is_tl else "Itapat ang cursor sa bar upang makita ang buwanang sukat."

    summary_headline = (
        f"In Padre Garcia, <strong>{wetter_year}</strong> recorded <strong>{diff:.1f} mm</strong> (+{diff_pct:.0f}%) more rainfall than {drier_year}."
        if not is_tl else
        f"Sa Padre Garcia, ang taong <strong>{wetter_year}</strong> ay nagtala ng <strong>{diff:.1f} mm</strong> (+{diff_pct:.0f}%) mas maraming ulan kaysa sa {drier_year}."
    )

    verdict_badge = (
        f"📈 {wetter_year} was wetter by {diff:.0f} mm" if not is_tl else f"📈 Mas maulan noong {wetter_year} ng {diff:.0f} mm"
    )

    note_html = f"<div class='p-2.5 rounded-lg bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 text-xs mb-2 border border-amber-200/60 dark:border-amber-800/60'>{incomplete_note}</div>" if incomplete_note else ""

    return f"""<!-- RAINFALL_CARD -->
<div class="w-full my-2 space-y-4 animate-fade-in font-sans">
    {note_html}
    <!-- Header -->
    <div>
        <div class="flex items-center gap-2 mb-1.5 flex-wrap">
            <!-- <span class="text-base">⚖️</span> -->
            <span class="font-semibold text-slate-900 dark:text-white text-base">{header_title}</span>
            <span class="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-100 text-blue-800 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200/60 dark:border-blue-800/60">
                {badge_text}
            </span>
        </div>
        <p class="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">{summary_headline}</p>
    </div>

    <!-- Chart Container Card -->
    <div class="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900/90 border border-slate-200/90 dark:border-slate-800 shadow-xs">
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
            <div>
                <h4 class="font-semibold text-slate-900 dark:text-white text-sm sm:text-base">{chart_title}</h4>
                <p class="text-[11px] sm:text-xs text-slate-400 dark:text-slate-500 mt-0.5">{chart_subtitle}</p>
            </div>
            <!-- Legend -->
            <div class="flex items-center gap-3 text-xs self-start sm:self-auto pt-1">
                <span class="inline-flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
                    <span class="w-3 h-3 rounded-xs bg-blue-500 inline-block"></span>
                    <span>{year1}</span>
                </span>
                <span class="inline-flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
                    <span class="w-3 h-3 rounded-xs bg-amber-500 inline-block"></span>
                    <span>{year2}</span>
                </span>
            </div>
        </div>

        <!-- SVG Grouped Bar Chart -->
        <div class="w-full overflow-x-auto">
            <svg viewBox="0 0 {chart_width} {chart_height}" class="w-full h-52 sm:h-60 select-none">
                {''.join(grid_svg)}
                {''.join(bars_svg)}
                {''.join(labels_svg)}
            </svg>
        </div>
    </div>

    <!-- 2-Column Side-by-Side Metric Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        <!-- Year 1 Card -->
        <div class="p-4 rounded-xl bg-white dark:bg-slate-900/90 border border-blue-200/80 dark:border-blue-900/50 shadow-xs">
            <div class="flex items-center justify-between mb-2">
                <span class="font-semibold text-slate-900 dark:text-white text-sm sm:text-base flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                    <span>{year1} Summary</span>
                </span>
                <span class="text-[11px] px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">{p1_labels}</span>
            </div>
            <div class="space-y-1 text-xs text-slate-600 dark:text-slate-300">
                <div class="flex justify-between py-0.5 border-b border-slate-100 dark:border-slate-800/60">
                    <span class="text-slate-400">Total Rainfall:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{total1:.1f} mm</strong>
                </div>
                <div class="flex justify-between py-0.5 border-b border-slate-100 dark:border-slate-800/60">
                    <span class="text-slate-400">Monthly Average:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{(total1 / 12.0):.1f} mm</strong>
                </div>
                <div class="flex justify-between py-0.5">
                    <span class="text-slate-400">Average Temp:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{avg_temp1:.1f}°C</strong>
                </div>
            </div>
        </div>

        <!-- Year 2 Card -->
        <div class="p-4 rounded-xl bg-white dark:bg-slate-900/90 border border-amber-200/80 dark:border-amber-900/50 shadow-xs">
            <div class="flex items-center justify-between mb-2">
                <span class="font-semibold text-slate-900 dark:text-white text-sm sm:text-base flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                    <span>{year2} Summary</span>
                </span>
                <span class="text-[11px] px-2 py-0.5 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">{p2_labels}</span>
            </div>
            <div class="space-y-1 text-xs text-slate-600 dark:text-slate-300">
                <div class="flex justify-between py-0.5 border-b border-slate-100 dark:border-slate-800/60">
                    <span class="text-slate-400">Total Rainfall:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{total2:.1f} mm</strong>
                </div>
                <div class="flex justify-between py-0.5 border-b border-slate-100 dark:border-slate-800/60">
                    <span class="text-slate-400">Monthly Average:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{(total2 / 12.0):.1f} mm</strong>
                </div>
                <div class="flex justify-between py-0.5">
                    <span class="text-slate-400">Average Temp:</span>
                    <strong class="text-slate-900 dark:text-white font-semibold">{avg_temp2:.1f}°C</strong>
                </div>
            </div>
        </div>
    </div>

    <!-- Bottom Comparative Verdict -->
    <div class="p-3 rounded-xl bg-slate-100/80 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-700/60 flex items-center justify-between text-xs flex-wrap gap-2">
        <span class="font-medium text-slate-700 dark:text-slate-200">{verdict_badge}</span>
        <span class="text-[11px] text-slate-500 dark:text-slate-400">Difference: <strong>{diff:.1f} mm</strong></span>
    </div>
</div>
"""

def handle_year_comparison(year1, year2, padre_garcia_only=False):
    if master_df is None: return "Dataset not loaded."
    
    y1_data = master_df[master_df['Year'] == int(year1)]
    y2_data = master_df[master_df['Year'] == int(year2)]

    if y1_data.empty or y2_data.empty:
        max_year = int(master_df['Year'].max())
        min_year = int(master_df['Year'].min())
        future_years = [y for y in [year1, year2] if int(y) > max_year]
        if future_years:
            yr_list = ', '.join(future_years)
            return L(
                f"I can only compare years with historical records ({min_year}–{max_year}). **{yr_list}** has no recorded data yet.",
                f"Maari ko lamang ihambing ang mga taon na may historical records ({min_year}–{max_year}). Wala pang recorded data ang **{yr_list}**."
            )
        return L(
            f"I can only compare years between {min_year} and {max_year}. Please check your requested years!",
            f"Maari ko lamang ihambing ang mga taon mula {min_year} hanggang {max_year}. Pakiusap suriin ang iyong mga hininging taon!"
        )

    # Check for incomplete months
    max_months = master_df.groupby('Year')['Month'].count().max()
    incomplete_notes = []
    if len(y1_data) < max_months:
        incomplete_notes.append(f"Year {year1} has only {len(y1_data)} recorded months.")
    if len(y2_data) < max_months:
        incomplete_notes.append(f"Year {year2} has only {len(y2_data)} recorded months.")
    
    note_str = " ".join(incomplete_notes)
    conversation_state["last_topic"] = "year_summary"
    is_tl = conversation_state.get("lang") == "tl"

    # Check if Station Comparison Mode is active or explicitly asked
    show_all_stations = conversation_state.get("station_mode", False) and not padre_garcia_only

    # 1. Main modern card (Padre Garcia)
    main_card = render_comparison_chart_card(
        year1=str(year1),
        year2=str(year2),
        y1_data=y1_data,
        y2_data=y2_data,
        is_tl=is_tl,
        incomplete_note=note_str
    )

    # 2. If Station Comparison Mode is OFF, add an inviting tip chip
    if not show_all_stations:
        tip_text = (
            "<div class='mt-2 text-center'>"
            "<button onclick=\"sendQuickPrompt('Turn on station comparison mode')\" class='text-[11px] text-slate-500 hover:text-brand-600 dark:text-slate-400 dark:hover:text-brand-400 transition-colors inline-flex items-center gap-1.5 p-1.5 rounded-lg bg-slate-100/60 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-700/60'>"
            "<i class='fa-solid fa-arrows-split-up-and-left text-[10px]'></i>"
            "<span>Want to compare with Quezon/Bicol and Basco? <strong>Turn on station comparison mode &rarr;</strong></span>"
            "</button></div>"
            if not is_tl else
            "<div class='mt-2 text-center'>"
            "<button onclick=\"sendQuickPrompt('Turn on station comparison mode')\" class='text-[11px] text-slate-500 hover:text-brand-600 dark:text-slate-400 dark:hover:text-brand-400 transition-colors inline-flex items-center gap-1.5 p-1.5 rounded-lg bg-slate-100/60 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-700/60'>"
            "<i class='fa-solid fa-arrows-split-up-and-left text-[10px]'></i>"
            "<span>Gusto mo bang ihambing sa Quezon/Bicol at Basco? <strong>I-on ang station comparison mode &rarr;</strong></span>"
            "</button></div>"
        )
        return main_card + tip_text

    # 3. If Station Comparison Mode is ON: Append Quezon and Basco comparison metrics!
    r3_col = 'rain_basco' if 'rain_basco' in master_df.columns else 'rain_station3'
    t3_col = 'temp_basco' if 'temp_basco' in master_df.columns else 'temp_station3'

    # Station 2 (Quezon/Bicol)
    r1_s2 = y1_data['rain_station2'].sum()
    r2_s2 = y2_data['rain_station2'].sum()
    t1_s2 = y1_data['temp_station2'].mean()
    t2_s2 = y2_data['temp_station2'].mean()

    # Station 3 (Basco)
    r1_s3 = y1_data[r3_col].sum()
    r2_s3 = y2_data[r3_col].sum()
    t1_s3 = y1_data[t3_col].mean()
    t2_s3 = y2_data[t3_col].mean()

    regional_badge = " 3-Station Regional Comparison" if not is_tl else "3-Station Regional Comparison"
    s2_label = "Quezon / Bicol Region"
    s3_label = "Basco, Batanes"

    station_expansion = f"""
    <div class="mt-4 pt-3 border-t border-slate-200 dark:border-slate-800 space-y-2.5 animate-fade-in">
        <div class="flex items-center justify-between text-xs font-semibold text-slate-800 dark:text-slate-200">
            <span class="flex items-center gap-1.5"><i class="fa-solid fa-tower-broadcast text-blue-500"></i> {regional_badge}</span>
            <button onclick="sendQuickPrompt('Turn off station comparison mode')" class="text-[10px] text-slate-400 hover:text-rose-500">(Turn off)</button>
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            <!-- Quezon -->
            <div class="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800">
                <span class="font-medium text-slate-900 dark:text-white block mb-1">📍 {s2_label}</span>
                <div class="space-y-0.5 text-slate-600 dark:text-slate-300">
                    <div>• <strong>{year1}:</strong> {r1_s2:.1f} mm | {t1_s2:.1f}°C</div>
                    <div>• <strong>{year2}:</strong> {r2_s2:.1f} mm | {t2_s2:.1f}°C</div>
                    <div class="text-[11px] text-slate-400 pt-1">Diff: {abs(r1_s2 - r2_s2):.1f} mm ({year1 if r1_s2 > r2_s2 else year2} was wetter)</div>
                </div>
            </div>
            <!-- Basco -->
            <div class="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800">
                <span class="font-medium text-slate-900 dark:text-white block mb-1">📍 {s3_label}</span>
                <div class="space-y-0.5 text-slate-600 dark:text-slate-300">
                    <div>• <strong>{year1}:</strong> {r1_s3:.1f} mm | {t1_s3:.1f}°C</div>
                    <div>• <strong>{year2}:</strong> {r2_s3:.1f} mm | {t2_s3:.1f}°C</div>
                    <div class="text-[11px] text-slate-400 pt-1">Diff: {abs(r1_s3 - r2_s3):.1f} mm ({year1 if r1_s3 > r2_s3 else year2} was wetter)</div>
                </div>
            </div>
        </div>
    </div>
    """

    return main_card + station_expansion

def handle_rain_query(padre_garcia_only=False):
    if master_df is None: return "Dataset not loaded."
    r3_col = 'rain_basco' if 'rain_basco' in master_df.columns else 'rain_station3'
    s1_max = master_df.loc[master_df['rain_batangas'].idxmax()]
    s1_min = master_df.loc[master_df['rain_batangas'].idxmin()]

    conversation_state["last_topic"] = "rain_records"

    if padre_garcia_only:
        if conversation_state.get("lang") == "tl":
            return (
                f"🌧️ **Padre Garcia, Batangas — Rainfall Records:**\n"
                f"• **Wettest month:** {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — **{s1_max['rain_batangas']:.1f} mm**\n"
                f"• **Driest month:** {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — **{s1_min['rain_batangas']:.1f} mm**"
            )
        return (
            f"🌧️ **Padre Garcia, Batangas — Rainfall Records:**\n"
            f"• **Wettest month:** {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — **{s1_max['rain_batangas']:.1f} mm**\n"
            f"• **Driest month:** {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — **{s1_min['rain_batangas']:.1f} mm**"
        )

    s2_max = master_df.loc[master_df['rain_station2'].idxmax()]
    s3_max = master_df.loc[master_df[r3_col].idxmax()]

    if conversation_state.get("lang") == "tl":
        return (
            f"🌧️ **All-Time Rainfall Records:**\n\n"
            f"📍 **Padre Garcia, Batangas:**\n"
            f"  - Wettest: {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — **{s1_max['rain_batangas']:.1f} mm**\n"
            f"  - Driest: {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — **{s1_min['rain_batangas']:.1f} mm**\n\n"
            f"📍 **Quezon / Bicol Region:** Wettest — {MONTH_NAMES[int(s2_max['Month'])]} {int(s2_max['Year'])} sa **{s2_max['rain_station2']:.1f} mm**\n"
            f"📍 **Basco, Batanes:** Wettest — {MONTH_NAMES[int(s3_max['Month'])]} {int(s3_max['Year'])} sa **{s3_max[r3_col]:.1f} mm**"
        )
    return (
        f"🌧️ **All-Time Rainfall Records:**\n\n"
        f"📍 **Padre Garcia, Batangas:**\n"
        f"  - Wettest: {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — **{s1_max['rain_batangas']:.1f} mm**\n"
        f"  - Driest: {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — **{s1_min['rain_batangas']:.1f} mm**\n\n"
        f"📍 **Quezon / Bicol Region:** Wettest — {MONTH_NAMES[int(s2_max['Month'])]} {int(s2_max['Year'])} at **{s2_max['rain_station2']:.1f} mm**\n"
        f"📍 **Basco, Batanes:** Wettest — {MONTH_NAMES[int(s3_max['Month'])]} {int(s3_max['Year'])} at **{s3_max[r3_col]:.1f} mm**"
    )

def handle_temp_query(padre_garcia_only=False):
    if master_df is None: return "Dataset not loaded."
    t3_col = 'temp_basco' if 'temp_basco' in master_df.columns else 'temp_station3'
    s1_max = master_df.loc[master_df['temp_batangas'].idxmax()]
    s1_min = master_df.loc[master_df['temp_batangas'].idxmin()]

    conversation_state["last_topic"] = "temp_records"

    if padre_garcia_only:
        if conversation_state.get("lang") == "tl":
            return (
                f"🌡️ **Padre Garcia, Batangas — Temperature Records:**\n"
                f"• **Hottest month:** {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — **{s1_max['temp_batangas']:.1f}°C**\n"
                f"• **Coolest month:** {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — **{s1_min['temp_batangas']:.1f}°C**"
            )
        return (
            f"🌡️ **Padre Garcia, Batangas — Temperature Records:**\n"
            f"• **Hottest month:** {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — **{s1_max['temp_batangas']:.1f}°C**\n"
            f"• **Coolest month:** {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — **{s1_min['temp_batangas']:.1f}°C**"
        )

    s2_max = master_df.loc[master_df['temp_station2'].idxmax()]
    s3_max = master_df.loc[master_df[t3_col].idxmax()]

    if conversation_state.get("lang") == "tl":
        return (
            f"🌡️ **All-Time Temperature Records:**\n\n"
            f"📍 **Padre Garcia, Batangas:**\n"
            f"  - Hottest: {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — **{s1_max['temp_batangas']:.1f}°C**\n"
            f"  - Coolest: {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — **{s1_min['temp_batangas']:.1f}°C**\n\n"
            f"📍 **Quezon / Bicol Region:** Hottest — {MONTH_NAMES[int(s2_max['Month'])]} {int(s2_max['Year'])} sa **{s2_max['temp_station2']:.1f}°C**\n"
            f"📍 **Basco, Batanes:** Hottest — {MONTH_NAMES[int(s3_max['Month'])]} {int(s3_max['Year'])} sa **{s3_max[t3_col]:.1f}°C**"
        )
    return (
        f"🌡️ **All-Time Temperature Records:**\n\n"
        f"📍 **Padre Garcia, Batangas:**\n"
        f"  - Hottest: {MONTH_NAMES[int(s1_max['Month'])]} {int(s1_max['Year'])} — **{s1_max['temp_batangas']:.1f}°C**\n"
        f"  - Coolest: {MONTH_NAMES[int(s1_min['Month'])]} {int(s1_min['Year'])} — **{s1_min['temp_batangas']:.1f}°C**\n\n"
        f"📍 **Quezon / Bicol Region:** Hottest — {MONTH_NAMES[int(s2_max['Month'])]} {int(s2_max['Year'])} at **{s2_max['temp_station2']:.1f}°C**\n"
        f"📍 **Basco, Batanes:** Hottest — {MONTH_NAMES[int(s3_max['Month'])]} {int(s3_max['Year'])} at **{s3_max[t3_col]:.1f}°C**"
    )

# ==========================================
# LOCAL RELATED QUESTIONS SUGGESTION ENGINE
# ==========================================

# ==========================================
# LOCAL RELATED QUESTIONS SUGGESTION ENGINE
# ==========================================

TOPIC_SUGGESTIONS = {
    # Machine Learning & AI Forecasting
    "ml_model_used": {
        "en": [
            "How accurate is the rainfall prediction?",
            "What data is used for prediction?",
            "Predict rain for May"
        ],
        "tl": [
            "Gaano katumpak ang rainfall prediction?",
            "Anong data ang ginagamit para sa prediction?",
            "Hulaan ang ulan para sa May"
        ]
    },
    "prediction": {
        "en": [
            "What ML model is used in this chatbot?",
            "How accurate is the rainfall prediction?",
            "What data is used for prediction?"
        ],
        "tl": [
            "Anong ML model ang ginagamit sa chatbot na ito?",
            "Gaano katumpak ang rainfall prediction?",
            "Anong data ang ginagamit para sa prediction?"
        ]
    },
    "prediction_explainer": {
        "en": [
            "What ML model is used in this chatbot?",
            "How accurate is the rainfall prediction?",
            "Predict rain for May"
        ],
        "tl": [
            "Anong ML model ang ginagamit sa chatbot na ito?",
            "Gaano katumpak ang rainfall prediction?",
            "Hulaan ang ulan para sa May"
        ]
    },
    "prediction_accuracy": {
        "en": [
            "What data is used for prediction?",
            "What ML model is used in this chatbot?",
            "Predict rain for May"
        ],
        "tl": [
            "Anong data ang ginagamit para sa prediction?",
            "Anong ML model ang ginagamit sa chatbot na ito?",
            "Hulaan ang ulan para sa May"
        ]
    },
    "prediction_data": {
        "en": [
            "What ML model is used in this chatbot?",
            "How accurate is the rainfall prediction?",
            "Predict rain for May"
        ],
        "tl": [
            "Anong ML model ang ginagamit sa chatbot na ito?",
            "Gaano katumpak ang rainfall prediction?",
            "Hulaan ang ulan para sa May"
        ]
    },
    "historical_vs_predicted": {
        "en": [
            "Predict rain for May",
            "What was the wettest month on record?",
            "Compare 2015 and 2020"
        ],
        "tl": [
            "Hulaan ang ulan para sa May",
            "Ano ang pinakamalakas na ulan sa rekord?",
            "Ihambing ang 2015 at 2020"
        ]
    },

    # Health & Heat
    "heat_exhaustion_signs": {
        "en": [
            "What should I do if someone experiences heat exhaustion?",
            "What is heatstroke?",
            "When should I seek medical help during extreme heat?"
        ],
        "tl": [
            "Ano ang dapat gawin kung may nakakaranas ng heat exhaustion?",
            "Ano ang heatstroke?",
            "Kailan dapat humingi ng medikal na tulong sa matinding init?"
        ]
    },
    "heat_exhaustion_action": {
        "en": [
            "What are the signs of dehydration?",
            "What should I do during extreme heat?",
            "How can I contact the RHU?"
        ],
        "tl": [
            "Ano ang mga senyales ng dehydration?",
            "Ano ang dapat gawin sa matinding init?",
            "Paano makipag-ugnayan sa RHU?"
        ]
    },
    "heatstroke_definition": {
        "en": [
            "What should I do if someone experiences heat exhaustion?",
            "What are the signs of dehydration?",
            "Who should I contact during an emergency?"
        ],
        "tl": [
            "Ano ang dapat gawin kung may heat exhaustion?",
            "Ano ang mga senyales ng dehydration?",
            "Sino ang dapat tawagan sa emergency?"
        ]
    },
    "dehydration_signs": {
        "en": [
            "How can I prevent dehydration?",
            "How should I store water safely?",
            "What should I do during extreme heat?"
        ],
        "tl": [
            "Paano maiwasan ang dehydration?",
            "Paano mag-imbak ng tubig nang ligtas?",
            "Ano ang dapat gawin sa matinding init?"
        ]
    },
    "dehydration_prevention": {
        "en": [
            "What are the signs of dehydration?",
            "What should I do during extreme heat?",
            "How should I store water safely?"
        ],
        "tl": [
            "Ano ang mga senyales ng dehydration?",
            "Ano ang dapat gawin sa matinding init?",
            "Paano mag-imbak ng tubig nang ligtas?"
        ]
    },
    "extreme_heat_action": {
        "en": [
            "What are the signs of heat exhaustion?",
            "How can I prevent dehydration?",
            "Does El Niño cause higher temperatures?"
        ],
        "tl": [
            "Ano ang mga senyales ng heat exhaustion?",
            "Paano maiwasan ang dehydration?",
            "Nagdudulot ba ng mas mataas na temperatura ang El Niño?"
        ]
    },
    "medical_seek_heat": {
        "en": [
            "How can I contact the RHU?",
            "What should I do if someone experiences heat exhaustion?",
            "Who should I contact during an emergency?"
        ],
        "tl": [
            "Paano makipag-ugnayan sa RHU?",
            "Ano ang dapat gawin kung may heat exhaustion?",
            "Sino ang dapat tawagan sa emergency?"
        ]
    },

    # El Niño & Climate Basics
    "el_nino": {
        "en": [
            "What causes El Niño?",
            "What should I do during El Niño?",
            "What is the difference between El Niño and La Niña?"
        ],
        "tl": [
            "Bakit nagkakaroon ng El Niño?",
            "Ano ang dapat gawin sa panahon ng El Niño?",
            "Ano ang pagkakaiba ng El Niño at La Niña?"
        ]
    },
    "el_nino_definition": {
        "en": [
            "What are the effects of El Niño?",
            "What is the difference between El Niño and La Niña?",
            "What should I do during El Niño?"
        ],
        "tl": [
            "Ano ang mga epekto ng El Niño?",
            "Ano ang pagkakaiba ng El Niño at La Niña?",
            "Ano ang dapat gawin sa panahon ng El Niño?"
        ]
    },
    "el_nino_causes": {
        "en": [
            "How does El Niño work?",
            "How long does El Niño last?",
            "Does El Niño happen every year?"
        ],
        "tl": [
            "Paano gumagana ang El Niño?",
            "Gaano katagal ang El Niño?",
            "Nagkakaroon ba ng El Niño tuwing taon?"
        ]
    },
    "el_nino_how": {
        "en": [
            "How long does El Niño last?",
            "Does El Niño reduce rainfall?",
            "What is ENSO?"
        ],
        "tl": [
            "Gaano katagal ang El Niño?",
            "Nagbabawas ba ng ulan ang El Niño?",
            "Ano ang ENSO?"
        ]
    },
    "el_nino_signs": {
        "en": [
            "What should I prepare for El Niño?",
            "Does El Niño cause drought?",
            "What is ONI?"
        ],
        "tl": [
            "Ano ang dapat kong ihanda para sa El Niño?",
            "Nagdudulot ba ng drought ang El Niño?",
            "Ano ang ONI?"
        ]
    },
    "el_nino_preparation": {
        "en": [
            "How can I conserve water?",
            "How should I store water safely?",
            "What should I include in an emergency kit?"
        ],
        "tl": [
            "Paano makakatipid ng tubig?",
            "Paano mag-imbak ng tubig nang ligtas?",
            "Ano ang dapat isama sa emergency kit?"
        ]
    },
    "enso_difference": {
        "en": [
            "What is ONI?",
            "What is ENSO?",
            "What happens during El Niño in Padre Garcia?"
        ],
        "tl": [
            "Ano ang ONI?",
            "Ano ang ENSO?",
            "Anong nangyayari tuwing El Niño sa Padre Garcia?"
        ]
    },
    "enso_definition": {
        "en": [
            "What is the difference between El Niño and La Niña?",
            "What is ONI?",
            "How does ONI help monitor El Niño?"
        ],
        "tl": [
            "Ano ang pagkakaiba ng El Niño at La Niña?",
            "Ano ang ONI?",
            "Paano tinutulungan ng ONI na subaybayan ang El Niño?"
        ]
    },
    "la_nina": {
        "en": [
            "What is the difference between El Niño and La Niña?",
            "How long does La Niña last?",
            "What is ENSO?"
        ],
        "tl": [
            "Ano ang pagkakaiba ng El Niño at La Niña?",
            "Gaano katagal ang La Niña?",
            "Ano ang ENSO?"
        ]
    },

    # Water & Drought
    "water_conservation": {
        "en": [
            "How should I store water safely?",
            "Where to report water shortages?",
            "How can farmers prepare for El Niño?"
        ],
        "tl": [
            "Paano mag-imbak ng tubig nang ligtas?",
            "Saan mag-ulat ng kakulangan ng tubig?",
            "Paano maghanda ang mga magsasaka?"
        ]
    },
    "water_storage": {
        "en": [
            "How can I conserve water?",
            "Where to report water shortages?",
            "What should I do during water shortages?"
        ],
        "tl": [
            "Paano makakatipid ng tubig?",
            "Saan mag-ulat ng kakulangan ng tubig?",
            "Ano ang dapat gawin sa panahon ng water shortage?"
        ]
    },
    "water_shortage_action": {
        "en": [
            "How should I store water safely?",
            "What is the MDRRMO hotline number?",
            "How can I conserve water?"
        ],
        "tl": [
            "Paano mag-imbak ng tubig nang ligtas?",
            "Ano ang MDRRMO hotline number?",
            "Paano makatipid ng tubig?"
        ]
    },
    "drought_definition": {
        "en": [
            "Does El Niño cause drought?",
            "How does El Niño affect water supply?",
            "What should farmers do during drought?"
        ],
        "tl": [
            "Nagdudulot ba ng drought ang El Niño?",
            "Paano nakakaapekto ang El Niño sa water supply?",
            "Ano ang dapat gawin ng mga magsasaka sa drought?"
        ]
    },

    # Farming & Agriculture
    "el_nino_agriculture": {
        "en": [
            "Where can farmers ask for assistance?",
            "How can I protect plants and crops?",
            "What should farmers do during drought?"
        ],
        "tl": [
            "Saan humingi ng tulong ang mga magsasaka?",
            "Paano protektahan ang mga pananim?",
            "Ano ang dapat gawin ng magsasaka sa drought?"
        ]
    },
    "farmers_drought": {
        "en": [
            "Where can farmers ask for assistance?",
            "How can I conserve water?",
            "How can I protect plants and crops?"
        ],
        "tl": [
            "Saan humingi ng tulong ang mga magsasaka?",
            "Paano makatipid ng tubig?",
            "Paano protektahan ang mga pananim?"
        ]
    },
    "protect_crops": {
        "en": [
            "What should farmers do during drought?",
            "Where can farmers ask for assistance?",
            "How can I conserve water?"
        ],
        "tl": [
            "Ano ang dapat gawin ng mga magsasaka sa drought?",
            "Saan humingi ng tulong ang mga magsasaka?",
            "Paano makatipid ng tubig?"
        ]
    },
    "farmer_assistance": {
        "en": [
            "What should farmers do during drought?",
            "How can I protect plants and crops?",
            "What local government services are available during El Niño?"
        ],
        "tl": [
            "Ano ang dapat gawin ng mga magsasaka sa drought?",
            "Paano protektahan ang mga pananim?",
            "Anong mga serbisyo ng local government ang available sa El Niño?"
        ]
    },

    # Emergency & Contacts
    "emergency_contacts": {
        "en": [
            "What is the MDRRMO hotline number?",
            "How can I contact the RHU?",
            "Where can I ask for assistance during El Niño?"
        ],
        "tl": [
            "Ano ang MDRRMO hotline number?",
            "Paano makipag-ugnayan sa RHU?",
            "Saan humingi ng tulong sa panahon ng El Niño?"
        ]
    },
    "mdrrmo_contact": {
        "en": [
            "How can I contact the RHU?",
            "Where to report water shortages?",
            "What should I include in an emergency kit?"
        ],
        "tl": [
            "Paano makipag-ugnayan sa RHU?",
            "Saan mag-ulat ng kakulangan ng tubig?",
            "Ano ang dapat isama sa emergency kit?"
        ]
    },
    "rhu_contact": {
        "en": [
            "When should I seek medical help during extreme heat?",
            "What is the MDRRMO hotline number?",
            "What are the signs of heat exhaustion?"
        ],
        "tl": [
            "Kailan dapat humingi ng medikal na tulong sa matinding init?",
            "Ano ang MDRRMO hotline number?",
            "Ano ang mga senyales ng heat exhaustion?"
        ]
    },

    # Data & Comparisons
    "year_summary": {
        "en": [
            "Compare 2015 and 2020",
            "Predict rain for May",
            "What was the wettest month on record?"
        ],
        "tl": [
            "Ihambing ang 2015 at 2020",
            "Hulaan ang ulan para sa May",
            "Ano ang pinakamalakas na ulan sa rekord?"
        ]
    },
    "month_comparison": {
        "en": [
            "Compare 2015 and 2020",
            "Predict rain for May",
            "What was the wettest month on record?"
        ],
        "tl": [
            "Ihambing ang 2015 at 2020",
            "Hulaan ang ulan para sa May",
            "Ano ang pinakamalakas na ulan sa rekord?"
        ]
    },
    "rain_records": {
        "en": [
            "What was the hottest month on record?",
            "Predict rain for May",
            "Compare 2015 and 2020"
        ],
        "tl": [
            "Ano ang pinakamainit na buwan sa rekord?",
            "Hulaan ang ulan para sa May",
            "Ihambing ang 2015 at 2020"
        ]
    },
    "temp_records": {
        "en": [
            "What was the wettest month on record?",
            "What are the signs of heat exhaustion?",
            "Predict rain for May"
        ],
        "tl": [
            "Ano ang pinakamalakas na ulan sa rekord?",
            "Ano ang mga senyales ng heat exhaustion?",
            "Hulaan ang ulan para sa May"
        ]
    },
    "padre_garcia_info": {
        "en": [
            "What happens during El Niño in Padre Garcia?",
            "Where can farmers ask for assistance?",
            "Who should I contact during an emergency?"
        ],
        "tl": [
            "Anong nangyayari tuwing El Niño sa Padre Garcia?",
            "Saan humingi ng tulong ang mga magsasaka?",
            "Sino ang dapat tawagan sa emergency?"
        ]
    },
    # Health & First Aid Emergency Topics
    "heatstroke_emergency": {
        "en": [
            "Who should I contact during an emergency?",
            "What is the difference between heat exhaustion and heatstroke?",
            "How can I contact the RHU?"
        ],
        "tl": [
            "Sino ang dapat tawagan sa emergency?",
            "Ano ang pagkakaiba ng heat exhaustion at heatstroke?",
            "Paano makipag-ugnayan sa RHU?"
        ]
    },
    "heat_exhaustion_action": {
        "en": [
            "What should I do if someone has heatstroke?",
            "How can I prevent dehydration?",
            "When should I seek medical help during extreme heat?"
        ],
        "tl": [
            "Ano ang dapat gawin kung may heatstroke?",
            "Paano maiwasan ang dehydration?",
            "Kailan dapat humingi ng tulong medikal sa matinding init?"
        ]
    },
    "dehydration_first_aid": {
        "en": [
            "How can I prevent dehydration?",
            "What are the signs of heat exhaustion?",
            "How should I store water safely?"
        ],
        "tl": [
            "Paano maiwasan ang dehydration?",
            "Ano ang mga senyales ng heat exhaustion?",
            "Paano mag-imbak ng tubig nang ligtas?"
        ]
    }
}


def is_already_asked(candidate, user_message):
    """Returns True if the candidate question is basically what the user just asked."""
    c_words = set(re.findall(r'\b[a-z0-9]{3,}\b', candidate.lower()))
    u_words = set(re.findall(r'\b[a-z0-9]{3,}\b', user_message.lower()))
    
    # Exclude common filler words from the overlap check
    ignore_words = {"what", "when", "where", "which", "does", "this", "that", "para", "kung", "ano", "anong", "bakit", "paano"}
    c_words -= ignore_words
    u_words -= ignore_words

    if not c_words or not u_words:
        return False

    overlap = len(c_words.intersection(u_words))
    # If 50% or more of key words match, consider it already asked
    return (overlap / len(c_words)) >= 0.5


def generate_related_questions(topic, user_message, reply_text, is_tl=False):
    """
    Finds 2–3 fresh follow-up questions based on the active topic,
    automatically filtering out any question the user already asked.
    """
    lang_key = "tl" if is_tl else "en"
    candidates = []

    # 1. Match by topic
    if topic and topic in TOPIC_SUGGESTIONS:
        candidates = TOPIC_SUGGESTIONS[topic].get(lang_key, [])

    # 2. Keyword fallback if topic has no registered suggestions
    if not candidates:
        user_clean = user_message.lower()
        if any(w in user_clean for w in ["model", "predict", "forecast", "hulaan"]):
            candidates = [
                "How accurate is the rainfall prediction?" if not is_tl else "Gaano katumpak ang rainfall prediction?",
                "What data is used for prediction?" if not is_tl else "Anong data ang ginagamit para sa prediction?",
                "Predict rain for May" if not is_tl else "Hulaan ang ulan para sa May"
            ]
        elif any(w in user_clean for w in ["exhaustion", "stroke", "heat", "init"]):
            candidates = [
                "What should I do if someone experiences heat exhaustion?" if not is_tl else "Ano ang dapat gawin kung may heat exhaustion?",
                "How can I prevent dehydration?" if not is_tl else "Paano maiwasan ang dehydration?",
                "When should I seek medical help during extreme heat?" if not is_tl else "Kailan dapat humingi ng tulong medikal sa matinding init?"
            ]
        elif any(w in user_clean for w in ["water", "tubig"]):
            candidates = [
                "How can I conserve water?" if not is_tl else "Paano makakatipid ng tubig?",
                "How should I store water safely?" if not is_tl else "Paano mag-imbak ng tubig nang ligtas?",
                "Where to report water shortages?" if not is_tl else "Saan mag-ulat ng kakulangan ng tubig?"
            ]

    # Don't show suggestion chips on greetings / farewells
    clean_msg = user_message.lower().strip()
    if any(clean_msg == w for w in ["hi", "hello", "hey", "kamusta", "kumusta", "musta", "salamat", "thanks", "bye", "paalam"]):
        return ""

    # 3. Filter out questions the user already asked!
    fresh_questions = [
        q for q in candidates 
        if not is_already_asked(q, user_message)
    ]

    if not fresh_questions:
        return ""

    label = "Mga iminungkahing tanong:" if is_tl else "Suggested questions:"

    # Build clean single-line buttons
    btn_html_list = []
    for q in fresh_questions[:3]:
        safe_q = q.replace("'", "\\'")
        btn_html_list.append(
            f'<button onclick="sendQuickPrompt(\'{safe_q}\')" '
            f'class="text-xs px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-brand-50 hover:text-brand-600 hover:border-brand-300 '
            f'dark:bg-slate-800/80 dark:hover:bg-slate-800 dark:text-slate-300 dark:hover:text-brand-400 '
            f'border border-slate-200/80 dark:border-slate-700/60 transition-colors text-left flex items-center gap-1.5">'
            f'<span>{q}</span><i class="fa-solid fa-arrow-right text-[9px] opacity-60"></i></button>'
        )

    chips_container = (
        f'<div class="mt-3.5 pt-2.5 border-t border-slate-200/70 dark:border-slate-800/80">'
        f'<div class="text-[11px] font-medium text-slate-400 dark:text-slate-500 mb-1.5 flex items-center gap-1">'
        f'<i class="fa-regular fa-compass text-[10px]"></i><span>{label}</span></div>'
        f'<div class="flex flex-wrap gap-1.5">{"".join(btn_html_list)}</div></div>'
    )

    return chips_container

# ==========================================
# MAIN ROUTER
# ==========================================

def chatbot_event_router(user_input):
    clean_input = user_input.lower().strip()
    clean_input = clean_input.replace('ñ', 'n').replace('é', 'e').replace('í', 'i')

    # Detect language (Tagalog or English)
    conversation_state["lang"] = detect_language(clean_input)

    # Check if user specifically mentions Padre Garcia
    padre_garcia_only = any(k in clean_input for k in ["padre garcia", "batangas only", "only padre garcia", "local only", "sa padre garcia", "sa batangas"])

    # Check if user is in station mode (Padre Garcia only)
    # Technical information handler — before identity so "what are your..." doesn't hit identity first
    # ==========================================
    # STATION COMPARISON MODE TOGGLE
    # ==========================================
    if any(k in clean_input for k in [
        "turn on station comparison", "enable station comparison", "station comparison mode on",
        "turn on station mode", "enable station mode", "compare all stations", "3 stations mode",
        "i-on ang station comparison", "buksan ang station comparison", "all stations mode on"
    ]):
        conversation_state["station_mode"] = True
        conversation_state["last_topic"] = "station_mode_toggle"
        return L(
            "🌐 **Station Comparison Mode is now ON!**\n\n"
            "Responses for climate records and year comparisons will now include all **3 monitored stations**:\n"
            "1. 📍 **Padre Garcia, Batangas** (Local station)\n"
            "2. 📍 **Quezon / Bicol Region** (Regional reference)\n"
            "3. 📍 **Basco, Batanes** (Northern reference)\n\n"
            "💡 *Try asking: 'Compare 2015 and 2020' or 'What happened during El Niño?' to see data across all stations.*\n"
            "*(To return to local only, say: 'Turn off station comparison mode')*",

            "🌐 **Naka-ON na ang Station Comparison Mode!**\n\n"
            "Ang mga susunod na ulat at paghahambing ng klima ay magsasama na ng **tatlong monitored stations**:\n"
            "1. 📍 **Padre Garcia, Batangas** (Lokal na istasyon)\n"
            "2. 📍 **Quezon / Bicol Region** (Rehiyonal na istasyon)\n"
            "3. 📍 **Basco, Batanes** (Hilagang istasyon)\n\n"
            "💡 *Subukan itanong: 'Ihambing ang 2015 at 2020' upang makita ang datos sa tatlong lugar.*\n"
            "*(Para bumalik sa Padre Garcia lamang, sabihin: 'I-off ang station comparison mode')*"
        )

    if any(k in clean_input for k in [
        "turn off station comparison", "disable station comparison", "station comparison mode off",
        "turn off station mode", "disable station mode", "padre garcia only", "local only",
        "i-off ang station comparison", "isara ang station comparison"
    ]):
        conversation_state["station_mode"] = False
        conversation_state["last_topic"] = "station_mode_toggle"
        return L(
            "📍 **Station Comparison Mode is now OFF.**\n\n"
            "The chatbot has returned to **Padre Garcia, Batangas** localized mode. All weather summaries and data will focus exclusively on your local community.",
            "📍 **Naka-OFF na ang Station Comparison Mode.**\n\n"
            "Nagbalik na ang chatbot sa **Padre Garcia, Batangas** mode lamang. Ang lahat ng ulat ng panahon ay nakatuon na muli sa inyong lokal na komunidad."
        )
    
    if any(k in clean_input for k in [
        "technical information", "tech stack", "tech info", "what is your backend",
        "what is your frontend", "what is your stack", "your backend", "your frontend",
        "how were you built", "how are you built", "how was this built", "what technology",
        "what technologies", "what tools", "what programming language", "what language",
        "what framework", "what database", "what model", "your architecture",
        "system architecture", "technical details", "technical specs",
        "ano ang iyong backend", "ano ang iyong frontend", "paano ka ginawa",
        "anong technology", "anong framework", "anong database", "anong language",
        "technical info", "impormasyon tungkol sa sistema",
    ]):
        return L(
            "⚙️ **ReadyKomunidad — Technical Information**\n\n"
            "Here's a breakdown of the tools and technologies powering this chatbot:\n\n"
            "🐍 **Backend**\n"
            "• **Language:** Python 3.11\n"
            "• **Web Framework:** Flask — handles the server, routes, and API endpoints\n"
            "• **Chat API:** `/api/chat` (POST) — receives user messages, returns bot replies\n"
            "• **History API:** `/api/history` (GET) — restores conversation on page refresh\n\n"
            "🎨 **Frontend**\n"
            "• **Markup:** HTML5 (single-file — `readykomunidad.html`)\n"
            "• **Styling:** Tailwind CSS (via CDN) — utility-first responsive design\n"
            "• **Icons:** Font Awesome 6\n"
            "• **Fonts:** Inter (Google Fonts)\n"
            "• **JavaScript:** Vanilla JS — handles chat rendering, dark mode, session management, and the scroll-to-bottom button\n\n"
            "🗄️ **Database**\n"
            "• **SQLite** (via Python's built-in `sqlite3`) — stores chat logs and a climate knowledge base\n"
            "• Two tables: `chatlogs` (conversation history) and `knowledge_base` (extensible Q&A entries)\n\n"
            "🤖 **Machine Learning**\n"
            "• **Model:** Random Forest Regressor (`scikit-learn`)\n"
            "• **Task:** Monthly rainfall prediction for Padre Garcia, Batangas\n"
            "• **Features:** Month, temperature, regional rainfall, ONI value\n"
            "• **Saved as:** `climate_rain_model.pkl` (via `joblib`)\n\n"
            "📊 **Data**\n"
            "• **Libraries:** pandas (data processing), scikit-learn (ML model)\n"
            "• **Sources:** Open-Meteo weather data (3 stations), ENSO/ONI dataset\n"
            "• **Coverage:** 2010–2026 | Stations: Padre Garcia, Quezon/Bicol, Basco (Batanes)\n"
            "• **Master dataset:** `chatbot_master_climate_data.csv` (pre-processed)\n\n"
            "🔁 **How it works:**\n"
            "1. User types a message in the browser\n"
            "2. JavaScript sends it to `/api/chat` via a POST request\n"
            "3. Flask routes it through a keyword-based intent router in `web_app.py`\n"
            "4. The router returns a response (from data, ML model, or knowledge base)\n"
            "5. Both the question and reply are saved to SQLite\n"
            "6. The response is rendered back in the chat UI",
            "⚙️ **ReadyKomunidad — Technical Information**\n\n"
            "Narito ang mga tools at technologies na nagpapatakbo sa chatbot na ito:\n\n"
            "🐍 **Backend**\n"
            "• **Language:** Python 3.11\n"
            "• **Web Framework:** Flask — nag-hahandle ng server, routes, at API endpoints\n"
            "• **Chat API:** `/api/chat` (POST) — tumatanggap ng user messages, nagbabalik ng bot replies\n"
            "• **History API:** `/api/history` (GET) — nagri-restore ng conversation sa page refresh\n\n"
            "🎨 **Frontend**\n"
            "• **Markup:** HTML5 (single-file — `readykomunidad.html`)\n"
            "• **Styling:** Tailwind CSS (via CDN) — utility-first responsive design\n"
            "• **Icons:** Font Awesome 6\n"
            "• **Fonts:** Inter (Google Fonts)\n"
            "• **JavaScript:** Vanilla JS — nag-hahandle ng chat rendering, dark mode, session management, at scroll-to-bottom button\n\n"
            "🗄️ **Database**\n"
            "• **SQLite** (gamit ang built-in na `sqlite3` ng Python) — nag-iimbak ng chat logs at climate knowledge base\n"
            "• Dalawang tables: `chatlogs` (conversation history) at `knowledge_base` (extensible Q&A entries)\n\n"
            "🤖 **Machine Learning**\n"
            "• **Model:** Random Forest Regressor (`scikit-learn`)\n"
            "• **Task:** Monthly rainfall prediction para sa Padre Garcia, Batangas\n"
            "• **Features:** Month, temperature, regional rainfall, ONI value\n"
            "• **Saved as:** `climate_rain_model.pkl` (via `joblib`)\n\n"
            "📊 **Data**\n"
            "• **Libraries:** pandas (data processing), scikit-learn (ML model)\n"
            "• **Sources:** Open-Meteo weather data (3 stations), ENSO/ONI dataset\n"
            "• **Coverage:** 2010–2026 | Stations: Padre Garcia, Quezon/Bicol, Basco (Batanes)\n"
            "• **Master dataset:** `chatbot_master_climate_data.csv` (pre-processed)\n\n"
            "🔁 **Paano gumagana:**\n"
            "1. Nagta-type ang user ng message sa browser\n"
            "2. Pinapadala ito ng JavaScript sa `/api/chat` via POST request\n"
            "3. Irino-route ito ng Flask sa keyword-based intent router sa `web_app.py`\n"
            "4. Nagbabalik ang router ng response (mula sa data, ML model, o knowledge base)\n"
            "5. Nise-save ang tanong at sagot sa SQLite\n"
            "6. Nire-render ang response pabalik sa chat UI"
        )

    # 0. ReadyKomunidad-related questions
    if any(k in clean_input for k in ["what is readykomunidad", "ano ang readykomunidad", "what is this chatbot", "ano itong chatbot", "about this chatbot", "about readykomunidad", "tungkol sa readykomunidad", "ano ito", "pano ito", "who are you", "sino ka", "ano ka", "introduce yourself", "ipakilala mo ang sarili", "readykomunidad", "ReadyKomunidad", "sinoka","anong pangalan mo", "anong pangalan mo", "ano ka", "what are you"]):
        return L(
            "🤖 Hi, I am your ReadyKomunidad! \n\n"
            "I'm a bilingual climate awareness chatbot designed to help the residents of **Padre Garcia, Batangas** "
            "to stay informed about climate phenomena like El Niño, its local impacts, and how to prepare for it.\n\n"
            "Do you want to ask anything?",
            
            "🤖 Mabuhay! Ako si ReadyKomunidad! \n\n"
            "Ako ay isang bilingual climate awareness chatbot, na dinisenyo upang tulungan "
            "ang mga residents ng **Padre Garcia, Batangas** na mapanatiling informed tungkol sa mga climate phenomena "
            "tulad ng El Niño, ang local impacts nito, at kung paano maghanda.\n\n"
            "May nais ka bang malaman?"
            
        )

    # Technical information handler
    if any(k in clean_input for k in ["who made", "who created", "who built", "who developed", "sino gumawa", "sino ang gumawa", "sino ang nagawa"]):
        return L(
            "👥 **Who Developed ReadyKomunidad?**\n"
            "ReadyKomunidad was made by **Group 2** as part of their **CSE 103 / CS 312** course requirement.\n\n"
            "The project aims to help residents of **Padre Garcia, Batangas** understand El Niño, its local effects, and the necessarry preparedness measures.",
            "👥 **Sino ang Gumawa ng ReadyKomunidad?**\n"
            "Ang ReadyKomunidad ay ginawa ng **Group 2** bilang bahagi ng kanilang course requirement sa **CSE 103 / CS 312**.\n\n"
            "Layunin ng proyekto na tulungan ang mga residente ng **Padre Garcia, Batangas** na maunawaan ang El Niño, ang mga eoekto nito, at mga kinakailangang hakbang para makapaghanda rito."
        )

    # NEW: What is your purpose (specific handler) - MUST come before "what can you do"
    if any(k in clean_input for k in [
        "what is your purpose", "whats your purpose", "what's your purpose",
        "ano ang iyong layunin", "what is the purpose", "ano ang layunin",
        "purpose of this chatbot", "bakit ka ginawa", "why were you created", "why were you made",
        "ano ang dahilan", "purpose mo", "layunin mo"
    ]):
        return L(
            "🎯 **Purpose of ReadyKomunidad**\n\n"
            "My purpose is to **help the residents of Padre Garcia, Batangas** understand and prepare for **El Niño** and its effects on their community.\n\n"
            "**Specifically, I aim to:**\n"
            "• **Educate** — Explain what El Niño is and how it affects local weather, agriculture, and daily life\n"
            "• **Inform** — Provide historical climate data so you can see patterns and trends\n"
            "• **Predict** — Use machine learning to forecast rainfall based on past data\n"
            "• **Prepare** — Share practical tips for water conservation, crop protection, and health safety\n"
            "• **Empower** — Make climate information accessible in both English and Filipino\n\n"
            "💡 *The goal is simple: help the community stay informed and prepared before El Niño strikes.*",
            "🎯 **Layunin ng ReadyKomunidad**\n\n"
            "Ang layunin ko ay **tulungan ang mga residents ng Padre Garcia, Batangas** na maunawaan at maghanda para sa **El Niño** at ang mga epekto nito sa kanilang komunidad.\n\n"
            "**Sa partikular, layunin ko na:**\n"
            "• **Magturo** — Ipaliwanag kung ano ang El Niño at kung paano ito nakakaapekto sa local weather, agrikultura, at pang-araw-araw na buhay\n"
            "• **Magbigay ng impormasyon** — Magbigay ng historical climate data para makita mo ang mga patterns at trends\n"
            "• **Hulaan** — Gumamit ng machine learning para tantiyahin ang rainfall base sa past data\n"
            "• **Maghanda** — Magbahagi ng praktikal na tips para sa water conservation, crop protection, at health safety\n"
            "• **Magbigay ng kapangyarihan** — Gawing accessible ang climate information sa English at Filipino\n\n"
            "💡 *Ang layunin ko ay simple: tulungan ang komunidad na manatiling informed at handa bago dumating ang El Niño.*"
        )

    if any(k in clean_input for k in ["what can you do", "what can this do", "ano magagawa mo", "what are you for", "anong kaya mo", "how can you help", "what can readykomunidad do?", "what can ReadyKomunidad do?", "ano ang magagawa", "anong magagawa", "what topics can you answer", "what can i ask you", "what topics do you cover", "what questions can you answer", "anong mga tanong", "anong paksa", "what subjects", "what can you help me with", "anong maitutulong mo", "ano ang mga kaya mong gawin", "paano mo ako matutulungan"]):
        return L(
            "🌦️ **What Can ReadyKomunidad Do?**\n\n"
            "• **Understanding El Niño** — What El Nino is, its possible effects, and how to prepare for it.\n"
            "• **Climate History** — Explore past weather data per year or month (2010–2026)\n"
            "• **Year Comparisons** — Compare recorded rainfall and tempeatures between years e.g. *'Compare 2015 and 2020'*\n"
            "• **Rainfall & Temperature Records** —Find recorded all-time highs and lows and other available weather information across supported stations\n"
            "• **ML Rainfall Predictions** — Ask for rainfall predictions for a particular month/year if supported e.g. *'Predict rain for May'*\n\n"
            "**My goal is to help residents make informed decisions and become better prepared for the possible effects of El Niño.**\n\n"
            "What else would you like to know? 😊",
            "🌦️ **Ano ang Magagawa ng ReadyKomunidad?**\n\n"
            "• **Pag-unawa sa El Niño** — Ano ang El Niño, ang mga effects nito, at kung paano maghanda.\n"
            "• **Climate History** — Suriin ang weather data bawat taon o buwan (2010–2026)\n"
            "• **Pagkokumpara ng Taon** — Ihambing ang naitala na rainfall at temperature sa pagitan ng mga taon hal. *'Ihambing ang 2015 at 2020'*\n"
            "• **Rainfall at Temperature Records** — Hanapin ang pinakamataas at pinakamababang naitala at iba pang weather information sa lahat ng stations\n"
            "• **ML Rainfall Predictions** — Humingi ng rainfall predictions para sa isang buwan/taon hal. *'Hulaan ang ulan para sa May'*\n\n"
            "**Layunin ko na tulungan ang mga residente upang maging handa para sa posibleng epekto ng El Niño.**"
        )

    if any(k in clean_input for k in ["ai ka ba", "are you ai", "are you an ai", "are you a bot", "are you a chatbot", "are you a robot"]):
            return L(
                "🤖 Yes! I am an **AI chatbot!** called **ReadyKomunidad**\n\n"
                "I am designed to provide information and assistance about El Niño, preparedness for extreme heat and water shortages, and community safety. 🌱"
                "What else would you like to know? 😊",
                "🤖 Oo! Ako ay isang **AI chatbot!** na tinatawag na **ReadyKomunidad**\n\n"
                "Ako ay dinisenyo upang magbigay ng impormasyon at tulong tungkol sa El Niño, paghahanda para sa matinding init at kakulangan ng tubig, at kaligtasan ng komunidad. 🌱"
                "Ano pa ang nais mong malaman? 😊"
                )
    
    if any(k in clean_input for k in ["who is this for", "para kanino", "target", "sino ang gagamit", "intended for", "para sa sino", "sino ang para"]):
        return L(
            "🏘️ **Who is ReadyKomunidad For?**\n"
            "This chatbot is made for the **residents of Padre Garcia, Batangas** — farmers, households, "
            "and community members who want to understand how El Niño affects their local area.\n\n"
            "The goal is to help the community **prepare and respond** to climate events before they happen.",
            "🏘️ **Para Sino ang ReadyKomunidad?**\n"
            "Ang chatbot na ito ay para sa mga **residents ng Padre Garcia, Batangas** — farmers, households, "
            "at mga community members na nais maunawaan kung paano nakakaapekto ang El Niño sa kanilang lugar.\n\n"
            "Layunin nitong tulungan ang community na **maghanda at tumugon** sa mga climate events bago ito mangyari."
        )
# ==========================================
    # PADRE GARCIA OVERVIEW HANDLER
    # ==========================================
    if any(k in clean_input for k in [
        "what is padre garcia", "about padre garcia", "tell me about padre garcia",
        "information about padre garcia", "ano ang padre garcia", "tungkol sa padre garcia",
        "ipakilala ang padre garcia", "kwento ng padre garcia", "history of padre garcia"
    ]) or clean_input in ["padre garcia", "padre garcia batangas"]:
        conversation_state["last_topic"] = "padre_garcia_info"
        return L(
            "🌱 **Welcome to Padre Garcia, Batangas!**\n\n"
            "![Padre Garcia Municipal Arch](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/LRT%2CBatangasjf9633_15.JPG/640px-LRT%2CBatangasjf9633_15.JPG)\n"
            "![Padre Garcia Church](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Padre_Garcia%2C_Batangas_Church.jpg/640px-Padre_Garcia%2C_Batangas_Church.jpg)\n\n"
            "**Padre Garcia** is a thriving municipality located in the eastern portion of Batangas, Philippines. "
            "It is widely known for its strong agricultural roots, hardworking community, and rich local heritage.\n\n"
            "🐄 **Cattle Trading Capital of the Philippines**\n"
            "The town earned this title because of its famous **Livestock Auction Market** — the largest of its kind in the country. "
            "Every week, livestock raisers, traders, and buyers from Luzon, Visayas, and Mindanao gather in Padre Garcia to trade cattle, hogs, and goats.\n\n"
            "📋 **Quick Facts About Padre Garcia:**\n"
            "• **Province:** Batangas\n"
            "• **Region:** CALABARZON (Region IV-A)\n"
            "• **Barangays:** 18 rural and urban barangays\n"
            "• **Municipal Center:** Barangay Poblacion\n"
            "• **Primary Livelihoods:** Livestock trading, crop farming, and agribusiness\n\n"
            "🎉 **What Makes Padre Garcia Special?**\n"
            "• **Livestock Industry:** A vital driver of the local economy providing income for thousands of families.\n"
            "• **Kabakahan Festival:** Celebrated annually in early December, this vibrant festival honors the cattle industry with street dancing, parades, and cultural contests.\n"
            "• **Agricultural Heart:** Expansive farmlands yielding rice, corn, sugarcane, vegetables, and fruit crops.\n\n"
            "🤝 **Why is Padre Garcia Important to ReadyKomunidad?**\n"
            "Padre Garcia is the primary home and community that **ReadyKomunidad** was created to serve.\n\n"
            "Because Padre Garcia's economy depends heavily on farming and livestock, climate shifts like **El Niño** present serious local risks:\n"
            "• **Water Supply:** Depleted wells and irrigation canals during dry spells.\n"
            "• **Crop Stress:** Reduced rice and corn yields from heat and lack of moisture.\n"
            "• **Livestock Health:** Heat exhaustion and dehydration among farm animals.\n"
            "• **Community Health:** Increased risk of heat stroke among farmers and outdoor workers.\n\n"
            "💡 *ReadyKomunidad aims to empower every resident of Padre Garcia with timely climate data, AI rainfall predictions, and early preparedness tips.*",

            "🌱 **Maligayang Pagdating sa Padre Garcia, Batangas!**\n\n"
            "![Padre Garcia Municipal Arch](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/LRT%2CBatangasjf9633_15.JPG/640px-LRT%2CBatangasjf9633_15.JPG)\n"
            "![Padre Garcia Church](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Padre_Garcia%2C_Batangas_Church.jpg/640px-Padre_Garcia%2C_Batangas_Church.jpg)\n\n"
            "Ang **Padre Garcia** ay isang maunlad na bayan na matatagpuan sa silangang bahagi ng Batangas, Pilipinas. "
            "Kilala ito sa masiglang agrikultura, masisipag na mamamayan, at makulay na kasaysayan.\n\n"
            "🐄 **Cattle Trading Capital of the Philippines (Kabisera ng Bakahan)**\n"
            "Ikinararangal ng Padre Garcia ang titulong ito dahil sa tanyag nitong **Livestock Auction Market** — ang pinakamalaking pamilihan ng baka sa buong bansa. "
            "Linggo-linggo, dumadayo rito ang mga magsasaka, mangangalakal, at mamimili mula sa iba't ibang panig ng Luzon, Visayas, at Mindanao upang magbenta at bumili ng baka, baboy, at kambing.\n\n"
            "📋 **Mabilis na Impormasyon Ukol sa Padre Garcia:**\n"
            "• **Lalawigan:** Batangas\n"
            "• **Rehiyon:** CALABARZON (Region IV-A)\n"
            "• **Bilang ng Barangay:** 18 barangay\n"
            "• **Sentro ng Bayan:** Barangay Poblacion\n"
            "• **Pangunahing Kabuhayan:** Pagbabakahan, pagtatanim, at kalakalan\n\n"
            "🎉 **Bakit Espesyal ang Padre Garcia?**\n"
            "• **Industriya ng Pagbabakahan:** Pangunahing pinagkukunan ng kita ng libu-libong pamilya at negosyo.\n"
            "• **Kabakahan Festival:** Ipinagdiriwang tuwing simula ng Disyembre upang pasalamatan at ipagmalaki ang industriya ng baka sa pamamagitan ng street dancing at parada.\n"
            "• **Masaganang Bukirin:** Malalawak na sakahan ng palay, mais, tubo, at mga gulay.\n\n"
            "🤝 **Bakit Mahalaga ang Padre Garcia sa ReadyKomunidad?**\n"
            "Ang Padre Garcia ang pangunahing komunidad na nilalayong paglingkuran ng **ReadyKomunidad**.\n\n"
            "Dahil ang kabuhayan sa bayan ay nakasalalay sa agrikultura at mga alagang hayop, malaki ang epekto ng **El Niño** sa mga sumusunod:\n"
            "• **Suplay ng Tubig:** Pagkatuyo ng mga balon at irigasyon sa mahabang tag-araw.\n"
            "• **Pananim:** Pagkalanta ng palay at mais dahil sa matinding sikat ng araw.\n"
            "• **Alagang Hayop:** Panganib ng heat stress at kakulangan sa inumin ng mga baka.\n"
            "• **Kalusugan:** Banta ng heat stroke at dehydration sa mga magsasaka at manggagawa.\n\n"
            "💡 *Layunin ng ReadyKomunidad na maghatid ng napapanahong impormasyon, rainfall predictions, at gabay sa paghahanda para sa bawat Garceño.*"
        )


    # NEW: Scope and Limitations
    if any(k in clean_input for k in [
        "scope and limitation", "scope and limitations", "what are your scope", "what are your limitations",
        "scope", "limitations", "what can you not do", "what cant you do", "what can't you do",
        "ano ang limitasyon", "ano ang hindi mo kayang gawin", "ano ang saklaw", "scope mo",
        "limitation mo", "hindi mo kaya", "ano ang di mo kaya"
    ]):
        return L(
            "📋 **Scope and Limitations of ReadyKomunidad**\n\n"
            "**✅ What I CAN do:**\n"
            "• Explain El Niño, La Niña, and ENSO phenomena\n"
            "• Provide historical climate data for Padre Garcia, Batangas (2010–2026)\n"
            "• Compare rainfall and temperature across different years and months\n"
            "• Predict rainfall using machine learning based on historical patterns\n"
            "• Give preparedness tips and health safety advice during El Niño\n"
            "• Answer questions in both English and Filipino\n\n"
            "**❌ What I CANNOT do:**\n"
            "• Provide real-time weather forecasts (I use historical data, not live weather)\n"
            "• Give medical diagnoses or professional health advice\n"
            "• Predict exact future El Niño/La Niña events with certainty\n"
            "• Cover areas outside Padre Garcia, Batangas in detail (my focus is local)\n"
            "• Answer questions unrelated to climate, weather, or El Niño\n\n"
            "💡 *For live weather updates, check PAGASA. For medical emergencies, contact healthcare professionals.*",
            "📋 **Scope at Limitations ng ReadyKomunidad**\n\n"
            "**✅ Ano ang KAYA ko:**\n"
            "• Ipaliwanag ang El Niño, La Niña, at ENSO phenomena\n"
            "• Magbigay ng historical climate data para sa Padre Garcia, Batangas (2010–2026)\n"
            "• Ihambing ang rainfall at temperature sa iba't ibang taon at buwan\n"
            "• Hulaan ang rainfall gamit ang machine learning base sa historical patterns\n"
            "• Magbigay ng preparedness tips at health safety advice sa panahon ng El Niño\n"
            "• Sumagot ng mga tanong sa English at Filipino\n\n"
            "**❌ Ano ang HINDI ko kaya:**\n"
            "• Magbigay ng real-time weather forecasts (gumagamit ako ng historical data, hindi live weather)\n"
            "• Mag-diagnose ng sakit o magbigay ng propesyonal na medical advice\n"
            "• Hulaan nang eksakto ang mga susunod na El Niño/La Niña events\n"
            "• Saklawin ang mga lugar sa labas ng Padre Garcia, Batangas nang detalyado (local focus ko)\n"
            "• Sumagot ng mga tanong na walang kinalaman sa climate, weather, o El Niño\n\n"
            "💡 *Para sa live weather updates, tignan ang PAGASA. Para sa medical emergencies, makipag-ugnayan sa healthcare professionals.*"
        )

    # NEW: Climate data query (generic)
    if any(k in clean_input for k in [
        "climate data", "weather data", "climate information", "weather information",
        "data ng klima", "data ng panahon", "impormasyon ng klima", "climate records",
        "historical data", "historical climate", "historical weather"
    ]) and len(clean_input.split()) <= 4:  # Only trigger for short queries like "climate data"
        return L(
            "📊 **Climate Data Available**\n\n"
            "I have historical climate data for **Padre Garcia, Batangas** covering **2010–2026**.\n\n"
            "**Available data includes:**\n"
            "• Monthly rainfall (mm)\n"
            "• Average temperatures (°C)\n"
            "• Humidity levels (%)\n"
            "• ENSO phase (El Niño, La Niña, Neutral)\n"
            "• ONI (Oceanic Niño Index) values\n\n"
            "**Example questions you can ask:**\n"
            "• *'What was the rainfall in June 2015?'*\n"
            "• *'Compare 2015 and 2020'*\n"
            "• *'What was the hottest month on record?'*\n"
            "• *'Show me climate data for 2018'*\n\n"
            "💡 *Try asking about a specific year or month!*",
            "📊 **Available na Climate Data**\n\n"
            "Mayroon akong historical climate data para sa **Padre Garcia, Batangas** mula **2010–2026**.\n\n"
            "**Available na data:**\n"
            "• Monthly rainfall (mm)\n"
            "• Average temperatures (°C)\n"
            "• Humidity levels (%)\n"
            "• ENSO phase (El Niño, La Niña, Neutral)\n"
            "• ONI (Oceanic Niño Index) values\n\n"
            "**Mga halimbawa ng tanong na pwede mong itanong:**\n"
            "• *'Ano ang rainfall noong June 2015?'*\n"
            "• *'Ihambing ang 2015 at 2020'*\n"
            "• *'Ano ang pinakamainit na buwan sa record?'*\n"
            "• *'Ipakita ang climate data para sa 2018'*\n\n"
            "💡 *Subukan magtanong tungkol sa isang partikular na taon o buwan!*"
        )

    # Full question list handler
    if any(k in clean_input for k in [
        "list all questions", "list of questions", "all possible questions",
        "what questions can i ask", "what can i ask about", "full list of questions",
        "all questions you can answer", "complete list", "list everything",
        "what are all the topics", "all topics", "show me all questions",
        "lahat ng tanong", "listahan ng tanong", "anong lahat ng tanong",
        "lahat ng pwedeng itanong", "lahat ng maaaring itanong",
        "buong listahan", "lahat ng paksa"
    ]):
        return L(
            "📋 **All Questions ReadyKomunidad Can Answer**\n\n"

            "💬 **Greetings & General**\n"
            "• Hi / Hello / Hey / Kumusta\n"
            "• Are you available? / Can I ask you something?\n"
            "• Thank you / Goodbye\n\n"

            "🤖 **About This Chatbot**\n"
            "• What is ReadyKomunidad? / Who are you?\n"
            "• Who made this chatbot?\n"
            "• What can you do? / What topics do you cover?\n"
            "• Who is this chatbot for?\n"
            "• Do you support Filipino?\n"
            "• What is your tech stack? / How were you built?\n\n"

            "🌊 **El Niño**\n"
            "• What is El Niño?\n"
            "• Why is it called El Niño?\n"
            "• What causes El Niño?\n"
            "• How does El Niño work?\n"
            "• How long does El Niño last?\n"
            "• Is El Niño dangerous?\n"
            "• Why does El Niño cause dry conditions?\n"
            "• Does El Niño cause drought?\n"
            "• Does El Niño reduce rainfall?\n"
            "• What are the effects of El Niño?\n"
            "• How does El Niño affect the Philippines?\n"
            "• How does El Niño affect water supply?\n"
            "• How does El Niño affect agriculture and crops?\n"
            "• Can El Niño cause higher temperatures?\n"
            "• How does El Niño affect daily activities?\n"
            "• Does El Niño affect electricity consumption?\n"
            "• Can El Niño affect the local economy?\n"
            "• Can El Niño affect people's health?\n"
            "• What happens during prolonged dry weather?\n"
            "• What happens during El Niño in Padre Garcia?\n"
            "• Does El Niño happen every year?\n"
            "• Why should we know about El Niño?\n\n"

            "🌧️ **La Niña & ENSO**\n"
            "• What is La Niña?\n"
            "• Why is it called La Niña?\n"
            "• What causes La Niña? / How does La Niña happen?\n"
            "• How long does La Niña last?\n"
            "• Is La Niña the opposite of El Niño?\n"
            "• What is ENSO?\n"
            "• What is the difference between El Niño, La Niña, and ENSO?\n\n"

            "📊 **ONI (Oceanic Niño Index)**\n"
            "• What is ONI?\n"
            "• What does a positive ONI value mean?\n"
            "• What does a negative ONI value mean?\n"
            "• How does ONI help monitor El Niño?\n"
            "• What is the relationship between ONI and El Niño?\n"
            "• What is the difference between ONI and ENSO?\n\n"

            "🏥 **Health & Safety**\n"
            "• Can El Niño affect people's health?\n"
            "• What is heatstroke?\n"
            "• What are the signs of heat exhaustion?\n"
            "• What are the signs of dehydration?\n"
            "• How can I prevent dehydration?\n"
            "• What should I do during extreme heat?\n"
            "• What should I do if someone experiences heat exhaustion?\n"
            "• When should I seek medical help during extreme heat?\n\n"

            "🛡️ **Preparedness**\n"
            "• What should I do during El Niño?\n"
            "• How can I stay safe during El Niño?\n"
            "• How can farmers prepare for El Niño?\n"
            "• How can I save / conserve water?\n"
            "• How can I protect my crops during El Niño?\n"
            "• How should I store water safely?\n"
            "• What should I include in an emergency kit?\n\n"

            "📈 **Climate Data (Historical Records)**\n"
            "• What was the weather like in [year]? *(e.g. 2015, 2020)*\n"
            "• What was the rainfall in [month] [year]? *(e.g. June 2015)*\n"
            "• Compare [year1] and [year2] *(e.g. Compare 2015 and 2020)*\n"
            "• Which year had the highest/lowest rainfall?\n"
            "• What was the wettest/driest month on record?\n"
            "• What are the all-time rainfall/temperature records?\n"
            "• What are the 3 monitored weather stations?\n\n"

            "🤖 **Rainfall Prediction (ML)**\n"
            "• What is rainfall prediction?\n"
            "• Can you predict rainfall?\n"
            "• Predict rain for [month] *(e.g. Predict rain for May)*\n"
            "• Predict rainfall for [month] [year] *(e.g. Predict rain for May 2027)*\n"
            "• What ML model is used?\n"
            "• How accurate is the rainfall prediction?\n"
            "• What data is used for prediction?\n"
            "• What is the difference between historical and predicted rainfall?\n\n"

            "🏘️ **Community & LGU**\n"
            "• Where can I ask for assistance during El Niño?\n"
            "• What is the MDRRMO hotline?\n"
            "• How can I contact the RHU?\n"
            "• Where do I report a water shortage?\n"
            "• Where can farmers ask for assistance?\n"
            "• Who to contact during an El Niño-related emergency?\n"
            "• What LGU services are available during El Niño?\n\n"

            "💡 *Tip: You can ask in English or Filipino! For data queries, just include a year (e.g. '2015') or month + year (e.g. 'June 2015') in your question.*",

            "📋 **Lahat ng Tanong na Maaaring Sagutin ng ReadyKomunidad**\n\n"

            "💬 **Pagbati at Pangkalahatan**\n"
            "• Hi / Hello / Kumusta / Musta\n"
            "• Available ka ba? / Pwede bang magtanong?\n"
            "• Salamat / Paalam\n\n"

            "🤖 **Tungkol sa Chatbot**\n"
            "• Ano ang ReadyKomunidad? / Sino ka?\n"
            "• Sino ang gumawa ng chatbot na ito?\n"
            "• Ano ang magagawa mo? / Anong paksa ang alam mo?\n"
            "• Para kanino ang chatbot na ito?\n"
            "• Marunong ka ba mag-Tagalog?\n"
            "• Paano ka ginawa? / Anong technology ang ginamit?\n\n"

            "🌊 **El Niño**\n"
            "• Ano ang El Niño?\n"
            "• Bakit tinatawag na El Niño?\n"
            "• Ano ang sanhi ng El Niño?\n"
            "• Paano gumagana ang El Niño?\n"
            "• Gaano katagal ang El Niño?\n"
            "• Mapanganib ba ang El Niño?\n"
            "• Bakit nagdudulot ng dry conditions ang El Niño?\n"
            "• Nagdudulot ba ng drought ang El Niño?\n"
            "• Nagdudulot ba ng mas kaunting ulan ang El Niño?\n"
            "• Ano ang epekto ng El Niño?\n"
            "• Paano nakakaapekto ang El Niño sa Pilipinas?\n"
            "• Paano nakakaapekto ang El Niño sa suplay ng tubig?\n"
            "• Paano nakakaapekto ang El Niño sa agrikultura at mga pananim?\n"
            "• Nagdudulot ba ng mas mataas na temperatura ang El Niño?\n"
            "• Paano nakakaapekto ang El Niño sa pang-araw-araw na gawain?\n"
            "• Nakakaapekto ba ang El Niño sa paggamit ng kuryente?\n"
            "• Nakakaapekto ba ang El Niño sa lokal na ekonomiya?\n"
            "• Nakakaapekto ba ang El Niño sa kalusugan ng tao?\n"
            "• Ano ang nangyayari kapag matagal na walang ulan?\n"
            "• Ano ang nangyayari sa El Niño sa Padre Garcia?\n"
            "• Nagkakaroon ba ng El Niño tuwing taon?\n"
            "• Bakit mahalaga na malaman ang tungkol sa El Niño?\n\n"

            "🌧️ **La Niña at ENSO**\n"
            "• Ano ang La Niña?\n"
            "• Bakit tinatawag na La Niña?\n"
            "• Ano ang sanhi ng La Niña? / Paano nangyayari ang La Niña?\n"
            "• Gaano katagal ang La Niña?\n"
            "• Kabaligtaran ba ng La Niña ang El Niño?\n"
            "• Ano ang ENSO?\n"
            "• Ano ang pagkakaiba ng El Niño, La Niña, at ENSO?\n\n"

            "📊 **ONI (Oceanic Niño Index)**\n"
            "• Ano ang ONI?\n"
            "• Ano ang ibig sabihin ng positibong ONI?\n"
            "• Ano ang ibig sabihin ng negatibong ONI?\n"
            "• Paano tinutulungan ng ONI na subaybayan ang El Niño?\n"
            "• Ano ang relasyon ng ONI at El Niño?\n"
            "• Ano ang pagkakaiba ng ONI at ENSO?\n\n"

            "🏥 **Kalusugan at Kaligtasan**\n"
            "• Nakakaapekto ba ang El Niño sa kalusugan ng tao?\n"
            "• Ano ang heatstroke?\n"
            "• Ano ang mga senyales ng heat exhaustion?\n"
            "• Ano ang mga senyales ng dehydration?\n"
            "• Paano maiwasan ang dehydration?\n"
            "• Ano ang dapat gawin sa matinding init?\n"
            "• Ano ang dapat gawin kung may nakakaranas ng heat exhaustion?\n"
            "• Kailan dapat humingi ng medikal na tulong sa matinding init?\n\n"

            "🛡️ **Paghahanda**\n"
            "• Ano ang dapat gawin sa panahon ng El Niño?\n"
            "• Paano manatiling ligtas sa panahon ng El Niño?\n"
            "• Paano mahanda ang mga magsasaka para sa El Niño?\n"
            "• Paano makatipid ng tubig?\n"
            "• Paano protektahan ang mga pananim sa panahon ng El Niño?\n"
            "• Paano mag-imbak ng tubig nang ligtas?\n"
            "• Ano ang dapat isama sa emergency kit?\n\n"

            "📈 **Climate Data (Historical Records)**\n"
            "• Kumusta ang panahon noong [taon]? *(hal. 2015, 2020)*\n"
            "• Ano ang ulan noong [buwan] [taon]? *(hal. Hunyo 2015)*\n"
            "• Ihambing ang [taon1] at [taon2] *(hal. Ihambing ang 2015 at 2020)*\n"
            "• Aling taon ang may pinakamataas/pinakamababang ulan?\n"
            "• Kailan ang pinakamalakas/pinakamahinang ulan sa rekord?\n"
            "• Ano ang all-time rainfall/temperature records?\n"
            "• Ano ang tatlong weather stations na nino-monitor?\n\n"

            "🤖 **Rainfall Prediction (ML)**\n"
            "• Ano ang rainfall prediction?\n"
            "• Kaya mo bang hulaan ang ulan?\n"
            "• Hulaan ang ulan para sa [buwan] *(hal. Hulaan ang ulan para sa Mayo)*\n"
            "• Hulaan ang ulan para sa [buwan] [taon] *(hal. Hulaan ang ulan para sa Mayo 2027)*\n"
            "• Anong ML model ang ginagamit?\n"
            "• Gaano katumpak ang rainfall prediction?\n"
            "• Anong data ang ginagamit para sa prediction?\n"
            "• Ano ang pagkakaiba ng historical at predicted rainfall?\n\n"

            "🏘️ **Komunidad at LGU**\n"
            "• Saan dapat humingi ng tulong sa panahon ng El Niño?\n"
            "• Ano ang hotline ng MDRRMO?\n"
            "• Paano makipag-ugnayan sa RHU?\n"
            "• Saan mag-ulat ng kakulangan ng tubig?\n"
            "• Saan humingi ng tulong ang mga magsasaka?\n"
            "• Sino ang dapat tawagan sa panahon ng emergency?\n"
            "• Anong serbisyo ng LGU ang available sa panahon ng El Niño?\n\n"

            "💡 *Tip: Maaaring magtanong sa English o Filipino! Para sa data queries, isama ang taon (hal. '2015') o buwan + taon (hal. 'Hunyo 2015') sa iyong tanong.*"
        )

    if any(k in clean_input for k in ["bilingual", "english and filipino","magtagalog", "magfilipino", "marunong ka ba mag tagalog", "marunong ka ba mag-filipino", "marunong ka ba magtagalog", "tagalog ka ba", "tagalog"]):
        return L(
            "🗣️ **Bilingual Support**\n"
            "Yes! ReadyKomunidad supports both **English and Filipino**.\n\n"
            "For example:\n"
            "• *'Ano ang El Niño?'*\n"
            "• *'Paano nakakaapekto ang La Niña sa Padre Garcia?'*\n"
            "• *'Ihambing ang 2015 at 2020'*",
            "🗣️ **Bilingual Support**\n"
            "Oo! Sinusuportahan ng ReadyKomunidad ang parehong **English at Filipino**.\n\n"
            "Halimbawa:\n"
            "• *'Ano ang El Niño?'*\n"
            "• *'Paano nakakaapekto ang La Niña sa Padre Garcia?'*\n"
            "• *'Ihambing ang 2015 at 2020'*"
        )

    # 1. Greetings
    words_in_input = clean_input.split()
    if any(w in ["hi", "hello", "hey", "kamusta", "good day", "kumusta", "musta", "good morning", "good night","afternoon", "tanghali", "hapon"] for w in words_in_input):
        if conversation_state["lang"] == "tl":
            return random.choice([
                "Kumusta! 👋 Ako si **ReadyKomunidad** — isang chatbot na tumutulong sa mga taga-Padre Garcia, Batangas na maunawaan ang El Niño, ang mga epekto nito, at kung paano maghanda.",
                "Mabuhay! 👋 Ako si **ReadyKomunidad**, isang climate awareness chatbot para sa mga residente ng Padre Garcia, Batangas. Matutulungan kita na maunawaan ang El Niño, ang mga epekto nito sa komunidad, at kung paano maghanda.",
                "Kumusta! 👋 Nandito si **ReadyKomunidad** — ang gabay mo sa pag-unawa sa El Niño para sa mga residente ng Padre Garcia, Batangas. Maaari mong itanong sa akin tungkol sa mga epekto nito, mga paraan ng paghahanda, at lokal na detalye ukol sa klima."
            ])
        return random.choice([
            "Hi there! 👋 I'm **ReadyKomunidad**, a climate awareness chatbot for the residents of Padre Garcia, Batangas. I can help you understand El Niño, its effects on your community, and how to stay prepared. \n\nIs there any concern I can help you with?",
            "Hello! 👋 **ReadyKomunidad** here — your guide to El Niño awareness for Padre Garcia, Batangas. You can ask me about its effects, preparedness tips, and local climate data. \n\nIs there any concern I can help you with?"
        ])

    # 1b. Gratitude / farewells
    if any(k in clean_input for k in ["thank you", "thanks", "thank u", "tyvm", "salamat", "maraming salamat", "salamat po", "thank you po"]) or clean_input.split() == ["ty"] or clean_input.strip() == "ty":
        return L(
            random.choice([
                "You're welcome! 😊 Feel free to ask anything else about El Niño, climate data, or how to prepare.",
                "Glad I could help! 🌤️ If you have more questions about El Niño or the local climate, I'm here.",
                "No problem! 👍 Ask me anything — El Niño effects, rainfall records, predictions, and more.",
            ]),
            random.choice([
                "Walang anuman! 😊 Huwag mag-atubiling magtanong pa tungkol sa El Niño, climate data, o kung paano maghanda.",
                "Natutuwa akong makatulong! 🌤️ Kung may mga katanungan ka pa tungkol sa El Niño o local climate, nandito lang ako.",
                "Sige lang! 👍 Magtanong ka anytime — effects ng El Niño, rainfall records, predictions, at marami pa.",
            ])
        )

    if any(k in clean_input for k in ["bye", "goodbye", "see you", "paalam", "hanggang sa muli"]):
        return L(
            "Goodbye! 👋 Stay safe and climate-aware. Come back anytime you have questions!",
            "Paalam! 👋 Mag-ingat at laging maging handa sa climate events. Bumalik ka anytime!"
        )

    # 1c. Conversational openers — "can I ask?", "may I ask?", etc.
    if any(k in clean_input for k in [
        "can i ask", "can i ask you", "can i ask a question", "can i ask something",
        "may i ask", "may i ask you", "may i ask a question", "may i ask something",
        "i have a question", "i have a question for you", "i want to ask",
        "i want to ask something", "i want to ask you", "question for you",
        "can you answer", "can you help me", "can you answer my question",
        "are you available", "available ka ba", "available ka na ba", "available na ba",
        "are you there", "nandito ka ba", "ready ka ba", "are you ready",
        "maaari ba akong magtanong", "puwede ba akong magtanong", "may tanong ako",
        "pwede bang magtanong", "gusto kong magtanong", "ask kita",
        "tanong ko lang", "tanong lang", "may itanong ako", "pwede ba akong magtanong", "may tatanong ako",
    ]):
        return L(
            random.choice([
                "Of course! 😊 Go ahead and ask — I'm here to help with anything about El Niño, climate data, or preparedness for Padre Garcia.",
                "Sure thing! 👍 What would you like to know? I can answer questions about El Niño, local climate, water conservation, preparedness, and more.",
                "Absolutely! Feel free to ask anything. 🌤️ I'm ReadyKomunidad — your climate awareness chatbot for Padre Garcia, Batangas.",
            ]),
            random.choice([
                "Siyempre! 😊 Itanong mo lang — nandito ako para tumulong sa lahat ng tungkol sa El Niño, climate data, o preparedness para sa Padre Garcia.",
                "Oo naman! 👍 Ano ang gusto mong malaman? Maaari kong sagutin ang mga tanong tungkol sa El Niño, local climate, water conservation, preparedness, at marami pa.",
                "Syempre naman! Huwag kang mahihiyang magtanong. 🌤️ Ako si ReadyKomunidad — ang climate awareness chatbot para sa Padre Garcia, Batangas.",
            ])
        )

    # 1b. Gratitude / farewells
    if any(k in clean_input for k in ["thank you", "thanks", "thank u", "tyvm", "salamat", "maraming salamat", "salamat po", "thank you po"]) or clean_input.split() == ["ty"] or clean_input.strip() == "ty":
        return L(
            random.choice([
                "You're welcome! 😊 Feel free to ask anything else about El Niño, climate data, or how to prepare.",
                "Glad I could help! 🌤️ If you have more questions about El Niño or the local climate, I'm here.",
                "No problem! 👍 Ask me anything — El Niño effects, rainfall records, predictions, and more.",
            ]),
            random.choice([
                "Walang anuman! 😊 Huwag mag-atubiling magtanong pa tungkol sa El Niño, climate data, o kung paano maghanda.",
                "Natutuwa akong makatulong! 🌤️ Kung may mga katanungan ka pa tungkol sa El Niño o local climate, nandito lang ako.",
                "Sige lang! 👍 Magtanong ka anytime — effects ng El Niño, rainfall records, predictions, at marami pa.",
            ])
        )

    if any(k in clean_input for k in ["bye", "goodbye", "see you", "paalam", "hanggang sa muli"]):
        return L(
            "Goodbye! 👋 Stay safe and climate-aware. Come back anytime you have questions!",
            "Paalam! 👋 Mag-ingat at laging maging handa sa climate events. Bumalik ka anytime!"
        )

    # 2. What is ONI? (positive/negative checks come first to avoid substring conflict)
    if any(k in clean_input for k in ["positive oni", "oni positive", "what does a positive", "positibong oni", "high oni", "mataas na oni"]):
        conversation_state["last_topic"] = "oni_positive"
        return L(
            "📈 **What Does a Positive ONI Value Mean?**\n\n"
            "A **positive ONI value** means the Pacific Ocean surface is **warmer than the long-term average** in the Niño 3.4 region — the key monitoring zone.\n\n"
            "☀️ **Thresholds:**\n"
            "• **+0.5 to +0.9** — Weak El Niño\n"
            "• **+1.0 to +1.4** — Moderate El Niño\n"
            "• **+1.5 to +1.9** — Strong El Niño ☀️☀️\n"
            "• **+2.0 and above** — Very Strong El Niño ☀️☀️☀️\n\n"
            "For a value to officially count as El Niño, it needs to stay at **+0.5 or above for at least 5 consecutive overlapping 3-month periods**.\n\n"
            "🇵🇭 *For the Philippines, a positive ONI typically signals below-normal rainfall, drier conditions, and higher temperatures — especially felt in Padre Garcia, Batangas.*",
            "📈 **Ano ang Ibig Sabihin ng Positive ONI Value?**\n\n"
            "Ang **positive ONI value** ay nangangahulugang ang Pacific Ocean surface ay **mas mainit kaysa sa long-term average** sa Niño 3.4 region — ang key monitoring zone.\n\n"
            "☀️ **Thresholds:**\n"
            "• **+0.5 to +0.9** — Weak El Niño\n"
            "• **+1.0 to +1.4** — Moderate El Niño\n"
            "• **+1.5 to +1.9** — Strong El Niño ☀️☀️\n"
            "• **+2.0 pataas** — Very Strong El Niño ☀️☀️☀️\n\n"
            "Para ma-officially classify bilang El Niño, kailangan itong manatili sa **+0.5 o mas mataas sa loob ng hindi bababa sa 5 magkakasunod na overlapping 3-month periods**.\n\n"
            "🇵🇭 *Para sa Pilipinas, ang positive ONI ay karaniwang signal ng below-normal rainfall, mas dry na kondisyon, at mas mataas na temperature — na partikular na nararamdaman sa Padre Garcia, Batangas.*"
        )

    if any(k in clean_input for k in ["negative oni", "oni negative", "what does a negative", "negatibong oni", "low oni", "mababang oni"]):
        conversation_state["last_topic"] = "oni_negative"
        return L(
            "📉 **What Does a Negative ONI Value Mean?**\n\n"
            "A **negative ONI value** means the Pacific Ocean surface is **cooler than the long-term average** — the signature of La Niña.\n\n"
            "🌧️ **Thresholds:**\n"
            "• **-0.5 to -0.9** — Weak La Niña\n"
            "• **-1.0 to -1.4** — Moderate La Niña 🌧️🌧️\n"
            "• **-1.5 and below** — Strong La Niña 🌧️🌧️🌧️\n\n"
            "For a value to officially count as La Niña, it needs to stay at **-0.5 or below for at least 5 consecutive overlapping 3-month periods**.\n\n"
            "🇵🇭 *For the Philippines, a negative ONI typically signals above-normal rainfall, stronger monsoons, and an increased risk of flooding and typhoons.*",
            "📉 **Ano ang Ibig Sabihin ng Negative ONI Value?**\n\n"
            "Ang **negative ONI value** ay nangangahulugang ang Pacific Ocean surface ay **mas malamig kaysa sa long-term average** — ito ang signature ng La Niña.\n\n"
            "🌧️ **Thresholds:**\n"
            "• **-0.5 to -0.9** — Weak La Niña\n"
            "• **-1.0 to -1.4** — Moderate La Niña 🌧️🌧️\n"
            "• **-1.5 pababa** — Strong La Niña 🌧️🌧️🌧️\n\n"
            "Para ma-officially classify bilang La Niña, kailangan itong manatili sa **-0.5 o mas mababa sa loob ng hindi bababa sa 5 magkakasunod na overlapping 3-month periods**.\n\n"
            "🇵🇭 *Para sa Pilipinas, ang negative ONI ay karaniwang signal ng above-normal rainfall, mas malakas na monsoons, at mas mataas na risk ng flooding at typhoons.*"
        )

    if any(k in clean_input for k in ["what is oni", "ano ang oni", "what does oni mean", "oceanic nino index", "oni value", "what is the oni", "explain oni", "ipaliwanag ang oni", "ano ang ibig sabihin ng oni"]):
        if conversation_state["lang"] == "tl":
            return (
                "📊 **Ano ang ONI (Oceanic Niño Index)?**\n\n"
                "Ang ONI ay isang numero na sumusukat kung gaano **ka-mainit o ka-lamig** ang surface ng Pacific Ocean "
                "kumpara sa long-term average. Ito ang pangunahing indicator na ginagamit upang i-classify ang El Niño at La Niña.\n\n"
                "**Paano basahin ang ONI:**\n\n"
                "☀️☀️☀️ **+2.0 at pataas** — Very Strong El Niño\n"
                "☀️☀️ **+1.5 hanggang +1.9** — Strong El Niño\n"
                "☀️ **+0.5 hanggang +1.4** — Weak/Moderate El Niño\n"
                "🟡 **-0.4 hanggang +0.4** — Neutral (normal na kondisyon)\n"
                "🌧️ **-0.5 hanggang -0.9** — Weak La Niña\n"
                "🌧️🌧️ **-1.0 hanggang -1.4** — Moderate La Niña\n"
                "🌧️🌧️🌧️ **-1.5 at pababa** — Strong La Niña\n\n"
                f"Batay sa mga historical records (2010–2026), umabot ang ONI mula **{master_df['oni_value'].min():.2f}** hanggang **{master_df['oni_value'].max():.2f}**.\n"
                f"Ang pinakamalakas na El Niño sa record ay umabot sa ONI na **{master_df['oni_value'].max():.2f}** "
                f"noong {MONTH_NAMES[int(master_df.loc[master_df['oni_value'].idxmax(), 'Month'])]} "
                f"{int(master_df.loc[master_df['oni_value'].idxmax(), 'Year'])}."
            )
        return (
            "📊 **What is ONI (Oceanic Niño Index)?**\n\n"
            "The ONI is a number that measures how much **warmer or cooler** the Pacific Ocean surface is "
            "compared to the long-term average. It is the primary indicator used to classify El Niño and La Niña events.\n\n"
            "**How to read the ONI value:**\n\n"
            "☀️☀️☀️ **+2.0 and above** — Very Strong El Niño\n"
            "☀️☀️ **+1.5 to +1.9** — Strong El Niño\n"
            "☀️ **+0.5 to +1.4** — Weak/Moderate El Niño\n"
            "🟡 **-0.4 to +0.4** — Neutral (normal conditions)\n"
            "🌧️ **-0.5 to -0.9** — Weak La Niña\n"
            "🌧️🌧️ **-1.0 to -1.4** — Moderate La Niña\n"
            "🌧️🌧️🌧️ **-1.5 and below** — Strong La Niña\n\n"
            f"Based on historical records (2010–2026), ONI values have ranged from **{master_df['oni_value'].min():.2f}** to **{master_df['oni_value'].max():.2f}**.\n"
            f"The strongest El Niño on record reached an ONI of **{master_df['oni_value'].max():.2f}** "
            f"in {MONTH_NAMES[int(master_df.loc[master_df['oni_value'].idxmax(), 'Month'])]} "
            f"{int(master_df.loc[master_df['oni_value'].idxmax(), 'Year'])}."
        )

    # ENSO–El Niño relationship — separate from ONI vs ENSO
    if any(k in clean_input for k in [
        "enso related to el nino", "how is enso related", "enso and el nino",
        "el nino and enso", "relationship between enso and el nino",
        "relationship between el nino and enso", "how does enso relate to el nino",
        "enso vs el nino", "el nino vs enso", "enso el nino relationship",
        "connect enso", "enso connection", "enso link", "enso relate",
        "pagkakaugnay ng enso at el nino", "pagkakaugnay ng el nino at enso",
        "paano nakakonekta ang enso at el nino", "enso at el nino",
    ]):
        conversation_state["last_topic"] = "enso_el_nino_relationship"
        return L(
            "🔄 **How Is ENSO Related to El Niño?**\n\n"
            "El Niño is not a separate phenomenon from ENSO — it is actually **one phase of the ENSO cycle**.\n\n"
            "**ENSO** (El Niño–Southern Oscillation) is the name for the entire climate cycle. It has three phases:\n\n"
            "☀️ **El Niño** — the warm phase. Sea surface temperatures in the central and eastern Pacific rise above normal. "
            "For the Philippines, this means below-normal rainfall, drier conditions, and higher temperatures.\n\n"
            "🟡 **Neutral** — neither warm nor cool. Conditions stay close to the long-term average.\n\n"
            "🌧️ **La Niña** — the cool phase. Sea surface temperatures drop below normal. "
            "For the Philippines, this means above-normal rainfall, stronger monsoons, and increased flood risk.\n\n"
            "🔗 **In short:**\n"
            "• ENSO is the **cycle** — El Niño is one of its **phases**\n"
            "• Every El Niño event *is* an ENSO event, but not every ENSO event is El Niño\n"
            "• Scientists use the **ONI (Oceanic Niño Index)** to officially track which ENSO phase is active\n\n"
            "💡 *Think of ENSO as a slow ocean-atmosphere seesaw — El Niño is when it tips warm, La Niña is when it tips cool, and Neutral is when it's balanced.*\n\n"
            f"📊 *In our Padre Garcia dataset (2010–2026), ONI values ranged from **{master_df['oni_value'].min():.2f}** to **{master_df['oni_value'].max():.2f}**, "
            f"with the strongest El Niño (ONI {master_df['oni_value'].max():.2f}) recorded in {int(master_df.loc[master_df['oni_value'].idxmax(), 'Year'])}.*",

            "🔄 **Paano Nakakonekta ang ENSO at El Niño?**\n\n"
            "Ang El Niño ay hindi isang hiwalay na phenomenon mula sa ENSO — ito ay **isa sa mga phase ng ENSO cycle**.\n\n"
            "Ang **ENSO** (El Niño–Southern Oscillation) ang pangalan ng buong climate cycle. May tatlong phases ito:\n\n"
            "☀️ **El Niño** — ang warm phase. Ang sea surface temperatures sa central at eastern Pacific ay tumataas nang higit sa normal. "
            "Para sa Pilipinas, nangangahulugang below-normal rainfall, mas dry na kondisyon, at mas mataas na temperature.\n\n"
            "🟡 **Neutral** — hindi mainit o malamig. Ang mga kondisyon ay malapit sa long-term average.\n\n"
            "🌧️ **La Niña** — ang cool phase. Ang sea surface temperatures ay bumababa nang higit sa normal. "
            "Para sa Pilipinas, nangangahulugang above-normal rainfall, mas malakas na monsoon, at mas mataas na flood risk.\n\n"
            "🔗 **Sa madaling salita:**\n"
            "• Ang ENSO ang **cycle** — ang El Niño ay isa sa mga **phase** nito\n"
            "• Ang bawat El Niño event ay ENSO event, ngunit hindi lahat ng ENSO event ay El Niño\n"
            "• Ginagamit ng mga siyentipiko ang **ONI (Oceanic Niño Index)** para opisyal na subaybayan kung aling ENSO phase ang aktibo\n\n"
            "💡 *Isipin ang ENSO bilang isang mabagal na ocean-atmosphere na seesaw — ang El Niño ay kapag ito ay tumilt sa mainit, ang La Niña ay kapag ito ay tumilt sa malamig, at ang Neutral ay kapag ito ay balanse.*\n\n"
            f"📊 *Sa aming Padre Garcia dataset (2010–2026), ang ONI values ay mula **{master_df['oni_value'].min():.2f}** hanggang **{master_df['oni_value'].max():.2f}**, "
            f"na ang pinaka-malakas na El Niño (ONI {master_df['oni_value'].max():.2f}) ay naitala noong {int(master_df.loc[master_df['oni_value'].idxmax(), 'Year'])}.*"
        )

    # ONI vs ENSO comparison — must be BEFORE the broad "difference between" / "pagkakaiba" handler
    if any(k in clean_input for k in [
        "oni vs enso", "oni and enso", "compare oni", "oni compared",
        "oni with enso", "enso vs oni", "enso and oni",
        "difference between oni and enso", "difference between enso and oni",
        "relationship between oni and enso", "relationship between enso and oni",
        "oni enso difference", "pagkakaiba ng oni at enso",
        "pagkakaiba ng enso at oni", "oni vs enso", "how does oni relate to enso",
        "how oni relates", "oni relate",
    ]):
        conversation_state["last_topic"] = "oni_vs_enso"
        return L(
            "📊 **ONI vs ENSO — What's the Difference?**\n\n"
            "Great question — these two terms are closely related but not the same thing.\n\n"
            "🔄 **ENSO** (El Niño–Southern Oscillation) is the **big picture climate cycle** — "
            "the natural back-and-forth between El Niño, Neutral, and La Niña phases driven by "
            "Pacific Ocean temperatures and atmospheric pressure changes. It describes the *phenomenon*.\n\n"
            "📏 **ONI** (Oceanic Niño Index) is the **measuring tool** used to track ENSO. "
            "It is a 3-month running average of sea surface temperature anomalies in the "
            "Niño 3.4 region of the Pacific Ocean. Scientists use it to *officially classify* "
            "which ENSO phase we are currently in.\n\n"
            "🔗 **How they relate:**\n"
            "• ENSO is the cycle — ONI is the number that measures it\n"
            "• An ONI of **+0.5 or above** (for 5+ consecutive periods) → classified as **El Niño**\n"
            "• An ONI of **-0.5 or below** (for 5+ consecutive periods) → classified as **La Niña**\n"
            "• An ONI between **-0.4 and +0.4** → classified as **Neutral**\n\n"
            "💡 *Think of it this way: ENSO is the weather story, ONI is the score that tells you which chapter you're in.*\n\n"
            f"📊 **In our Padre Garcia dataset (2010–2026):**\n"
            f"• ONI values ranged from **{master_df['oni_value'].min():.2f}** to **{master_df['oni_value'].max():.2f}**\n"
            f"• The strongest El Niño on record (ONI {master_df['oni_value'].max():.2f}) occurred in "
            f"{master_df.loc[master_df['oni_value'].idxmax(), 'Year']:.0f}",
            "📊 **ONI vs ENSO — Ano ang Pagkakaiba?**\n\n"
            "Magandang tanong — malapit ang kaugnayan ng dalawa pero hindi sila pareho.\n\n"
            "🔄 **ENSO** (El Niño–Southern Oscillation) ang **malawak na climate cycle** — "
            "ang natural na back-and-forth sa pagitan ng El Niño, Neutral, at La Niña phases na "
            "driven ng Pacific Ocean temperatures at atmospheric pressure changes. Ito ang nagdedeskreto ng *phenomenon*.\n\n"
            "📏 **ONI** (Oceanic Niño Index) ang **measuring tool** na ginagamit para ma-track ang ENSO. "
            "Ito ay isang 3-month running average ng sea surface temperature anomalies sa "
            "Niño 3.4 region ng Pacific Ocean. Ginagamit ito ng mga scientists para *officially ma-classify* "
            "kung aling ENSO phase tayo ngayon.\n\n"
            "🔗 **Paano sila magkaugnay:**\n"
            "• Ang ENSO ang cycle — ang ONI ang numero na sumusukat nito\n"
            "• Ang ONI na **+0.5 o mas mataas** (sa loob ng 5+ consecutive periods) → classified bilang **El Niño**\n"
            "• Ang ONI na **-0.5 o mas mababa** (sa loob ng 5+ consecutive periods) → classified bilang **La Niña**\n"
            "• Ang ONI sa pagitan ng **-0.4 at +0.4** → classified bilang **Neutral**\n\n"
            "💡 *Para mas madaling maintindihan: ang ENSO ang kwento ng panahon, ang ONI ang score na nagsasabi kung aling kabanata ka ngayon.*\n\n"
            f"📊 **Sa aming Padre Garcia dataset (2010–2026):**\n"
            f"• Ang ONI values ay mula **{master_df['oni_value'].min():.2f}** hanggang **{master_df['oni_value'].max():.2f}**\n"
            f"• Ang pinaka-malakas na El Niño sa record (ONI {master_df['oni_value'].max():.2f}) ay naganap noong "
            f"{master_df.loc[master_df['oni_value'].idxmax(), 'Year']:.0f}"
        )

    # CL-NEW-6. How does ONI help monitor El Niño? / Relationship between ONI and El Niño
    if any(k in clean_input for k in [
        "how does oni help monitor", "how oni helps monitor", "oni help monitor",
        "how oni monitors el nino", "oni monitor el nino", "how is oni used",
        "relationship between oni and el nino", "relationship between el nino and oni",
        "oni and el nino relationship", "how does oni relate to el nino",
        "oni el nino connection", "pagkakaugnay ng oni at el nino",
        "paano ginagamit ang oni", "paano tinutukoy ang el nino gamit ang oni",
        "kung paano nasusubaybayan ang el nino", "subaybayan ang el nino"
    ]):
        conversation_state["last_topic"] = "oni_el_nino_relationship"
        return L(
            "📊 **How Does ONI Help Monitor El Niño?**\n\n"
            "The **ONI (Oceanic Niño Index)** is the primary tool scientists use to officially detect, track, and classify El Niño and La Niña events.\n\n"
            "🔍 **What ONI measures:**\n"
            "• ONI tracks the **sea surface temperature (SST) anomaly** in the Niño 3.4 region — a key area of the central Pacific Ocean between 5°N–5°S and 170°W–120°W\n"
            "• It uses a **3-month running average** to smooth out short-term fluctuations and capture sustained trends\n\n"
            "📏 **How El Niño is officially declared:**\n"
            "• If ONI reaches **+0.5°C or above** for at least **5 consecutive overlapping 3-month periods** → officially classified as **El Niño**\n"
            "• If ONI falls to **-0.5°C or below** for the same duration → officially classified as **La Niña**\n"
            "• Values between **-0.4 and +0.4** → **Neutral** conditions\n\n"
            "🔗 **The relationship in plain terms:**\n"
            "• El Niño *is* the phenomenon — ONI *measures* it\n"
            "• A higher positive ONI = a stronger, more impactful El Niño\n"
            "• Scientists, PAGASA, and disaster risk managers use ONI to issue early warnings months in advance\n\n"
            f"📊 *Based on our Padre Garcia dataset (2010–2026), ONI values ranged from **{master_df['oni_value'].min():.2f}** to **{master_df['oni_value'].max():.2f}**. The peak El Niño (ONI {master_df['oni_value'].max():.2f}) occurred in {int(master_df.loc[master_df['oni_value'].idxmax(), 'Year'])}.*",
            "📊 **Paano Tinutulungan ng ONI na Subaybayan ang El Niño?**\n\n"
            "Ang **ONI (Oceanic Niño Index)** ang pangunahing tool na ginagamit ng mga siyentipiko para opisyal na ma-detect, ma-track, at ma-classify ang mga El Niño at La Niña event.\n\n"
            "🔍 **Ano ang sinusukat ng ONI:**\n"
            "• Sinusukat ng ONI ang **sea surface temperature (SST) anomaly** sa Niño 3.4 region — isang key na lugar sa central Pacific Ocean sa pagitan ng 5°N–5°S at 170°W–120°W\n"
            "• Gumagamit ito ng **3-month running average** para ma-smooth out ang mga panandaliang pagbabago at makuha ang mga sustained na trend\n\n"
            "📏 **Paano opisyal na idinedeklara ang El Niño:**\n"
            "• Kung ang ONI ay umabot sa **+0.5°C o mas mataas** sa loob ng hindi bababa sa **5 magkakasunod na overlapping 3-month periods** → opisyal na classified bilang **El Niño**\n"
            "• Kung ang ONI ay bumaba sa **-0.5°C o mas mababa** sa parehong tagal → opisyal na classified bilang **La Niña**\n"
            "• Ang mga value sa pagitan ng **-0.4 at +0.4** → **Neutral** na kondisyon\n\n"
            "🔗 **Ang relasyon sa simpleng salita:**\n"
            "• Ang El Niño *ang* phenomenon — ang ONI *ang sumusukat* nito\n"
            "• Mas mataas na positive ONI = mas malakas at mas malaking epekto ng El Niño\n"
            "• Ginagamit ng mga siyentipiko, PAGASA, at mga disaster risk manager ang ONI para mag-issue ng maagang babala buwan bago pa mangyari\n\n"
            f"📊 *Batay sa aming Padre Garcia dataset (2010–2026), ang ONI values ay mula **{master_df['oni_value'].min():.2f}** hanggang **{master_df['oni_value'].max():.2f}**. Ang pinaka-mataas na El Niño (ONI {master_df['oni_value'].max():.2f}) ay naganap noong {int(master_df.loc[master_df['oni_value'].idxmax(), 'Year'])}.*"
        )

    # 3. El Niño vs La Niña difference
    if any(k in clean_input for k in ["difference between", "el nino vs la nina", "el nino and la nina", "pagkakaiba ng el nino at la nina", "el nino at la nina", "pagkakaiba", "ano ang pagkakaiba"]):
        conversation_state["last_topic"] = "enso_difference"
        return L(
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
            "• Agriculture impact: Flood damage to crops, but also good soil moisture for planting",
            "☀️🌧️ **El Niño vs La Niña — Ano ang Pagkakaiba?**\n\n"
            "Ang **El Niño** at **La Niña** ay dalawang natural na climate patterns na dulot ng pagbabago ng temperatura ng tubig sa gitna at silangang bahagi ng Pacific Ocean. Magkaiba ang epekto ng mga ito sa ating klima at panahon.\n\n"

"**☀️ El Niño** – Nangyayari kapag mas umiinit kaysa normal ang tubig sa Pacific Ocean. Sa Pilipinas, maaari itong magdulot ng mas mainit na panahon, mas kaunting pag-ulan, tagtuyot, at kakulangan sa suplay ng tubig.\n"
"**🌧️ La Niña** – Nangyayari naman kapag mas lumalamig kaysa normal ang tubig sa Pacific Ocean. Maaari itong magdulot ng mas maraming pag-ulan sa Pilipinas, na nagpapataas ng panganib ng pagbaha at landslide sa mga lugar na madaling maapektuhan.\n\n"

"**Tandaan:** El Niño ay karaniwang nauugnay sa mas mainit at tuyong kondisyon, habang La Niña ay mas malamig at mas maulang kondisyon. Pareho silang maaaring makaapekto sa agrikultura, kalusugan, at pang-araw-araw na pamumuhay.\n\n"

"Ano pa ang gusto mo itanong? 😊"
        )

    # 3. What is El Niño?
    if any(k in clean_input for k in ["what is el nino", "what is el niño", "define el nino", "define el niño", "definition of el nino", "explain el nino", "meaning of el nino", "ano ang el nino", "ano ang el niño", "el nino what", "el niño what", "about el nino", "tungkol sa el nino", "ipaliwanag ang el nino", "ibig sabihin ng el nino"]):
        conversation_state["last_topic"] = "el_nino_definition"
        return L(
            "🌊 **What is El Niño?**\n"
            "**El Niño** is a natural climate phenomenon characterized by the abnormal warming of sea surface temperatures "
            "in the central and eastern equatorial Pacific Ocean. "
            "In the Philippines, it leads to reduced rainfall, cause longer periods of dry weather and higher temperatures. These conditions may lead to water shortages and drought, affecting agriculture, crops, and daily activities.\n\n"
            "![El Niño Diagram 1](https://www.mamopanel.org/media/uploads/files/Screen_Shot_2017-04-20_at_16.52.00.png)\n"
            "![El Niño Diagram 2](https://www.enso.info/bild/bom_elnino.jpg)\n"
            "![El Niño vs La Niña](https://www.epicgardening.com/wp-content/uploads/2023/08/La-Nina-920x511.jpeg)\n"
            "*Its effects can vary depending on the location and strength of the event.*\n"
            "",
            "🌊 **Ano ang El Niño?**\n"
            "Ang **El Niño ay isang natural na climate pattern na nangyayari kapag mas umiinit kaysa sa normal ang sea surface temperature sa gitna at silangang bahagi ng tropical Pacific Ocean. "
            "Sa Pilipinas, maari itong magdulot ng mas kaunting rainfall, mas mahahabang dry spells, at mas mataas na temperature. Dahil dito, maaring magkaroon ng water shortages at drought na nakakaapekto sa agriculture, crops, at pang-araw-araw na gawain.\n\n"
            "![El Niño Diagram 1](https://www.mamopanel.org/media/uploads/files/Screen_Shot_2017-04-20_at_16.52.00.png)\n"
            "![El Niño Diagram 2](https://www.enso.info/bild/bom_elnino.jpg)\n"
            "![El Niño](https://cdn.zmescience.com/wp-content/uploads/2016/09/El-Nino-map-el-nino.jpg)\n"
            "*Nagkakaiba ang epekto ng El Niño depende sa lugar at strength ng pangyayari.*"
        )

    # 4. What is La Niña?
    if any(k in clean_input for k in ["what is la nina", "define la nina", "explain la nina", "ano ang la nina"]):
        return handle_la_nina_general()

    # CL-NEW-1. Why is it called El Niño?
    if any(k in clean_input for k in [
        "why is it called el nino", "why el nino is called", "why is el nino called",
        "where did el nino get its name", "origin of el nino name", "name el nino",
        "bakit tinatawag na el nino", "sino nagpangalan ng el nino", "origin ng pangalan ng el nino",
        "why named el nino", "what does el nino mean", "meaning of the name el nino"
    ]):
        conversation_state["last_topic"] = "el_nino_name_origin"
        return L(
            "📖 **Why Is It Called 'El Niño'?**\n\n"
            "The name **El Niño** comes from Spanish and means **'The Little Boy'** or **'The Christ Child.'**\n\n"
            "It was named by **Peruvian fishermen** in the 1600s. They noticed that every few years, warm ocean currents would appear along the coast of South America around **December** — close to Christmas time. "
            "Because of its arrival near the Christmas season, they named it *El Niño de Navidad* — 'The Christ Child of Christmas.'\n\n"
            "Over time, scientists adopted the name to describe the broader climate phenomenon — the periodic warming of the central and eastern Pacific Ocean that disrupts global weather patterns.\n\n"
            "🇵🇭 *For the Philippines, El Niño typically means below-normal rainfall, drier conditions, and higher temperatures — particularly affecting farming communities like Padre Garcia, Batangas.*",
            "📖 **Bakit Tinatawag na 'El Niño'?**\n\n"
            "Ang pangalang **El Niño** ay nagmula sa Espanyol at ang ibig sabihin ay **'The Little Boy'** o **'The Christ Child'** (Sanggol na Hesus).\n\n"
            "Pinangalanan ito ng mga **mangingisdang Peruano** noong 1600s. Napansin nila na tuwing ilang taon, may mainit na ocean current na dumarating sa baybayin ng South America mga panahon ng **Disyembre** — malapit sa Pasko. "
            "Dahil sa pagdating nito malapit sa kapaskuhan, pinangalanan nila itong *El Niño de Navidad* — 'Ang Sanggol na Hesus ng Pasko.'\n\n"
            "Sa paglipas ng panahon, ginamit ng mga siyentipiko ang pangalang ito para ilarawan ang mas malawak na climate phenomenon — ang pana-panahong pag-init ng central at eastern Pacific Ocean na nagdudulot ng disruption sa global weather patterns.\n\n"
            "🇵🇭 *Para sa Pilipinas, ang El Niño ay karaniwang nagdudulot ng below-normal rainfall, mas dry na kondisyon, at mas mataas na temperature — lalo na nakakaapekto sa mga farming communities tulad ng Padre Garcia, Batangas.*"
        )

    # CL-NEW-2. Why is it called La Niña?
    if any(k in clean_input for k in [
        "why is it called la nina", "why la nina is called", "why is la nina called",
        "where did la nina get its name", "origin of la nina name", "name la nina",
        "bakit tinatawag na la nina", "origin ng pangalan ng la nina",
        "why named la nina", "what does la nina mean", "meaning of the name la nina"
    ]):
        conversation_state["last_topic"] = "la_nina_name_origin"
        return L(
            "📖 **Why Is It Called 'La Niña'?**\n\n"
            "The name **La Niña** comes from Spanish and means **'The Little Girl'** or **'The Girl Child.'**\n\n"
            "Scientists coined the name as the direct counterpart to **El Niño** (The Little Boy). Since La Niña produces the **opposite effects** — cooling of the Pacific Ocean instead of warming — it was fitting to give it the feminine counterpart of the same name.\n\n"
            "The term was formally adopted in the **1980s** as scientists wanted a clear, simple way to distinguish between the two opposite phases of the **ENSO cycle**.\n\n"
            "🌧️ *While El Niño brings dryness and heat to the Philippines, La Niña brings the opposite — above-normal rainfall, stronger typhoons, and increased flooding risk.*",
            "📖 **Bakit Tinatawag na 'La Niña'?**\n\n"
            "Ang pangalang **La Niña** ay nagmula sa Espanyol at ang ibig sabihin ay **'The Little Girl'** o **'The Girl Child.'**\n\n"
            "Ginawa ng mga siyentipiko ang pangalang ito bilang direktang katapat ng **El Niño** (The Little Boy). Dahil ang La Niña ay nagdudulot ng **kabaligtarang epekto** — paglamig ng Pacific Ocean sa halip na pag-init — naaangkop na bigyan ito ng pambabae na bersyon ng parehong pangalan.\n\n"
            "Ang termino ay pormal na ginamit noong **1980s** nang nais ng mga siyentipiko ng isang malinaw at simpleng paraan para makilala ang dalawang magkasalungat na phases ng **ENSO cycle**.\n\n"
            "🌧️ *Habang ang El Niño ay nagdadala ng tagtuyot at init sa Pilipinas, ang La Niña naman ay nagdadala ng kabaligtaran — above-normal na ulan, mas malakas na bagyo, at mas mataas na risk ng pagbaha.*"
        )

    # CL-NEW-3. Is El Niño dangerous?
    if any(k in clean_input for k in [
        "is el nino dangerous", "is el nino harmful", "is el nino bad",
        "is el nino a threat", "can el nino kill", "how dangerous is el nino",
        "mapanganib ba ang el nino", "nakakapinsala ba ang el nino",
        "masama ba ang el nino", "banta ba ang el nino", "delikado ba ang el nino"
    ]):
        conversation_state["last_topic"] = "el_nino_danger"
        return L(
            "⚠️ **Is El Niño Dangerous?**\n\n"
            "Yes — El Niño can be very dangerous, especially for communities that depend on agriculture, consistent water supply, and outdoor livelihoods like Padre Garcia, Batangas.\n\n"
            "🔴 **The biggest risks:**\n"
            "• **Drought and water shortage** — prolonged dry spells deplete rivers, wells, and reservoirs\n"
            "• **Crop failure** — without rain, rice, corn, and vegetables fail, threatening food security\n"
            "• **Heat-related illness** — extreme temperatures lead to heat stroke and dehydration, especially for children and the elderly\n"
            "• **Wildfires** — dry conditions dramatically increase the risk of grassland and forest fires\n"
            "• **Power shortages** — hydroelectric power generation drops as water levels fall\n"
            "• **Economic losses** — farmers, fisherfolk, and businesses dependent on water all suffer financial impacts\n\n"
            "📊 *Historical data from Padre Garcia shows that strong El Niño events (like 2015–2016) caused some of the lowest recorded rainfall levels — directly linked to drought and agricultural losses in Batangas.*\n\n"
            "💡 *The danger can be reduced with early preparation: storing water, using drought-resistant crops, following PAGASA advisories, and knowing what to do during extreme heat.*",
            "⚠️ **Mapanganib Ba ang El Niño?**\n\n"
            "Oo — ang El Niño ay maaaring maging napaka-mapanganib, lalo na para sa mga komunidad na umaasa sa agrikultura, patuloy na suplay ng tubig, at outdoor na kabuhayan tulad ng Padre Garcia, Batangas.\n\n"
            "🔴 **Ang mga pinakamalalaking panganib:**\n"
            "• **Drought at kakulangan ng tubig** — ang mahahabang dry spells ay nagpapaubos ng mga ilog, balon, at reservoir\n"
            "• **Crop failure** — kung walang ulan, ang palay, mais, at gulay ay nabibigo, nagbabanta sa food security\n"
            "• **Heat-related illness** — ang matinding temperatura ay nagdudulot ng heat stroke at dehydration, lalo na para sa mga bata at matatanda\n"
            "• **Wildfire** — ang dry na kondisyon ay malaki ang nagpapataas ng panganib ng mga sunog sa damo at kagubatan\n"
            "• **Power shortages** — bumababa ang hydroelectric power generation habang bumababa ang water levels\n"
            "• **Economic losses** — ang mga magsasaka, mangingisda, at mga negosyong umaasa sa tubig ay lahat naghihirap\n\n"
            "📊 *Ang historical data mula Padre Garcia ay nagpapakita na ang malakas na El Niño events (tulad ng 2015–2016) ay nagdulot ng ilan sa pinakamababang naitala na rainfall levels — direktang linked sa drought at agricultural losses sa Batangas.*\n\n"
            "💡 *Ang panganib ay maaaring mabawasan sa pamamagitan ng maagang paghahanda: pag-iimbak ng tubig, paggamit ng drought-resistant crops, pagsunod sa mga advisory ng PAGASA, at pag-alam kung ano ang dapat gawin sa panahon ng matinding init.*"
        )

    # NEW-1. El Niño temperature threshold (ONI-based)
    if any(phrase in clean_input for phrase in ["anong temperature", "what temperature", "temperatura ng el nino", "temp ng el nino", "nagsisimula ang el nino", "starts el nino", "el nino threshold", "anong temp"]):
        return L(
            "🌡️ **El Niño Temperature Threshold:**\n\n"
            "El Niño is declared when the **Oceanic Niño Index (ONI)** — a 3-month running average of sea surface temperature anomalies "
            "in the Niño 3.4 region of the Pacific — reaches **+0.5°C or higher** above normal for at least 5 consecutive overlapping seasons.\n\n"
            "For the Philippines:\n"
            "• When it reaches +0.5°C to +0.9°C → **Weak El Niño**\n"
            "• When +1.0°C to +1.4°C → **Moderate El Niño**\n"
            "• When +1.5°C or higher → **Strong El Niño** (like 2015-2016)\n\n"
            "The higher the ONI, the more severe the impact on rainfall and temperature in Padre Garcia.\n\n"
            "💡 *Ask me 'What is ONI?' to learn how this index is calculated.*",
            "🌡️ **El Niño Temperature Threshold:**\n\n"
            "Ang El Niño ay ide-declare kapag ang **Oceanic Niño Index (ONI)** — isang 3-month running average ng sea surface temperature anomalies "
            "sa Niño 3.4 region ng Pacific — ay umabot sa **+0.5°C o mas mataas** kaysa sa normal para sa hindi bababa sa 5 consecutive overlapping seasons.\n\n"
            "Para sa Pilipinas:\n"
            "• Kapag umabot sa +0.5°C hanggang +0.9°C → **Weak El Niño**\n"
            "• Kapag +1.0°C hanggang +1.4°C → **Moderate El Niño**\n"
            "• Kapag +1.5°C o mas mataas → **Strong El Niño** (tulad ng 2015-2016)\n\n"
            "Ang mas mataas ang ONI, mas grabe ang epekto sa rainfall at temperatura sa Padre Garcia.\n\n"
            "💡 *Tanungin mo ako ng 'Ano ang ONI?' para malaman kung paano kinakalkula ang index na ito.*"
        )

    # NEW-2. Should I care about El Niño? (importance/relevance)
    if any(phrase in clean_input for phrase in ["dapat bang may pake", " paki", "dapat ba akong mag alala", "should i care", "why should i care", "bakit ako mag aalala", "importante ba", "is it important", "may pake ba ako", "mahalaga ba", "pakialam", "maging informed"]):
        return L(
            "⚠️ **Should I Care About El Niño?**\n\n"
            "**Yes**, It is important to stay informed about El Niño because it helps us prepare for its possible effects.\n\n"
           
"**☀️ Why is it important to know about El Niño?**\n"

"1. To conserve water – Water shortages may occur when there is less rainfall.\n"
"2. To protect our health – We can prevent dehydration and protect ourselves from extreme heat.\n"
"3. To protect crops – Farmers can prepare for possible drought conditions.\n"
"4. To prepare our families – We become more aware of what actions to take before the situation gets worse.\n\n"

"**Remember: Staying informed helps us prepare and protect ourselves, our families, and our communities from the possible effects of El Niño. 🌱",
            "⚠️ **Dapat Bang May Pake Ka sa El Niño?**\n\n"
            "**Oo naman**, Mahalaga na maging informed tayo tungkol sa El Niño dahil nakakatulong ito para maintindihan ang mga posibleng epekto nito at malaman kung paano tayo makapaghahanda.\n\n"
            
"** ☀️ Bakit mahalagang malaman ang tungkol sa El Niño?**\n"

"1. Para makatipid sa tubig – Maaaring magkaroon ng kakulangan sa tubig kapag kaunti ang ulan.\n"
"2. Para maprotektahan ang kalusugan – Makakaiwas tayo sa dehydration at sobrang init.\n"
"3. Para mapangalagaan ang mga pananim – Makapaghahanda ang mga magsasaka sa posibleng tagtuyot.\n"
"4. Para maging handa ang pamilya – Mas alam natin kung ano ang dapat gawin bago pa lumala ang sitwasyon.\n\n"

"**Tandaan:** Ang pagiging informed ay nakakatulong upang maging handa at maprotektahan ang sarili, pamilya, at komunidad laban sa mga epekto ng El Niño. 🌱"
        )
    
    # NEW-3. Is El Niño serious? (severity/danger assessment)
    if any(phrase in clean_input for phrase in ["seryoso ba", "is it serious", "dangerous ba", "nakaka takot ba", "is el nino dangerous", "gaano kaseryoso", "how serious", "seryoso ba ito", "seryoso ba ang el nino"]):
        return L(
            "🚨 **How Serious is El Niño?**\n\n"
            "**Yes, it is serious** — especially for agricultural communities like Padre Garcia, Batangas.\n\n"
            "Here's the historical evidence:\n"
            "• 🌾 **1997-1998 Super El Niño** — Millions affected, PHP 6 billion+ in agricultural losses nationwide\n"
            "• ☀️ **2015-2016 Strong El Niño** — Severe drought, water rationing in Metro Manila, massive crop failures\n"
            "• 📉 **Economic Impact** — Rice and corn production dropped by 30-40% during strong El Niño events\n\n"
            "**For Padre Garcia:**\n"
            "Based on historical data (2010-2026), El Niño years show:\n"
            "• Rainfall reduction: ~20% below normal\n"
            "• Temperature increase: +1-2°C above average\n"
            "• Longer dry spells affecting planting schedules\n\n"
            "📌 **Bottom line:** This is not a joke. Early preparation saves lives and livelihoods.\n\n"
            "💡 *Ask me 'How can I prepare for El Niño?' or 'What is the ONI value this year?'*",
            "🚨 **Gaano Kaseryoso ang El Niño?**\n\n"
            "**Oo, seryoso ito** — lalo na sa mga agricultural communities tulad ng Padre Garcia, Batangas.\n\n"
            "Narito ang historical evidence:\n"
            "• 🌾 **1997-1998 Super El Niño** — Milyun-milyong apektado, PHP 6 bilyong+ agricultural losses sa buong bansa\n"
            "• ☀️ **2015-2016 Strong El Niño** — Matinding drought, water rationing sa Metro Manila, malawakang crop failures\n"
            "• 📉 **Economic Impact** — Bumaba ng 30-40% ang rice at corn production sa strong El Niño events\n\n"
            "**Para sa Padre Garcia:**\n"
            "Batay sa historical data (2010-2026), ang El Niño years ay nagpapakita ng:\n"
            "• Pagbaba ng rainfall: ~20% below normal\n"
            "• Pagtaas ng temperatura: +1-2°C above average\n"
            "• Mas mahahabang dry spells na nakakaapekto sa planting schedules\n\n"
            "📌 **Bottom line:** Hindi ito biro. Ang maagang paghahanda ay nakakapagligtas ng buhay at kabuhayan.\n\n"
            "💡 *Tanungin mo ako ng 'Paano ako maghahanda para sa El Niño?' o 'Ano ang ONI value ngayong taon?'*"
        )
    
    # NEW-SIGNS. How to know if El Niño is happening / Signs of El Niño
    if any(k in clean_input for k in [
        "how to know if there is el nino", "how to know if el nino","signs el nino", "signs of el nino", "symptoms of el nino",
        "palatandaan ng el nino", "tanda ng el nino", "paano malaman kung may el nino", 
        "how to detect el nino", "indications of el nino", "how to know if el nino started",
        "paano malaman kung nagsimula na ang el nino"
    ]):
        conversation_state["last_topic"] = "el_nino_signs"
        return L(
            "🔍 **How to Know if El Niño is Happening:**\n\n"
            "You can observe these **signs** in Padre Garcia, Batangas:\n\n"
            "1. ☀️ **Extended dry periods** — Less rainfall than usual, especially during the wet season (June–November)\n"
            "2. 🌡️ **Higher temperatures** — Noticeably hotter days and nights, especially in March–May\n"
            "3. 💧 **Lower water levels** — Rivers, wells, and reservoirs drying up faster than normal\n"
            "4. 🌾 **Delayed or failed crop growth** — Plants wilting due to lack of water and intense heat\n\n"
            "📡 **Official Confirmation:**\n"
            "PAGASA monitors the **Oceanic Niño Index (ONI)**. When the ONI reaches **+0.5°C or higher** for 5 consecutive months, "
            "El Niño is officially declared.\n\n",
            "🔍 **Paano Malaman Kung May El Niño:**\n\n"
            "Maaari mong obserbahan ang mga **palatandaan** na ito sa Padre Garcia, Batangas:\n\n"
            "1. ☀️ **Mahabang panahon ng walang ulan** — Mas kaunti ang ulan kaysa normal, lalo na sa tag-ulan (Hunyo–Nobyembre)\n"
            "2. 🌡️ **Mas mataas ang temperatura** — Mas mainit ang araw at gabi, lalo sa Marso–Mayo\n"
            "3. 💧 **Bumababang lebel ng tubig** — Mas mabilis matuyo ang mga ilog, balon, at reservoir kaysa normal\n"
            "4. 🌾 **Delayed o nabigong paglaki ng pananim** — Nalalagas ang mga halaman dahil sa kakulangan ng tubig at matinding init\n\n"
            "📡 **Opisyal na Kumpirmasyon:**\n"
            "Sinusubaybayan ng PAGASA ang **Oceanic Niño Index (ONI)**. Kapag umabot ang ONI sa **+0.5°C o mas mataas** sa loob ng 5 consecutive months, "
            "opisyal nang ide-declare ang El Niño.\n\n"
           
        )

    # NEW-PREP. What should I prepare for El Niño?
    if any(k in clean_input for k in [
        "what should i prepare", "what should i prepare for el nino", "how to prepare for el nino",
        "ano ang dapat ihanda", "paano maghanda para sa el nino", "ano ang dapat kong ihanda",
        "preparation for el nino", "paghahanda para sa el nino", "how can i prepare",
        "paano ako maghahanda", "ano ang kailangan ihanda"
    ]):
        conversation_state["last_topic"] = "el_nino_preparation"
        return L(
            "🛡️ **What Should You Prepare for El Niño?**\n\n"
            "Here are **6 essential preparations** for Padre Garcia residents:\n\n"
            "1. 💧 **Store water** — Fill drums, tanks, and containers. Prioritize drinking water.\n"
            "2. 🌾 **Plant drought-resistant crops** — Consider sweet potato, cassava, or mungbean instead of water-intensive crops.\n"
            "3. 🏡 **Insulate your home** — Use light-colored roofing or shading to reduce indoor heat.\n"
            "4. 🩺 **Prepare for heat-related illness** — Keep ORS (oral rehydration salts), fans, and first aid supplies ready.\n"
            "5. 📻 **Stay informed** — Monitor PAGASA advisories via radio, TV, or online for updates on El Niño intensity.\n"
            "6. 🌱 **Use mulch and compost** — Helps soil retain moisture and reduces water needs for plants.\n\n"
            "⚠️ **For farmers:** Adjust planting schedules based on rainfall forecasts. Consider crop insurance if available.\n\n"
            "💡 *Ask me 'What are the health risks of El Niño?' or 'How can I save water during El Niño?' for more tips.*",
            "🛡️ **Ano ang Dapat Mong Ihanda para sa El Niño?**\n\n"
            "Narito ang **6 mahahalagang paghahanda** para sa mga residente ng Padre Garcia:\n\n"
            "1. 💧 **Mag-imbak ng tubig** — Punuin ang mga drum, tangke, at lalagyan. Unahin ang inuming tubig.\n"
            "2. 🌾 **Magtanim ng drought-resistant crops** — Isaalang-alang ang kamote, kamoteng kahoy, o munggo sa halip na pananim na maraming tubig ang kailangan.\n"
            "3. 🏡 **Palamigin ang bahay** — Gumamit ng light-colored na bubong o shading para bawasan ang init sa loob.\n"
            "4. 🩺 **Maghanda para sa heat-related illness** — Maglaan ng ORS (oral rehydration salts), electric fan, at first aid supplies.\n"
            "5. 📻 **Manatiling informed** — Subaybayan ang PAGASA advisories sa radyo, TV, o online para sa updates sa intensity ng El Niño.\n"
            "6. 🌱 **Gumamit ng mulch at compost** — Tumutulong ito na mapanatili ng lupa ang moisture at mabawasan ang pangangailangan ng tubig para sa halaman.\n\n"
            "⚠️ **Para sa mga magsasaka:** Ayusin ang planting schedules batay sa rainfall forecasts. Isaalang-alang ang crop insurance kung available.\n\n"
            "💡 *Tanungin mo ako ng 'Ano ang health risks ng El Niño?' o 'Paano ako makakatipid ng tubig sa panahon ng El Niño?' para sa mas maraming tips.*"
        )

    # CL-2. What causes El Niño?
    if any(k in clean_input for k in ["what causes el nino", "what causes el niño", "bakit nagkakaroon ng el nino", "bakit may el nino", "why does el nino happen", "sanhi ng el nino", "cause of el nino", "causes el nino", "nagkakaroon ng el nino"]):
        conversation_state["last_topic"] = "el_nino_causes"
        return L(
            "🌊 **What Causes El Niño?**\n\n"
            "El Niño is caused by a weakening — or even a reversal — of the **trade winds** that normally blow warm surface water westward across the Pacific Ocean.\n\n"
            "Here's what happens step by step:\n"
            "1. Normally, strong trade winds push warm water toward Southeast Asia and Australia, keeping the eastern Pacific cooler.\n"
            "2. During El Niño, those trade winds weaken or reverse direction.\n"
            "3. Warm water that was pushed west sloshes back east toward South America.\n"
            "4. This raises sea surface temperatures in the central and eastern Pacific — the core signal scientists look for.\n\n"
            "🔄 It's part of the **ENSO cycle** (El Niño–Southern Oscillation), a natural back-and-forth between El Niño, Neutral, and La Niña phases driven by ocean-atmosphere interactions.\n\n"
            "⚠️ *No single human action causes El Niño — it is a natural cycle. However, climate change may be influencing its frequency and intensity.*",
            "🌊 **Bakit Nagkakaroon ng El Niño?**\n\n"
            "Ang El Niño ay nangyayari kapag humina — o bumaligtad — ang **trade winds** na normal na nagdadala ng mainit na surface water patungong kanluran sa Pacific Ocean.\n\n"
            "Ganito ang nangyayari step by step:\n"
            "1. Normally, ang malakas na trade winds ay nagtutulak ng mainit na tubig patungo sa Southeast Asia at Australia, kaya mas malamig ang eastern Pacific.\n"
            "2. Sa panahon ng El Niño, humihina o bumabalik ang direksyon ng mga trade winds na ito.\n"
            "3. Ang mainit na tubig na dati ay tinutulak patungo sa kanluran ay bumabalik sa silangan, papalapit sa South America.\n"
            "4. Dahil dito, tumataas ang sea surface temperature sa central at eastern Pacific — ito ang pangunahing signal na hinahanap ng mga scientists.\n\n"
            "🔄 Ito ay bahagi ng **ENSO cycle** (El Niño–Southern Oscillation) — isang natural na back-and-forth sa pagitan ng El Niño, Neutral, at La Niña phases, na driven ng ocean-atmosphere interactions.\n\n"
            "⚠️ *Walang single na human action ang nagdudulot ng El Niño — natural cycle ito. Pero posibleng may influence ang climate change sa frequency at intensity nito.*"
        )

    # CL-3. How does El Niño work?
    if any(k in clean_input for k in ["how does el nino work", "how el nino works", "paano gumagana ang el nino", "mechanics of el nino", "paano nagtatrabaho ang el nino", "how it works", "process of el nino"]):
        conversation_state["last_topic"] = "el_nino_how"
        return L(
            "⚙️ **How Does El Niño Work?**\n\n"
            "El Niño is driven by the relationship between the ocean and the atmosphere in the Pacific:\n\n"
            "🌬️ **Step 1 — Trade winds weaken**\n"
            "Normally, east-to-west trade winds keep warm water pooled near Asia. When those winds weaken, the warm water spreads back eastward.\n\n"
            "🌡️ **Step 2 — Ocean warms up in the east**\n"
            "Sea surface temperatures rise above normal across the central and eastern Pacific. Scientists measure this using the **ONI (Oceanic Niño Index)**.\n\n"
            "☁️ **Step 3 — Rainfall patterns shift globally**\n"
            "Where the ocean is warmest, warm moist air rises — creating clouds and rain there. This disrupts normal rainfall patterns worldwide:\n"
            "• More rain over the eastern Pacific and parts of South America\n"
            "• Less rain over Southeast Asia, Australia, and the Philippines\n\n"
            "🌾 **Step 4 — Local impacts**\n"
            "For the Philippines and Padre Garcia specifically, the reduced rainfall means drier conditions, higher temperatures, and increased drought risk.\n\n"
            "🔄 Eventually the ocean and atmosphere rebalance, and the cycle returns to Neutral — or shifts to La Niña.",
            "⚙️ **Paano Gumagana ang El Niño?**\n\n"
            "Ang El Niño ay driven ng relationship ng ocean at atmosphere sa Pacific:\n\n"
            "🌬️ **Step 1 — Humihina ang trade winds**\n"
            "Normally, ang east-to-west trade winds ay nagpapanatili ng mainit na tubig malapit sa Asia. Kapag humina ang mga winds na ito, ang mainit na tubig ay kumakalat pabalik sa silangan.\n\n"
            "🌡️ **Step 2 — Umiinit ang ocean sa silangan**\n"
            "Tumataaas ang sea surface temperature nang higit sa normal sa central at eastern Pacific. Sinusukat ito ng mga scientists gamit ang **ONI (Oceanic Niño Index)**.\n\n"
            "☁️ **Step 3 — Nagbabago ang rainfall patterns globally**\n"
            "Kung saan pinaka-mainit ang ocean, ang mainit na humid air ay umaangat — lumilikha ng clouds at ulan doon. Nagdudulot ito ng disruption sa normal na rainfall patterns sa buong mundo:\n"
            "• Mas maraming ulan sa eastern Pacific at ilang bahagi ng South America\n"
            "• Mas kaunting ulan sa Southeast Asia, Australia, at ang Pilipinas\n\n"
            "🌾 **Step 4 — Local na epekto**\n"
            "Para sa Pilipinas at Padre Garcia specifically, ang reduced rainfall ay nangangahulugang mas dry na kondisyon, mas mataas na temperature, at mas mataas na drought risk.\n\n"
            "🔄 Sa bandang huli, nag-rebalance ang ocean at atmosphere, at ang cycle ay bumabalik sa Neutral — o lumipat sa La Niña."
        )

    # CL-6. How long does El Niño last?
    if any(k in clean_input for k in ["how long does el nino last", "gaano katagal ang el nino", "how long el nino", "duration of el nino", "tagal ng el nino", "gaano katagal", "how many months", "ilang buwan ang el nino"]):
        conversation_state["last_topic"] = "el_nino_duration"
        return L(
            "⏱️ **How Long Does El Niño Last?**\n\n"
            "A typical El Niño event lasts about **9 to 12 months**, though stronger events can persist for **18 months or more**.\n\n"
            "📅 **General timeline:**\n"
            "• El Niño usually develops between **April and June**\n"
            "• It peaks in intensity around **December to January** (this is why it was named 'El Niño' — the Christ child — by South American fishermen who noticed the warm current around Christmas)\n"
            "• It typically weakens and ends by **March to May** of the following year\n\n"
            "🔄 After it ends, conditions either return to **Neutral** or shift to **La Niña**. The full ENSO cycle from one El Niño to the next typically takes **2 to 7 years**.\n\n"
            "💡 *The 2015–2016 El Niño was one of the strongest on record and lasted roughly 18 months, significantly affecting rainfall across Padre Garcia and the rest of the Philippines.*",
            "⏱️ **Gaano Katagal ang El Niño?**\n\n"
            "Ang typical na El Niño event ay tumatagal ng mga **9 hanggang 12 buwan**, pero ang mas malakas na events ay maaaring tumagal ng **18 buwan o higit pa**.\n\n"
            "📅 **General timeline:**\n"
            "• Karaniwang nag-develop ang El Niño sa pagitan ng **April at June**\n"
            "• Naaabot ang peak ng intensity nito sa paligid ng **December hanggang January** (kaya tinawag itong 'El Niño' — the Christ child — ng mga South American fishermen na napansin ang mainit na current malapit sa Pasko)\n"
            "• Karaniwang humihina at nagtatapos ito sa **March hanggang May** ng susunod na taon\n\n"
            "🔄 Pagkatapos, bumabalik ang mga kondisyon sa **Neutral** o lumilipat sa **La Niña**. Ang buong ENSO cycle mula sa isang El Niño hanggang sa susunod ay karaniwang tumatagal ng **2 hanggang 7 taon**.\n\n"
            "💡 *Ang 2015–2016 El Niño ay isa sa pinaka-malakas na naitala at tumagal ng mga 18 buwan, na malaki ang epekto sa rainfall sa Padre Garcia at sa buong Pilipinas.*"
        )

    # CL-7. What is ENSO?
    if any(k in clean_input for k in ["what is enso", "ano ang enso", "explain enso", "what does enso mean", "enso meaning", "enso cycle", "el nino southern oscillation", "ibig sabihin ng enso"]):
        conversation_state["last_topic"] = "enso_definition"
        return L(
            "🔄 **What is ENSO?**\n\n"
            "**ENSO** stands for **El Niño–Southern Oscillation**. It is a natural, recurring climate cycle in the Pacific Ocean that drives shifts in global weather patterns — including rainfall and temperature across the Philippines.\n\n"
            "ENSO has three phases:\n\n"
            "☀️ **El Niño** — Warmer-than-normal sea surface temperatures in the central/eastern Pacific. Causes drier conditions in the Philippines.\n\n"
            "🟡 **Neutral** — Sea temperatures near the long-term average. Normal weather conditions.\n\n"
            "🌧️ **La Niña** — Cooler-than-normal sea surface temperatures. Causes wetter-than-normal conditions and stronger typhoons in the Philippines.\n\n"
            "📊 Scientists track which phase we're in using the **ONI (Oceanic Niño Index)** — a running average of sea surface temperature anomalies in a key part of the Pacific.\n\n"
            "🔁 *The cycle typically shifts every 2–7 years, though it doesn't follow a fixed schedule.*",
            "🔄 **Ano ang ENSO?**\n\n"
            "Ang **ENSO** ay ang **El Niño–Southern Oscillation**. Ito ay isang natural, recurring na climate cycle sa Pacific Ocean na nagdudulot ng changes sa global weather patterns — kasama na ang rainfall at temperature sa Pilipinas.\n\n"
            "May tatlong phases ang ENSO:\n\n"
            "☀️ **El Niño** — Mas mainit kaysa sa normal ang sea surface temperatures sa central/eastern Pacific. Nagdudulot ng mas dry na kondisyon sa Pilipinas.\n\n"
            "🟡 **Neutral** — Malapit sa long-term average ang sea temperatures. Normal na weather conditions.\n\n"
            "🌧️ **La Niña** — Mas malamig kaysa sa normal ang sea surface temperatures. Nagdudulot ng mas maraming ulan at mas malakas na typhoons sa Pilipinas.\n\n"
            "📊 Ginagamit ng mga scientists ang **ONI (Oceanic Niño Index)** para ma-track kung aling phase tayo ngayon — isang running average ng sea surface temperature anomalies sa isang key na bahagi ng Pacific.\n\n"
            "🔁 *Ang cycle ay karaniwang nagbabago bawat 2–7 taon, kahit hindi ito sumusunod sa fixed schedule.*"
        )

    # CL-NEW-4. How does La Niña happen?
    if any(k in clean_input for k in [
        "how does la nina happen", "how la nina happens", "how does la nina occur",
        "what causes la nina", "cause of la nina", "sanhi ng la nina",
        "bakit nagkakaroon ng la nina", "paano nangyayari ang la nina",
        "how la nina works", "paano gumagana ang la nina", "la nina happen"
    ]):
        conversation_state["last_topic"] = "la_nina_causes"
        return L(
            "🌧️ **How Does La Niña Happen?**\n\n"
            "La Niña is the **cool phase** of the ENSO cycle — the opposite of El Niño. Here's how it develops:\n\n"
            "🌬️ **Step 1 — Trade winds strengthen**\n"
            "Unlike El Niño where trade winds weaken, during La Niña the **east-to-west trade winds become stronger than normal**. This pushes more warm surface water toward Southeast Asia and Australia.\n\n"
            "🌡️ **Step 2 — Eastern Pacific cools down**\n"
            "As warm water is pushed westward, the central and eastern Pacific Ocean becomes **cooler than normal**. Cold deep water rises to the surface (called *upwelling*) along the coast of South America.\n\n"
            "☁️ **Step 3 — Rainfall shifts toward Asia**\n"
            "With warm water pooled near Asia, moisture-laden air rises over the western Pacific — bringing more clouds, heavier rain, and stronger monsoons to Southeast Asia, including the Philippines.\n\n"
            "🔄 **The cycle:**\n"
            "La Niña usually follows an El Niño event, though it can also emerge from Neutral conditions. It is tracked using the **ONI (Oceanic Niño Index)** — a value of **-0.5 or below** for 5+ consecutive periods officially classifies it as La Niña.\n\n"
            "🇵🇭 *For Padre Garcia and the Philippines, La Niña means above-normal rainfall, stronger typhoon seasons, and increased flood risk.*",
            "🌧️ **Paano Nangyayari ang La Niña?**\n\n"
            "Ang La Niña ang **cool phase** ng ENSO cycle — ang kabaligtaran ng El Niño. Ganito ito nag-develop:\n\n"
            "🌬️ **Step 1 — Lumakas ang trade winds**\n"
            "Hindi tulad ng El Niño kung saan humihina ang trade winds, sa panahon ng La Niña ang **east-to-west trade winds ay nagiging mas malakas kaysa sa normal**. Nagdudulot ito ng mas maraming mainit na surface water na nagtutulak patungo sa Southeast Asia at Australia.\n\n"
            "🌡️ **Step 2 — Lumalamig ang eastern Pacific**\n"
            "Habang tinutulak ang mainit na tubig patungong kanluran, ang central at eastern Pacific Ocean ay nagiging **mas malamig kaysa sa normal**. Ang malamig na malalim na tubig ay umaakyat sa ibabaw (*upwelling*) sa baybayin ng South America.\n\n"
            "☁️ **Step 3 — Lumilipat ang ulan patungo sa Asia**\n"
            "Dahil nakaapon ang mainit na tubig malapit sa Asia, ang mamasa-masang hangin ay umaangat sa western Pacific — nagdudulot ng mas maraming ulap, mas malakas na ulan, at mas matinding monsoon sa Southeast Asia, kasama na ang Pilipinas.\n\n"
            "🔄 **Ang cycle:**\n"
            "Ang La Niña ay karaniwang sumusunod pagkatapos ng isang El Niño event, kahit maaari itong lumabas mula sa Neutral na kondisyon. Ito ay nino-monitor gamit ang **ONI (Oceanic Niño Index)** — ang value na **-0.5 o mas mababa** sa loob ng 5+ consecutive na periods ay officially na-classify bilang La Niña.\n\n"
            "🇵🇭 *Para sa Padre Garcia at ang Pilipinas, ang La Niña ay nangangahulugang above-normal na rainfall, mas malakas na typhoon season, at mas mataas na risk ng pagbaha.*"
        )

# PAGASA Inquiry Handler
    if any(k in clean_input for k in [
        "what is pagasa", "ano ang pagasa", "what is pag-asa", "ano ang pag-asa",
        "what does pagasa do", "ano ang ginagawa ng pagasa", "pagasa", "pag-asa", "pag asa"
    ]):
        conversation_state["last_topic"] = "pagasa_info"
        return L(
            "🌦️ **What is PAGASA?**\n\n"
            "**PAGASA** stands for the **Philippine Atmospheric, Geophysical and Astronomical Services Administration**.\n\n"
            "It is the national weather agency of the Philippines under DOST responsible for:\n"
            "• Monitoring typhoons, tropical depressions, and monsoons\n"
            "• Issuing official **El Niño and La Niña climate advisories**\n"
            "• Providing seasonal rainfall outlooks for agricultural planning in towns like Padre Garcia\n"
            "• Issuing extreme heat index and severe weather alerts\n\n"
            "📞 **PAGASA Hotline:** **1553** | **Website:** [www.pagasa.dost.gov.ph](https://www.pagasa.dost.gov.ph/)",

            "🌦️ **Ano ang PAGASA?**\n\n"
            "Ang **PAGASA** ay ang **Philippine Atmospheric, Geophysical and Astronomical Services Administration**.\n\n"
            "Ito ang pambansang ahensya ng panahon sa ilalim ng DOST na may pananagutan sa:\n"
            "• Pagsubaybay sa mga bagyo, habagat, at amihan\n"
            "• Paglalabas ng mga opisyal na **advisory para sa El Niño at La Niña**\n"
            "• Pagbibigay ng seasonal rainfall forecast na gabay ng mga magsasaka sa Padre Garcia\n"
            "• Paglalabas ng heat index alerts at babala sa matinding init\n\n"
            "📞 **PAGASA Hotline:** **1553** | **Website:** [www.pagasa.dost.gov.ph](https://www.pagasa.dost.gov.ph/)"
        )
    # CL-NEW-5. How long does La Niña last?
    if any(k in clean_input for k in [
        "how long does la nina last", "how long la nina", "duration of la nina",
        "gaano katagal ang la nina", "tagal ng la nina", "how many months la nina",
        "ilang buwan ang la nina", "how long does lanina last", "la nina duration"
    ]):
        conversation_state["last_topic"] = "la_nina_duration"
        return L(
            "⏱️ **How Long Does La Niña Last?**\n\n"
            "A typical La Niña event lasts about **9 to 12 months**, similar to El Niño. However, La Niña events can sometimes last **two years or more**, making them potentially longer-lived than El Niño.\n\n"
            "📅 **General timeline:**\n"
            "• La Niña usually develops between **June and August**\n"
            "• It reaches peak intensity around **December to February**\n"
            "• It typically weakens and ends by **April to May** of the following year\n\n"
            "🔄 **What comes after:**\n"
            "After La Niña ends, conditions usually return to **Neutral** before potentially shifting to El Niño. The complete ENSO cycle (from one phase back to the same phase) typically takes **2 to 7 years**.\n\n"
            "🌧️ *Extended La Niña events can mean two consecutive wet and stormy seasons for the Philippines — increasing the cumulative flood and landslide risk significantly.*\n\n"
            "📊 *Based on our Padre Garcia dataset (2010–2026), La Niña periods are associated with some of the highest monthly rainfall readings on record.*",
            "⏱️ **Gaano Katagal ang La Niña?**\n\n"
            "Ang isang typical na La Niña event ay tumatagal ng mga **9 hanggang 12 buwan**, katulad ng El Niño. Gayunpaman, ang mga La Niña event ay maaaring tumagal ng **dalawang taon o higit pa**, kaya posibleng mas matagal ito kaysa sa El Niño.\n\n"
            "📅 **General na timeline:**\n"
            "• Karaniwang nag-develop ang La Niña sa pagitan ng **Hunyo at Agosto**\n"
            "• Naaabot ang peak ng intensity nito sa paligid ng **Disyembre hanggang Pebrero**\n"
            "• Karaniwang humihina at nagtatapos ito sa **Abril hanggang Mayo** ng susunod na taon\n\n"
            "🔄 **Ano ang darating pagkatapos:**\n"
            "Pagkatapos ng La Niña, ang mga kondisyon ay karaniwang bumabalik sa **Neutral** bago posibleng lumipat sa El Niño. Ang kumpletong ENSO cycle (mula sa isang phase pabalik sa parehong phase) ay karaniwang tumatagal ng **2 hanggang 7 taon**.\n\n"
            "🌧️ *Ang matagal na La Niña events ay maaaring mangahulugang dalawang magkakasunod na wet at stormyng season para sa Pilipinas — malaki ang nagpapataas ng cumulative flood at landslide risk.*\n\n"
            "📊 *Batay sa aming Padre Garcia dataset (2010–2026), ang mga La Niña period ay associated sa ilan sa pinakamataas na monthly rainfall readings sa rekord.*"
        )

    # CL2-NEW. Does El Niño reduce rainfall?
    if any(k in clean_input for k in [
        "reduce rainfall", "reduces rainfall", "el nino reduce rain", "does el nino reduce",
        "less rainfall", "less rain", "lower rainfall", "bawasan ang ulan",
        "nagbabawas ng ulan", "nagbabawas ba ng ulan", "does el nino lower rainfall",
        "can el nino reduce", "kaunting ulan", "mas kaunting ulan el nino",
        "rainfall decrease", "decrease rainfall", "decrease in rainfall",
        "pagbaba ng ulan", "bumababa ang ulan"
    ]):
        conversation_state["last_topic"] = "el_nino_reduce_rainfall"
        return L(
            "🌧️ **Can El Niño Reduce Rainfall?**\n\n"
            "Yes — reducing rainfall is one of El Niño's most consistent and well-documented effects on the Philippines.\n\n"
            "During El Niño, the warm pool of water that normally sits near Southeast Asia shifts eastward toward the central Pacific. "
            "This pulls rain-producing storm systems away from the Philippines, resulting in significantly less rainfall — especially during the dry season (December to May).\n\n"
            "📊 **What our local data shows (2010–2026):**\n"
            "• **Padre Garcia average during El Niño months:** ~159.8 mm/month\n"
            "• **Padre Garcia average during normal months:** ~198.3 mm/month\n"
            "• That's roughly **20% less rainfall** during El Niño compared to neutral conditions\n\n"
            "⚠️ **The effects of reduced rainfall:**\n"
            "• Rivers, wells, and reservoirs dry up faster\n"
            "• Crops receive less water — rice and corn harvests drop\n"
            "• Drought conditions develop when dry spells extend for weeks or months\n"
            "• Water shortages affect households and communities\n\n"
            "🔢 **Intensity matters:**\n"
            "• Weak El Niño — moderate rainfall reduction\n"
            "• Strong El Niño (like 2015–2016) — severe reduction, drought conditions\n\n"
            "💡 *Not every month is equally affected — rainfall reduction is most pronounced from December to May, which already coincides with the Philippines' natural dry season.*",

            "🌧️ **Nagbabawas Ba ng Ulan ang El Niño?**\n\n"
            "Oo — ang pagbabawas ng ulan ay isa sa mga pinaka-konsistente at well-documented na epekto ng El Niño sa Pilipinas.\n\n"
            "Sa panahon ng El Niño, ang mainit na pool ng tubig na karaniwan ay malapit sa Southeast Asia ay lumilipat patungong silangan sa central Pacific. "
            "Inilalayo nito ang mga rain-producing storm systems mula sa Pilipinas, na nagdudulot ng mas kaunting ulan — lalo na sa dry season (Disyembre hanggang Mayo).\n\n"
            "📊 **Ano ang sinasabi ng aming lokal na data (2010–2026):**\n"
            "• **Padre Garcia average sa El Niño months:** ~159.8 mm/buwan\n"
            "• **Padre Garcia average sa normal months:** ~198.3 mm/buwan\n"
            "• Iyon ay humigit-kumulang **20% na mas kaunting ulan** sa El Niño kumpara sa neutral na kondisyon\n\n"
            "⚠️ **Ang mga epekto ng reduced rainfall:**\n"
            "• Mas mabilis natutuyo ang mga ilog, balon, at reservoir\n"
            "• Mas kaunting tubig para sa mga pananim — bumababa ang ani ng palay at mais\n"
            "• Nagde-develop ang drought kapag tumagal ang dry spells nang maraming linggo o buwan\n"
            "• Nakakaapekto sa mga household at komunidad ang water shortages\n\n"
            "🔢 **Mahalaga ang intensity:**\n"
            "• Weak El Niño — katamtamang pagbaba ng ulan\n"
            "• Strong El Niño (tulad ng 2015–2016) — malubhang pagbaba, drought conditions\n\n"
            "💡 *Hindi pantay ang epekto sa bawat buwan — pinakamalaki ang pagbaba ng ulan mula Disyembre hanggang Mayo, na siyang natural na dry season na ng Pilipinas.*"
        )

    # CL2-2. Does El Niño cause drought?
    if any(k in clean_input for k in ["cause drought", "causes drought", "el nino drought", "drought el nino", "nagdudulot ng drought", "nagdudulot ba ng drought", "does el nino cause drought", "tagtuyot ba", "drought ba"]):
        conversation_state["last_topic"] = "el_nino_drought"
        return L(
            "🏜️ **Does El Niño Cause Drought?**\n\n"
            "Yes — El Niño is one of the leading causes of drought in the Philippines.\n\n"
            "During El Niño, rainfall drops significantly below normal, especially from **December to May**. When this dry period stretches for weeks or months without significant rain, it becomes a drought.\n\n"
            "💧 **How it happens in Padre Garcia:**\n"
            "• Rainfall decreases well below the monthly average\n"
            "• Rivers, wells, and reservoirs slowly dry up\n"
            "• Soil loses moisture, becoming cracked and hard\n"
            "• Plants, crops, and livestock begin to suffer from water stress\n\n"
            "📊 *Our local data (2010–2026) shows that El Niño months consistently record the lowest rainfall levels — directly linked to drought conditions in Batangas.*\n\n"
            "⚠️ *Not every El Niño leads to severe drought — severity depends on the strength of the event and local rainfall patterns.*",
            "🏜️ **Nagdudulot Ba ng Drought ang El Niño?**\n\n"
            "Oo — ang El Niño ay isa sa mga pangunahing sanhi ng drought sa Pilipinas.\n\n"
            "Sa panahon ng El Niño, malaki ang pagbaba ng rainfall nang higit sa normal, lalo na mula **Disyembre hanggang Mayo**. Kapag tumagal ang dry period na ito nang maraming linggo o buwan nang walang sapat na ulan, nagiging drought na ito.\n\n"
            "💧 **Paano ito nangyayari sa Padre Garcia:**\n"
            "• Bumababa ang rainfall nang malayo sa monthly average\n"
            "• Unti-unting natutuyo ang mga ilog, balon, at reservoir\n"
            "• Nawawalan ng moisture ang lupa, nagiging crack at matigas\n"
            "• Nagsisimulang mag-suffer ang mga halaman, crops, at livestock dahil sa water stress\n\n"
            "📊 *Ang aming local data (2010–2026) ay nagpapakita na ang mga El Niño months ay palaging may pinakamababang rainfall levels — direktang linked sa drought conditions sa Batangas.*\n\n"
            "⚠️ *Hindi lahat ng El Niño ay nagdudulot ng malubhang drought — depende ang severity sa strength ng event at sa local rainfall patterns.*"
        )

    # CL2-3. How does El Niño affect water supply?
    if any(k in clean_input for k in ["water supply", "affect water", "water shortage", "suplay ng tubig", "kumonti ang tubig", "kakulangan ng tubig", "water resources", "water source", "water level", "epekto sa tubig", "epekto ng el nino sa water"]):
        # Don't intercept water shortage action/reporting questions — let CL4 handle them
        if any(k in clean_input for k in ["report", "where to report", "what to do during", "kapag walang", "mag-ulat", "during water shortage", "what should i do during water", "water shortages"]):
            pass
        else:
            conversation_state["last_topic"] = "el_nino_water"
            return L(
                "💧 **How Does El Niño Affect Water Supply?**\n\n"
                "El Niño severely reduces available water across all levels — from rivers and reservoirs down to household taps.\n\n"
                "🌊 **Surface Water:**\n"
                "• Rivers and streams carry less water due to reduced rainfall\n"
                "• Reservoirs and dams reach critically low levels\n"
                "• Irrigation canals may run dry during peak dry months\n\n"
                "🪣 **Groundwater:**\n"
                "• Wells dry up faster as less rain seeps into the ground to recharge aquifers\n"
                "• Communities relying on deep wells may experience supply cuts\n\n"
                "🏠 **For Households in Padre Garcia:**\n"
                "• Tap water availability can become irregular\n"
                "• Water rationing may be enforced by local government\n"
                "• Residents may need to travel further to collect water\n\n"
                "⚡ **Secondary Impact:**\n"
                "• Hydroelectric power plants produce less electricity as water levels drop, contributing to power shortages\n\n"
                "💡 *Tip: Store clean water in covered containers before dry season peaks. Monitor PAGASA advisories for drought alerts.*",
                "💧 **Paano Nakakaapekto ang El Niño sa Water Supply?**\n\n"
                "Malaki ang bawas na dulot ng El Niño sa available na tubig — mula sa mga ilog at reservoir hanggang sa mga gripo sa bahay.\n\n"
                "🌊 **Surface Water:**\n"
                "• Mas kaunti ang tubig sa mga ilog at streams dahil sa reduced rainfall\n"
                "• Ang mga reservoir at dam ay nagiging critically low\n"
                "• Ang mga irrigation canal ay maaring matuyo sa peak dry months\n\n"
                "🪣 **Groundwater:**\n"
                "• Mas mabilis natutuyo ang mga balon dahil mas kaunting ulan ang pumapasok sa lupa para mag-recharge ng aquifers\n"
                "• Ang mga communities na umaasa sa deep wells ay maaaring makaranas ng supply cuts\n\n"
                "🏠 **Para sa mga Household sa Padre Garcia:**\n"
                "• Maaaring maging irregular ang availability ng tap water\n"
                "• Maaring mag-enforce ng water rationing ang local government\n"
                "• Maaaring kailangan pang lumayo ang mga residente para kumuha ng tubig\n\n"
                "⚡ **Secondary Impact:**\n"
                "• Mas kaunting kuryente ang nai-produce ng mga hydroelectric plant habang bumababa ang water levels, nagdudulot ng power shortages\n\n"
                "💡 *Tip: Mag-ipon ng malinis na tubig sa covered containers bago mag-peak ang dry season. I-monitor ang PAGASA advisories para sa drought alerts.*"
            )

    # CL2-4 & CL2-5. How does El Niño affect agriculture and crops?
    if any(k in clean_input for k in ["affect agriculture", "affect crops", "affect farming", "affect farmers", "el nino agriculture", "el nino crops", "el nino farming", "epekto sa agrikultura", "epekto sa mga pananim", "epekto sa magsasaka", "pananim", "crop failure", "ani", "palay", "mais", "agriculture", "farming", "nakakaapekto sa pananim"]):
        conversation_state["last_topic"] = "el_nino_agriculture"
        return L(
            "🌾 **How Does El Niño Affect Agriculture and Crops?**\n\n"
            "Agriculture is one of the sectors hardest hit by El Niño, especially in farming communities like Padre Garcia, Batangas.\n\n"
            "🌱 **Crops:**\n"
            "• Reduced rainfall means crops receive less water — leading to wilting, stunted growth, and crop failure\n"
            "• Rice and corn are especially vulnerable to prolonged dry spells\n"
            "• Quality and yield of harvests drop significantly\n"
            "• Planting schedules are disrupted — farmers may need to delay or skip planting seasons\n\n"
            "🚜 **Irrigation:**\n"
            "• Irrigation canals and water sources run low or dry up completely\n"
            "• Farmers dependent on rain-fed agriculture face the biggest losses\n"
            "• Cost of farming increases as water needs to be sourced elsewhere\n\n"
            "🐄 **Livestock:**\n"
            "• Animals suffer from heat stress and water shortage\n"
            "• Pastureland dries out, reducing available feed\n"
            "• Increased livestock deaths during severe events\n\n"
            "💡 **What farmers can do:**\n"
            "• Switch to drought-tolerant crop varieties (e.g. drought-resistant rice strains)\n"
            "• Use mulching to retain soil moisture longer\n"
            "• Adjust planting schedules to avoid peak dry months\n"
            "• Store water before the dry season begins",
            "🌾 **Paano Nakakaapekto ang El Niño sa Agriculture at mga Pananim?**\n\n"
            "Ang agriculture ay isa sa mga sektorong pinaka-apektado ng El Niño, lalo na sa mga farming communities tulad ng Padre Garcia, Batangas.\n\n"
            "🌱 **Mga Pananim:**\n"
            "• Ang reduced rainfall ay nangangahulugang mas kaunting tubig para sa crops — nagdudulot ng pag-wilting, stunted growth, at crop failure\n"
            "• Ang palay at mais ay especially vulnerable sa mahahabang dry spells\n"
            "• Malaki ang pagbaba ng quality at yield ng mga ani\n"
            "• Nadidisrupt ang mga planting schedules — maaaring kailangan ng mga farmers na mag-delay o lumaktaw ng planting seasons\n\n"
            "🚜 **Irrigation:**\n"
            "• Ang mga irrigation canal at water sources ay bumababa o tuluyang natutuyo\n"
            "• Ang mga farmers na umaasa sa rain-fed agriculture ang pinaka-malaki ang losses\n"
            "• Tumataas ang gastos sa pagsasaka dahil kailangan pang hanapin ang tubig sa ibang lugar\n\n"
            "🐄 **Livestock:**\n"
            "• Naghihirap ang mga hayop dahil sa heat stress at kakulangan ng tubig\n"
            "• Natutuyo ang pastureland, nababawasan ang available na pagkain\n"
            "• Mas maraming livestock deaths sa mga malubhang events\n\n"
            "💡 **Ano ang magagawa ng mga farmers:**\n"
            "• Lumipat sa drought-tolerant na varieties ng pananim (hal. drought-resistant na strain ng palay)\n"
            "• Gumamit ng mulching para mapanatili ang soil moisture nang mas matagal\n"
            "• I-adjust ang planting schedules para maiwasan ang peak dry months\n"
            "• Mag-ipon ng tubig bago magsimula ang dry season"
        )

    # CL2-6. Can El Niño cause higher temperatures?
    if any(k in clean_input for k in ["higher temperature", "cause higher temperature", "hotter", "mas mainit", "temperature rise", "tumataas ang temperatura", "nagdudulot ng init", "can el nino cause heat", "can el nino cause temperature"]):
        conversation_state["last_topic"] = "el_nino_temperature"
        return L(
            "🌡️ **Can El Niño Cause Higher Temperatures?**\n\n"
            "Yes — El Niño is directly associated with above-normal temperatures in the Philippines.\n\n"
            "When El Niño suppresses rainfall, the usual cooling effect of clouds and rain is reduced. This allows more solar radiation to heat the land surface, driving temperatures higher than usual.\n\n"
            "🔥 **What this means locally:**\n"
            "• Days feel noticeably hotter than in normal or La Niña years\n"
            "• Nights offer less relief due to retained daytime heat\n"
            "• Heat index (feels-like temperature) can be significantly higher than actual temperature\n\n"
            "📊 **From our Padre Garcia data (2010–2026):**\n"
            "• Average temperature during El Niño months is measurably higher than during Neutral or La Niña months\n"
            "• The hottest recorded months in our dataset coincide with El Niño events\n\n"
            "⚠️ **Health risks from higher temperatures:**\n"
            "• Heat exhaustion and heat stroke, especially during outdoor work\n"
            "• Dehydration risk increases for children and the elderly\n"
            "• Increased demand for water and electricity",
            "🌡️ **Nagdudulot Ba ng Mas Mataas na Temperature ang El Niño?**\n\n"
            "Oo — ang El Niño ay direktang associated sa above-normal na temperatures sa Pilipinas.\n\n"
            "Kapag binabawasan ng El Niño ang rainfall, nababawasan din ang usual na cooling effect ng clouds at ulan. Dahil dito, mas maraming solar radiation ang nagpapainit ng land surface, kaya mas tumataas ang temperature kaysa sa normal.\n\n"
            "🔥 **Ano ang ibig sabihin nito sa lokal:**\n"
            "• Mas kapansin-pansin na mainit ang mga araw kaysa sa normal o La Niña years\n"
            "• Mas konting ginhawa sa gabi dahil napanatili ang init mula sa araw\n"
            "• Ang heat index (feels-like temperature) ay maaaring mas mataas pa kaysa sa actual temperature\n\n"
            "📊 **Mula sa aming Padre Garcia data (2010–2026):**\n"
            "• Ang average temperature sa El Niño months ay measurably mas mataas kaysa sa Neutral o La Niña months\n"
            "• Ang mga pinaka-mainit na buwan sa aming dataset ay nagtutugma sa mga El Niño events\n\n"
            "⚠️ **Health risks mula sa mas mataas na temperature:**\n"
            "• Heat exhaustion at heat stroke, lalo na sa outdoor work\n"
            "• Tumataas ang dehydration risk para sa mga bata at matatanda\n"
            "• Lumalaki ang demand para sa tubig at kuryente"
        )

    # CL2-7. Can El Niño affect people's health?
    if any(k in clean_input for k in ["affect health", "affect people", "health effects", "health impact", "kalusugan", "epekto sa kalusugan", "malusog", "people health", "human health", "nakakaapekto sa kalusugan", "nakakaapekto sa tao"]):
        conversation_state["last_topic"] = "el_nino_health"
        return L(
            "🏥 **Can El Niño Affect People's Health?**\n\n"
            "Yes — El Niño creates conditions that directly increase health risks for communities, especially vulnerable groups like the elderly, children, and outdoor workers.\n\n"
            "🌡️ **Heat-Related Illnesses:**\n"
            "• **Heat exhaustion** — heavy sweating, weakness, cold or pale skin, nausea\n"
            "• **Heat stroke** — high body temperature (above 40°C), confusion, loss of consciousness (medical emergency)\n"
            "• **Dehydration** — especially dangerous for infants, children, and the elderly\n\n"
            "💧 **Water and Food Safety:**\n"
            "• Water scarcity forces communities to use unsafe water sources, increasing risk of waterborne diseases (diarrhea, typhoid, cholera)\n"
            "• Food insecurity rises as crops fail, affecting nutrition\n\n"
            "😮‍💨 **Respiratory Issues:**\n"
            "• Drier conditions increase dust and air particulates, worsening asthma and other respiratory conditions\n"
            "• Wildfire smoke during extreme droughts can cause serious respiratory harm\n\n"
            "🧑‍🌾 **Who is most at risk in Padre Garcia:**\n"
            "• Farmers and outdoor workers exposed to heat for long hours\n"
            "• Children and elderly with limited heat tolerance\n"
            "• Low-income households with limited access to clean water\n\n"
            "💡 *Stay hydrated, avoid outdoor exposure during peak heat hours (10am–3pm), and watch for signs of heat exhaustion.*",
            "🏥 **Nakakaapekto Ba ang El Niño sa Kalusugan ng mga Tao?**\n\n"
            "Oo — ang El Niño ay naglilikha ng mga kondisyon na direktang nagpapataas ng health risks para sa mga komunidad, lalo na sa mga vulnerable groups tulad ng matatanda, mga bata, at outdoor workers.\n\n"
            "🌡️ **Mga Heat-Related Illness:**\n"
            "• **Heat exhaustion** — sobrang pagpapawis, panghihina, malamig o maputlang balat, pagduduwal\n"
            "• **Heat stroke** — mataas na body temperature (higit sa 40°C), pagkalito, pagkakamatay ng malay (medical emergency)\n"
            "• **Dehydration** — especially mapanganib para sa mga sanggol, bata, at matatanda\n\n"
            "💧 **Water at Food Safety:**\n"
            "• Ang water scarcity ay nagpipilit sa mga komunidad na gumamit ng hindi ligtas na water sources, nagpapataas ng risk ng waterborne diseases (diarrhea, typhoid, cholera)\n"
            "• Tumataas ang food insecurity habang nasisira ang mga pananim, nakakaapekto sa nutrisyon\n\n"
            "😮‍💨 **Respiratory Issues:**\n"
            "• Ang mas dry na kondisyon ay nagpapataas ng dust at air particulates, nagpapalala ng asthma at iba pang respiratory conditions\n"
            "• Ang wildfire smoke sa panahon ng matinding drought ay maaaring magdulot ng seryosong respiratory harm\n\n"
            "🧑‍🌾 **Sino ang pinakamalaki ang risk sa Padre Garcia:**\n"
            "• Mga farmers at outdoor workers na nalantad sa init nang matagal\n"
            "• Mga bata at matatanda na may limitadong heat tolerance\n"
            "• Mga low-income households na may limitadong access sa malinis na tubig\n\n"
            "💡 *Manatiling hydrated, iwasan ang outdoor exposure sa peak heat hours (10am–3pm), at bantayan ang mga senyales ng heat exhaustion.*"
        )

    # CL2-8. Does El Niño affect rainfall in Padre Garcia?
    if any(k in clean_input for k in ["rainfall in padre garcia", "ulan sa padre garcia", "padre garcia rainfall", "el nino padre garcia", "rain in padre garcia", "rain sa padre garcia", "rainfall padre garcia"]):
        conversation_state["last_topic"] = "el_nino_padre_garcia_rain"
        return handle_el_nino_padre_garcia()

    # CL2-9. What happens when there is prolonged dry weather?
    if any(k in clean_input for k in ["prolonged dry", "prolonged drought", "matagal na tagtuyot", "extended dry", "dry weather", "walang ulan", "matagal na walang ulan", "dry season", "tag-araw", "tagtuyo", "what happens when dry", "prolonged dry spell"]):
        conversation_state["last_topic"] = "prolonged_dry"
        return L(
            "☀️ **What Happens When There Is Prolonged Dry Weather?**\n\n"
            "Prolonged dry weather — especially when triggered by El Niño — creates a cascade of problems for communities, farms, and the environment.\n\n"
            "💧 **Water Crisis:**\n"
            "• Rivers, streams, and reservoirs shrink or dry up completely\n"
            "• Wells and groundwater levels drop — some communities lose their water source entirely\n"
            "• Water rationing is often imposed by local government units\n\n"
            "🌾 **Agriculture Breakdown:**\n"
            "• Crops wilt and die without sufficient moisture — rice and corn are hit hardest\n"
            "• Livestock become weak due to heat and limited drinking water\n"
            "• Harvests fail, threatening food security for farming families\n\n"
            "🌳 **Environment:**\n"
            "• Soil dries out and cracks, increasing erosion when rains eventually return\n"
            "• Forest and grassland fires become more frequent\n"
            "• Ecosystems under stress — fish populations drop in low rivers\n\n"
            "🏘️ **Community Impact:**\n"
            "• Health risks rise — waterborne diseases, heat illness, malnutrition\n"
            "• Economic losses for farmers reduce household income\n"
            "• Power supply becomes unreliable as hydroelectric capacity drops\n\n"
            "📍 *In Padre Garcia, prolonged dry spells during El Niño years (like 2015–2016) were among the most severe on record, with rainfall dropping far below normal for several consecutive months.*",
            "☀️ **Ano ang Nangyayari Kapag May Matagal na Dry Weather?**\n\n"
            "Ang matagal na dry weather — lalo na kapag triggered ng El Niño — ay nagdudulot ng maraming problema sa mga komunidad, farm, at kalikasan.\n\n"
            "💧 **Water Crisis:**\n"
            "• Ang mga ilog, stream, at reservoir ay lumiit o tuluyang natutuyo\n"
            "• Bumababa ang levels ng mga balon at groundwater — ang ilang komunidad ay tuluyang nawawalan ng water source\n"
            "• Madalas na nagpapatupad ng water rationing ang local government units\n\n"
            "🌾 **Agriculture Breakdown:**\n"
            "• Ang mga pananim ay nalalanta at namamatay nang wala pang sapat na moisture — ang palay at mais ang pinaka-apektado\n"
            "• Humihina ang mga hayop dahil sa init at limitadong inumin\n"
            "• Nabibigo ang mga ani, nagbabanta sa food security ng mga farming families\n\n"
            "🌳 **Environment:**\n"
            "• Natutuyo at bumabaltak ang lupa, nagpapataas ng erosion kapag bumalik na ang ulan\n"
            "• Mas madalas na nagkakaroon ng forest at grassland fires\n"
            "• Stressed ang mga ecosystem — bumababa ang mga isda sa mababang ilog\n\n"
            "🏘️ **Community Impact:**\n"
            "• Tumataas ang health risks — waterborne diseases, heat illness, malnutrition\n"
            "• Ang economic losses ng mga farmers ay nagpapababa ng household income\n"
            "• Nagiging unreliable ang power supply habang bumababa ang hydroelectric capacity\n\n"
            "📍 *Sa Padre Garcia, ang mahahabang dry spells sa mga El Niño years (tulad ng 2015–2016) ay kabilang sa mga pinaka-matindi sa rekord, na bumaba ang rainfall nang malayo sa normal sa loob ng ilang magkakasunod na buwan.*"
        )

    # CL2-10. How does El Niño affect daily activities?
    if any(k in clean_input for k in ["daily activities", "everyday life", "daily life", "pang-araw-araw", "buhay araw-araw", "daily routine", "affect routine", "pamumuhay", "normal na buhay"]):
        conversation_state["last_topic"] = "el_nino_daily"
        return L(
            "🏘️ **How Does El Niño Affect Daily Activities?**\n\n"
            "El Niño doesn't just affect weather — it touches many parts of everyday life for residents of Padre Garcia and surrounding communities.\n\n"
            "🚿 **Water Use:**\n"
            "• Water supply may become irregular — shorter shower times, limited laundry days\n"
            "• Households collect and store water in advance\n"
            "• Less water available for cooking, cleaning, and sanitation\n\n"
            "🌾 **Farming Routines:**\n"
            "• Farmers adjust or cancel planting schedules\n"
            "• More time spent hauling water for crops and livestock\n"
            "• Income drops as harvests fail or yields are reduced\n\n"
            "🏫 **School and Work:**\n"
            "• Extreme heat makes outdoor work and travel more exhausting\n"
            "• Schools may adjust or cancel outdoor activities during heat advisories\n"
            "• Workers in construction, farming, and transport are most exposed\n\n"
            "🏠 **Home Life:**\n"
            "• Higher electricity bills as fans and air conditioners run longer\n"
            "• Increased spending on bottled water or water delivery\n"
            "• Family health concerns — monitoring for dehydration, heat exhaustion\n\n"
            "💡 *Small adjustments help: water early in the morning or late afternoon, avoid outdoor work during peak heat (10am–3pm), and keep emergency water supplies ready.*",
            "🏘️ **Paano Nakakaapekto ang El Niño sa Pang-Araw-Araw na Gawain?**\n\n"
            "Hindi lang weather ang naapektuhan ng El Niño — maraming bahagi ng pang-araw-araw na buhay ng mga residente ng Padre Garcia at mga kalapit na komunidad ang naaapektuhan din.\n\n"
            "🚿 **Paggamit ng Tubig:**\n"
            "• Ang water supply ay maaaring maging irregular — mas maikling oras ng paliligo, limitadong araw ng paglalaba\n"
            "• Ang mga household ay nag-iipon at nagtatago ng tubig nang maaga\n"
            "• Mas kaunting tubig para sa pagluluto, paglilinis, at sanitation\n\n"
            "🌾 **Mga Gawain sa Pagsasaka:**\n"
            "• Ang mga farmers ay nag-a-adjust o nagkaka-cancel ng mga planting schedules\n"
            "• Mas maraming oras ang ginagastos sa pag-angkat ng tubig para sa mga pananim at hayop\n"
            "• Bumababa ang kita habang nabibigo ang mga ani o nababawasan ang yields\n\n"
            "🏫 **Paaralan at Trabaho:**\n"
            "• Ang matinding init ay nagpapahirap ng outdoor work at paglalakbay\n"
            "• Ang mga paaralan ay maaaring mag-adjust o mag-cancel ng mga outdoor activities sa panahon ng heat advisories\n"
            "• Ang mga workers sa construction, pagsasaka, at transport ang pinakamalantad\n\n"
            "🏠 **Home Life:**\n"
            "• Mas mataas na kuryente bills habang mas matagal gumagana ang mga electric fan at aircon\n"
            "• Mas malaking gastos sa bottled water o water delivery\n"
            "• Mga alalahanin sa kalusugan ng pamilya — pagbabantay sa dehydration, heat exhaustion\n\n"
            "💡 *Nakakatulong ang maliliit na adjustments: magdilig nang umaga o hapon, iwasan ang outdoor work sa peak heat (10am–3pm), at laging magpanatili ng emergency water supply.*"
        )

    # CL-NEW-7. Does El Niño affect electricity consumption?
    if any(k in clean_input for k in [
        "electricity consumption", "affect electricity", "does el nino affect electricity",
        "electricity el nino", "power consumption", "kuryente consumption",
        "el nino kuryente", "nakakaapekto ba sa kuryente", "epekto sa kuryente",
        "power bill", "electricity bill", "kuryente bill", "higher electricity",
        "does el nino cause power shortage", "power shortage el nino", "brownout el nino",
        "pemadpad el nino", "brownout", "power outage el nino"
    ]):
        conversation_state["last_topic"] = "el_nino_electricity"
        return L(
            "⚡ **Does El Niño Affect Electricity Consumption?**\n\n"
            "Yes — El Niño impacts electricity in two important ways: it **increases demand** while simultaneously **reducing supply**.\n\n"
            "📈 **Why electricity demand goes UP:**\n"
            "• Higher temperatures drive households and businesses to use electric fans and air conditioners longer\n"
            "• More hours of cooling appliances running = higher electricity bills\n"
            "• Water pumps work harder as groundwater levels drop\n\n"
            "📉 **Why electricity supply goes DOWN:**\n"
            "• Many power plants in the Philippines rely on **hydroelectric generation** — which depends on rainfall to keep reservoir levels high\n"
            "• During El Niño, reduced rainfall lowers dam and reservoir levels, cutting hydroelectric output\n"
            "• This supply-demand imbalance can lead to **brownouts and power interruptions**\n\n"
            "🏘️ **Impact on Padre Garcia households:**\n"
            "• Electricity bills increase as appliances run longer to cope with heat\n"
            "• Risk of brownouts is higher during strong El Niño events\n"
            "• Communities without alternatives (solar, generator) are most vulnerable\n\n"
            "💡 *Tips: Use electric fans instead of AC when possible, unplug idle appliances, and conserve power during peak hours (12nn–8pm).*",
            "⚡ **Nakakaapekto Ba ang El Niño sa Paggamit ng Kuryente?**\n\n"
            "Oo — nakakaapekto ang El Niño sa kuryente sa dalawang mahalagang paraan: **pinapataas ang demand** habang sabay na **binabawasan ang supply**.\n\n"
            "📈 **Bakit tumataas ang demand ng kuryente:**\n"
            "• Ang mas mataas na temperatura ay nagdudulot sa mga household at negosyo na mas matagal na gamitin ang electric fan at aircon\n"
            "• Mas maraming oras ng paggamit ng cooling appliances = mas mataas na kuryente bills\n"
            "• Mas mahirap nagtatrabaho ang mga water pump habang bumababa ang groundwater levels\n\n"
            "📉 **Bakit bumababa ang supply ng kuryente:**\n"
            "• Maraming power plant sa Pilipinas ay umaasa sa **hydroelectric generation** — na nangangailangan ng ulan para mapanatiling mataas ang mga reservoir\n"
            "• Sa panahon ng El Niño, ang reduced rainfall ay nagpapababa ng mga dam at reservoir levels, binabawasan ang hydroelectric output\n"
            "• Ang imbalance na ito ng supply at demand ay maaaring magdulot ng **brownout at power interruptions**\n\n"
            "🏘️ **Epekto sa mga household sa Padre Garcia:**\n"
            "• Tumataas ang kuryente bills habang mas matagal gumagana ang mga appliance para makayanan ang init\n"
            "• Mas mataas ang panganib ng brownout sa panahon ng malakas na El Niño events\n"
            "• Ang mga komunidad na walang alternatibo (solar, generator) ang pinaka-vulnerable\n\n"
            "💡 *Tips: Gumamit ng electric fan sa halip na AC kung maaari, i-unplug ang mga idle na appliance, at magtipid ng kuryente sa peak hours (12nn–8pm).*"
        )

    # CL-NEW-8. Can El Niño affect the local economy?
    if any(k in clean_input for k in [
        "local economy", "affect economy", "affect the economy", "economic impact",
        "el nino economy", "economic effects", "el nino business",
        "epekto sa ekonomiya", "nakakaapekto sa ekonomiya", "ekonomiya",
        "economic loss", "financial impact", "economic damage",
        "livelihood", "kabuhayan", "epekto sa kabuhayan", "affect livelihood"
    ]):
        conversation_state["last_topic"] = "el_nino_economy"
        return L(
            "💰 **Can El Niño Affect the Local Economy?**\n\n"
            "Yes — El Niño has a significant ripple effect on the local economy, especially in agricultural communities like Padre Garcia, Batangas.\n\n"
            "🌾 **Agriculture (hardest hit):**\n"
            "• Crop failures reduce farm income — rice, corn, and vegetable yields drop sharply\n"
            "• Livestock losses add to farmer financial strain\n"
            "• Farmers may take on debt to survive until the next season\n\n"
            "🐟 **Fisheries:**\n"
            "• Changes in ocean temperature affect fish populations and migration patterns\n"
            "• Fisherfolk experience reduced catch, lowering income\n\n"
            "💧 **Water-dependent businesses:**\n"
            "• Food processing, laundry services, and other water-intensive businesses face disruption\n"
            "• Tourism in water-dependent areas (lakes, rivers) may decline\n\n"
            "⚡ **Energy costs:**\n"
            "• Higher electricity bills increase operating costs for households and businesses\n"
            "• Small businesses with narrow margins are especially vulnerable\n\n"
            "🏘️ **Community-wide impact:**\n"
            "• Reduced agricultural income means less money circulating in the local economy\n"
            "• Government relief spending increases — diverting funds from development\n"
            "• Food prices may rise as local supply shrinks\n\n"
            "📊 *In Padre Garcia, which is a farming community in Batangas, strong El Niño events directly translate to lower household incomes, reduced farm productivity, and heavier reliance on government assistance.*",
            "💰 **Nakakaapekto Ba ang El Niño sa Lokal na Ekonomiya?**\n\n"
            "Oo — ang El Niño ay may malaking ripple effect sa lokal na ekonomiya, lalo na sa mga agricultural communities tulad ng Padre Garcia, Batangas.\n\n"
            "🌾 **Agrikultura (pinaka-apektado):**\n"
            "• Ang mga crop failure ay nagpapababa ng kita sa pagsasaka — malaki ang pagbaba ng ani ng palay, mais, at gulay\n"
            "• Ang mga livestock losses ay nagdaragdag sa financial strain ng mga magsasaka\n"
            "• Ang mga magsasaka ay maaaring mangutang para mabuhay hanggang sa susunod na season\n\n"
            "🐟 **Pangingisda:**\n"
            "• Ang mga pagbabago sa temperatura ng dagat ay nakakaapekto sa mga populasyon ng isda at kanilang paggalaw\n"
            "• Ang mga mangingisda ay nakakaranas ng mas kaunting huli, nagpapababa ng kita\n\n"
            "💧 **Mga negosyong umaasa sa tubig:**\n"
            "• Ang food processing, laundry services, at iba pang water-intensive na negosyo ay nakakaranas ng disruption\n"
            "• Ang turismo sa mga lugar na umaasa sa tubig (lawa, ilog) ay maaaring bumaba\n\n"
            "⚡ **Gastos sa kuryente:**\n"
            "• Ang mas mataas na kuryente bills ay nagpapataas ng operating costs ng mga household at negosyo\n"
            "• Ang mga maliliit na negosyo na may maliit na margin ay especially vulnerable\n\n"
            "🏘️ **Community-wide na epekto:**\n"
            "• Ang nabawasang kita sa agrikultura ay nangangahulugang mas kaunting pera ang umiikot sa lokal na ekonomiya\n"
            "• Tumataas ang government relief spending — nagdi-divert ng pondo mula sa development\n"
            "• Maaaring tumaas ang presyo ng pagkain habang lumiit ang lokal na supply\n\n"
            "📊 *Sa Padre Garcia, na isang farming community sa Batangas, ang mga malakas na El Niño events ay direktang nagdudulot ng mas mababang household income, nabawasang farm productivity, at mas malakas na pag-asa sa tulong ng gobyerno.*"
        )

    # CL-NEW-9. Why does El Niño cause dry conditions?
    if any(k in clean_input for k in [
        "why does el nino cause dry", "why does el nino cause dryness",
        "why el nino cause dry conditions", "why el nino makes it dry",
        "why is it dry during el nino", "reason for dry weather el nino",
        "bakit tuyo sa el nino", "bakit may tagtuyot sa el nino",
        "bakit walang ulan sa el nino", "sanhi ng tagtuyot sa el nino",
        "why el nino dry", "why does el nino reduce rain", "why less rain el nino"
    ]):
        conversation_state["last_topic"] = "el_nino_dry_cause"
        return L(
            "☀️ **Why Does El Niño Cause Dry Conditions?**\n\n"
            "El Niño creates dry conditions in the Philippines through a chain of atmospheric and oceanic changes:\n\n"
            "🌬️ **Step 1 — Trade winds weaken**\n"
            "Normally, strong trade winds push warm, moist air from the Pacific toward Southeast Asia — fueling rainfall. During El Niño, these winds weaken, reducing the moisture being delivered to the region.\n\n"
            "🌡️ **Step 2 — Warm water shifts eastward**\n"
            "The warm pool of water that usually sits near the Philippines and Southeast Asia moves toward the central Pacific. Rain-producing storm systems follow the warmth — so rainfall shifts away from the Philippines.\n\n"
            "☁️ **Step 3 — Less cloud formation over the Philippines**\n"
            "With less warm moisture in the air above the Philippines, fewer clouds form. Less cloud cover means less rain — and more direct sunlight heating the ground.\n\n"
            "🌾 **Step 4 — The dry season stretches longer**\n"
            "The normal dry season (roughly December to May) extends further and becomes more intense. In strong El Niño years, rainfall can stay below normal for 6–12 months straight.\n\n"
            "📊 *Our Padre Garcia data shows this clearly — during documented El Niño periods, monthly rainfall drops significantly below the historical average, sometimes for multiple consecutive months.*\n\n"
            "💡 *This is why farmers, water managers, and communities are advised to prepare water storage before El Niño peaks.*",
            "☀️ **Bakit Nagdudulot ng Dry Conditions ang El Niño?**\n\n"
            "Nagdudulot ng dry conditions ang El Niño sa Pilipinas sa pamamagitan ng isang chain ng mga atmospheric at oceanic na pagbabago:\n\n"
            "🌬️ **Step 1 — Humihina ang trade winds**\n"
            "Normally, ang malakas na trade winds ay nagtutulak ng mainit, mamasa-masang hangin mula sa Pacific patungo sa Southeast Asia — nagdudulot ng ulan. Sa panahon ng El Niño, humihina ang mga winds na ito, binabawasan ang moisture na naihahatid sa rehiyon.\n\n"
            "🌡️ **Step 2 — Lumilipat ang mainit na tubig patungong silangan**\n"
            "Ang mainit na pool ng tubig na karaniwang nakaupong malapit sa Pilipinas at Southeast Asia ay lumilipat patungo sa central Pacific. Ang mga rain-producing na storm system ay sumusunod sa init — kaya ang ulan ay lumilipat palayo sa Pilipinas.\n\n"
            "☁️ **Step 3 — Mas kaunting cloud formation sa Pilipinas**\n"
            "Dahil mas kaunting mainit na moisture sa hangin sa itaas ng Pilipinas, mas kaunting ulap ang nababago. Mas kaunting ulap = mas kaunting ulan — at mas direktang sikat ng araw ang nagpapainit ng lupa.\n\n"
            "🌾 **Step 4 — Mas matagal ang dry season**\n"
            "Ang normal na dry season (mga Disyembre hanggang Mayo) ay nagpapalawak at nagiging mas matindi. Sa malakas na El Niño years, ang rainfall ay maaaring manatiling below normal sa loob ng 6–12 magkakasunod na buwan.\n\n"
            "📊 *Malinaw itong makikita sa aming Padre Garcia data — sa mga naitala na El Niño periods, malaki ang pagbaba ng monthly rainfall mula sa historical average, minsan nang ilang magkakasunod na buwan.*\n\n"
            "💡 *Kaya naman pinapayuhan ang mga magsasaka, water managers, at komunidad na maghanda ng water storage bago mag-peak ang El Niño.*"
        )

    # 5a. General effects of El Niño (broad fallback — all specific "epekto sa X" cases are handled above)
    if any(k in clean_input for k in ["what are its effects", "effects of el nino", "impacts of el nino", "consequences", "what does el nino do", "how it affect", "affect the environment", "ano ang epekto", "epekto ng el nino", "pinsala ng el nino", "pagkakaroon ng epekto"]):
        return L(
            "⚠️ **Effects of El Niño:**\n\n"
            "El Niño disrupts normal weather patterns across the Philippines and the world:\n\n"
            "🌾 **Agriculture:**\n"
            "• Prolonged dry spells cause crop failure and reduced yields\n"
            "• Irrigation water sources dry up, affecting planting schedules\n"
            "• Livestock suffer from heat stress and water shortage\n\n"
            "💧 **Water Resources:**\n"
            "• Rivers, reservoirs, and groundwater levels drop significantly\n"
            "• Domestic water supply becomes scarce in many communities\n\n"
            "🌡️ **Environment:**\n"
            "• Above-normal temperatures increase wildfire risks\n"
            "• Coral bleaching occurs in coastal marine ecosystems\n"
            "• Soil moisture depletion accelerates land degradation\n\n"
            "🏥 **Health & Community:**\n"
            "• Heat-related illnesses increase, especially for the elderly\n"
            "• Food and water insecurity affects vulnerable households\n"
            "• Power shortages occur as hydroelectric capacity drops",
            "⚠️ **Mga Epekto ng El Niño:**\n\n"
            "Nadidisrupt ng El Niño ang normal na weather patterns sa buong Pilipinas at mundo:\n\n"
            "🌾 **Agriculture:**\n"
            "• Mahahabang dry spells nagdudulot ng crop failure at nababawasang ani\n"
            "• Natutuyo ang mga water sources para sa irrigation, naaapektuhan ang planting schedules\n"
            "• Naghihirap ang livestock dahil sa heat stress at water shortage\n\n"
            "💧 **Water Resources:**\n"
            "• Mahina na ang levels ng mga ilog, reservoir, at groundwater\n"
            "• Nawawala ang water supply sa maraming communities\n\n"
            "🌡️ **Environment:**\n"
            "• Above-average na temperature nagpapataas ng wildfire risk\n"
            "• Nangyayari ang coral bleaching sa mga coastal marine ecosystems\n"
            "• Soil moisture depletion nagpapabilis ng land degradation\n\n"
            "🏥 **Health at Community:**\n"
            "• Lumalaki ang heat-related illnesses, lalo na sa mga matatanda\n"
            "• Food at water insecurity naaapektuhan ang mga vulnerable households\n"
            "• Nagkakaroon ng power shortages habang bumababa ang hydroelectric capacity"
        )

    # 5b. What happens during El Niño specifically in Padre Garcia 
    if any(k in clean_input for k in [
        "what happens during", "what happens when",
        "pag el nino", "pag may el nino",
        "el nino in padre garcia", "el nino sa padre garcia",
        "anong nangyayari kapag el nino",
        "anong nangyayari sa el nino"
    ]):
        if conversation_state.get("lang") == "tl":
            return (
                "📍 **Ano ang Maaaring Mangyari sa Padre Garcia Kapag may El Niño?**\n\n"
                "Ang El Niño ay maaaring magdulot ng mas mainit at mas tuyong panahon "
                "dahil maaaring bumaba ang dami ng ulan. Narito ang ilang posibleng epekto:\n\n"
                "💧 **Kakulangan sa Tubig** - Maaaring bumaba ang suplay ng tubig sa mga balon at iba pang pinagkukunan ng tubig.\n\n"
                "🌾 **Epekto sa Agrikultura**\n"
                "Maaaring matuyo ang lupa, maapektuhan ang mga pananim, at mahirapan ang mga magsasaka.\n\n"
                "🌡️ **Matinding Init**\n"
                "Maaaring tumaas ang panganib ng dehydration, heat exhaustion, at iba pang problemang dulot ng init.\n\n"
                "🐄 **Epekto sa mga Alagang Hayop**\n"
                "Maaaring ma-stress ang mga alagang hayop dahil sa mataas na temperatura at kakulangan sa tubig.\n\n"
                "Tandaan: Posibleng epekto ang mga ito. Maaaring mag-iba ang aktuwal na kondisyon sa Padre Garcia "
                "depende sa panahon at lokal na sitwasyon. Sundin ang mga opisyal na abiso ng PAGASA at lokal na pamahalaan."
            )

        return (
            "📍 **What Happens During El Niño in Padre Garcia?**\n\n"
            "El Niño may bring hotter and drier conditions because rainfall can decrease. "
            "Here are some possible effects:\n\n"
            "💧 **Water Shortage**\n"
            "Water supplies from wells and other sources may decrease.\n\n"
            "🌾 **Agricultural Impacts**\n"
            "Soil may become dry, crops may be affected, and farmers may face difficulties.\n\n"
            "🌡️ **Extreme Heat**\n"
            "The risk of dehydration, heat exhaustion, and other heat-related illnesses may increase.\n\n"
            "🐄 **Effects on Livestock**\n"
            "Animals may experience heat stress and have difficulty accessing enough water.\n\n"
            "These are possible effects, not guaranteed conditions. Actual conditions in Padre Garcia "
            "may vary depending on the weather and local situation. Follow official PAGASA advisories "
            "and local government announcements."
        )

    # 6. Why should we know
    if any(k in clean_input for k in ["why we should know", "why should we know", "importance of knowing", "why is it important", "why care", "bakit mahalaga", "bakit kailangan natin malaman", "bakit kailangan alamin", "bakit dapat natin malaman"]):
        return L(
            "💡 **Why Should Padre Garcia Residents Know About El Niño?**\n"
            "• **Early Preparation:** Families and farmers can store water and plan ahead before dry spells hit.\n"
            "• **Protect Livelihoods:** Farmers can switch to drought-tolerant crops or adjust planting schedules.\n"
            "• **Community Safety:** LGUs and barangay officials can act early and manage resources proactively.\n"
            "• **Health Protection:** Awareness helps residents avoid heat-related illnesses during El Niño months.",
            "💡 **Bakit Dapat Malaman ng mga Residents ng Padre Garcia ang El Niño?**\n"
            "• **Early Preparation:** Maaaring mag-impok ng water at magplano nang mas maaga ang mga pamilya at farmers bago dumating ang dry spells.\n"
            "• **Protect Livelihoods:** Maaaring lumipat sa drought-tolerant crops o ayusin ang planting schedules.\n"
            "• **Community Safety:** Maaaring maagap kumilos ang LGU at mga barangay officials at pamahalaan ang resources nang proactive.\n"
            "• **Health Protection:** Ang awareness ay nakakatulong sa mga residente na iwasan ang heat-related illnesses sa panahon ng El Niño."
        )

    # 7. What should I do during El Niño?
    if any(k in clean_input for k in ["what should i do", "what to do", "how to prepare", "how do i prepare", "how can i prepare", "paano maghanda", "maghanda", "preparedness", "prepare for el nino", "tips", "advice", "ano ang gagawin", "gagawin during el nino", "dapat gawin", "paano maghanda sa el nino", "i prepare", "prepare"]):
        # Don't intercept heat exhaustion first aid questions
        if any(k in clean_input for k in ["heat exhaustion", "heat stroke", "someone experiences", "first aid", "extreme heat", "during extreme heat", "programs", "information about", "where can i get", "local programs", "government programs", "lgu programs", "water shortage", "water shortages", "kapag walang tubig", "tubig kulang"]):
            pass
        else:
            return L(
                "🛡️ **What Should You Do During El Niño?**\n\n"
                "Here are general preparedness tips for households and communities:\n\n"
                "💧 **Water Conservation:**\n"
                "• Store clean water in covered containers before dry season peaks\n"
                "• Fix leaking pipes and faucets to avoid wastage\n"
                "• Reuse water where possible (e.g. wash water for plants)\n"
                "• Limit non-essential water use like car washing\n\n"
                "🌾 **For Farmers:**\n"
                "• Switch to drought-tolerant crop varieties if possible\n"
                "• Adjust planting schedules to avoid peak dry months\n"
                "• Use mulching to retain soil moisture longer\n"
                "• Check irrigation systems and water storage before the dry season\n\n"
                "🏠 **For Households:**\n"
                "• Stock up on essential food and supplies early\n"
                "• Keep family members — especially children and elderly — hydrated\n"
                "• Avoid outdoor activities during the hottest parts of the day\n"
                "• Watch for heat exhaustion signs: dizziness, excessive sweating, weakness\n\n"
                "📢 **Stay Informed:**\n"
                "• Monitor weather updates from PAGASA regularly\n"
                "• Follow advisories from your local barangay and MDRRMO\n\n"
                "⚠️ *These are general guidelines. For protocols specific to Padre Garcia, "
                "coordinate with your local Municipal Disaster Risk Reduction and Management Office (MDRRMO).*",
                "🛡️ **Ano ang Dapat Gawin Sa Panahon ng El Niño?**\n\n"
                "Narito ang mga general tips sa preparedness para sa mga households at community:\n\n"
                "💧 **Water Conservation:**\n"
                "• Mag-impok ng malinis na water sa mga covered containers bago sumukdol ang dry season\n"
                "• Ayusin ang mga tumutulong pipes at gripo upang maiwasan ang pag-aaksaya\n"
                "• Gamitin muli ang water kung maaari (hal. wash water para sa mga halaman)\n"
                "• Limitahan ang non-essential water use tulad ng paghuhugas ng sasakyan\n\n"
                "🌾 **Para sa mga Farmers:**\n"
                "• Lumipat sa drought-tolerant crop varieties kung maaari\n"
                "• Ayusin ang planting schedules upang maiwasan ang pinakamasinsing dry months\n"
                "• Gumamit ng mulching upang mapanatili ang soil moisture nang mas matagal\n"
                "• Suriin ang mga irrigation systems at water storage bago ang dry season\n\n"
                "🏠 **Para sa mga Households:**\n"
                "• Mag-impok ng essential food at supplies nang mas maaga\n"
                "• Panatilihing hydrated ang mga miyembro ng pamilya — lalo na ang mga bata at matatanda\n"
                "• Iwasan ang mga outdoor activities sa pinakamainit na bahagi ng araw\n"
                "• Bantayan ang mga signs ng heat exhaustion: dizziness, excessive sweating, weakness\n\n"
                "📢 **Manatiling Informed:**\n"
                "• Regular na subaybayan ang mga weather updates mula sa PAGASA\n"
                "• Sundin ang mga advisories mula sa inyong barangay at MDRRMO\n\n"
                "⚠️ *Ito ay mga general guidelines lamang. Para sa mga protocols na specific sa Padre Garcia, "
                "makipag-ugnayan sa inyong Municipal Disaster Risk Reduction and Management Office (MDRRMO).*"
            )

    # CL3-2. How can I conserve water?
    if any(k in clean_input for k in ["conserve water", "save water", "water conservation", "tipid sa tubig", "makatipid ng tubig", "pag-iingat ng tubig", "how to save water", "how to conserve", "pagtipid ng tubig", "save safe water"]):
        conversation_state["last_topic"] = "water_conservation"
        return L(
            "💧 **How Can I Conserve Water During El Niño?**\n\n"
            "Water conservation is critical during El Niño when supply runs low. Here are practical steps:\n\n"
            "🚿 **At Home:**\n"
            "• Fix leaking faucets and pipes immediately — even a small drip wastes liters a day\n"
            "• Take shorter showers and turn off the tap while soaping\n"
            "• Reuse water — water used for washing vegetables can go to plants\n"
            "• Only run washing machines and dishwashers with full loads\n"
            "• Use a basin instead of running water when washing dishes\n\n"
            "🪣 **Storage:**\n"
            "• Store clean water in covered containers before the dry season peaks\n"
            "• Use large drums or water tanks to stockpile supply\n"
            "• Collect rainwater during brief rain events for non-drinking uses\n\n"
            "🌾 **For Farming:**\n"
            "• Water crops early morning or late afternoon to reduce evaporation\n"
            "• Use mulching to retain soil moisture longer\n"
            "• Switch to drip irrigation if available — uses up to 50% less water than flooding\n"
            "• Prioritize watering young and high-value crops first\n\n"
            "🏘️ **Community:**\n"
            "• Follow any water rationing schedules set by your local government\n"
            "• Report water leaks in public pipes to your LGU immediately",
            "💧 **Paano Makakatipid ng Tubig sa Panahon ng El Niño?**\n\n"
            "Kritikal ang water conservation sa panahon ng El Niño kapag mababa na ang supply. Narito ang mga praktikal na hakbang:\n\n"
            "🚿 **Sa Bahay:**\n"
            "• Ayusin agad ang mga leaking faucet at pipes — kahit maliit na patak, nakakatipid ng mga litro sa isang araw\n"
            "• Mag-shower nang mas maikli at isara ang gripo habang nagso-soap\n"
            "• I-reuse ang tubig — ang tubig na ginamit sa paghuhugas ng gulay ay maaaring gamitin sa halaman\n"
            "• I-run lang ang washing machine at dishwasher kapag puno na\n"
            "• Gumamit ng palanggana imbes na flowing water sa paghuhugas ng pinggan\n\n"
            "🪣 **Storage:**\n"
            "• Mag-ipon ng malinis na tubig sa mga covered containers bago mag-peak ang dry season\n"
            "• Gumamit ng malalaking drum o water tank para mag-stockpile ng supply\n"
            "• Mag-collect ng rainwater sa maiikling ulan para sa non-drinking uses\n\n"
            "🌾 **Para sa Pagsasaka:**\n"
            "• Magdilig nang umaga o hapon para mabawasan ang evaporation\n"
            "• Gumamit ng mulching para mapanatili ang soil moisture nang mas matagal\n"
            "• Lumipat sa drip irrigation kung available — gumagamit ng hanggang 50% na mas kaunting tubig\n"
            "• Unahin ang pagdidilig ng mga batang halaman at high-value crops\n\n"
            "🏘️ **Komunidad:**\n"
            "• Sundin ang anumang water rationing schedules na itinakda ng inyong local government\n"
            "• Iulat agad sa inyong LGU ang mga water leaks sa mga pampublikong pipes"
        )

    # CL3-3. What should I do during extreme heat?
    if any(k in clean_input for k in ["extreme heat", "during heat", "sobrang init", "matinding init", "heat wave", "mainit na panahon", "kapag mainit", "sa sobrang init", "what to do heat", "what should i do during extreme heat", "during hot weather", "sa matinding init", "what to do during extreme"]):
        # Don't intercept medical-seeking questions — let them fall to CL3-11
        if any(k in clean_input for k in ["seek medical", "medical assistance", "see a doctor", "hospital", "when should i", "when to seek", "medical help"]):
            pass
        else:
            conversation_state["last_topic"] = "extreme_heat_action"
            return L(
                "🌡️ **What Should I Do During Extreme Heat?**\n\n"
                "During El Niño, temperatures in Padre Garcia and nearby areas can rise significantly. Here's how to stay safe:\n\n"
                "🏠 **Stay Cool:**\n"
                "• Stay indoors during the hottest hours — **10am to 3pm**\n"
                "• Use electric fans or air conditioning if available\n"
                "• Keep curtains and blinds closed to block sunlight\n"
                "• Cool down with damp cloths on the neck, wrists, and forehead\n\n"
                "💧 **Stay Hydrated:**\n"
                "• Drink water regularly — **at least 8–10 glasses a day**, even if you don't feel thirsty\n"
                "• Avoid alcohol, caffeine, and sugary drinks — they increase dehydration\n"
                "• Eat hydrating foods: fruits, vegetables, soups\n\n"
                "👕 **Dress Appropriately:**\n"
                "• Wear loose, light-colored, breathable clothing\n"
                "• Use a hat or umbrella when going outside\n"
                "• Apply sunscreen to exposed skin\n\n"
                "⚠️ **Watch Out For:**\n"
                "• Heavy outdoor work during peak heat — take breaks in the shade every 30 minutes\n"
                "• Signs of heat exhaustion: dizziness, heavy sweating, weakness, nausea\n"
                "• Children and elderly are most at risk — check on them frequently\n\n"
                "📢 *Monitor PAGASA heat advisories and follow your barangay's announcements.*",
                "🌡️ **Ano ang Dapat Gawin Sa Panahon ng Matinding Init?**\n\n"
                "Sa panahon ng El Niño, malaki ang pagtaas ng temperature sa Padre Garcia at mga kalapit na lugar. Narito kung paano manatiling ligtas:\n\n"
                "🏠 **Manatiling Malamig:**\n"
                "• Manatili sa loob ng bahay sa pinaka-mainit na oras — **10am hanggang 3pm**\n"
                "• Gumamit ng electric fan o aircon kung available\n"
                "• Isara ang mga kurtina at blinds para harangan ang sikat ng araw\n"
                "• Maglagay ng basang tela sa leeg, pulso, at noo para lumamig\n\n"
                "💧 **Manatiling Hydrated:**\n"
                "• Regular na uminom ng tubig — **hindi bababa sa 8–10 baso sa isang araw**, kahit hindi ka nauuhaw\n"
                "• Iwasan ang alak, kape, at matamis na inumin — nagpapataas ng dehydration\n"
                "• Kumain ng mga pagkaing may mataas na tubig: prutas, gulay, sopas\n\n"
                "👕 **Mag-damit nang Naaangkop:**\n"
                "• Magsuot ng maluwag, maliwanag na kulay, at breathable na damit\n"
                "• Gumamit ng sombrero o payong kapag lumabas\n"
                "• Mag-apply ng sunscreen sa mga exposed na bahagi ng katawan\n\n"
                "⚠️ **Mag-ingat Sa:**\n"
                "• Mabigat na outdoor work sa peak heat — magpahinga sa lilim bawat 30 minuto\n"
                "• Mga senyales ng heat exhaustion: pagkahilo, sobrang pawis, panghihina, pagduduwal\n"
                "• Ang mga bata at matatanda ang pinaka-vulnerable — palaging suriin sila\n\n"
                "📢 *I-monitor ang PAGASA heat advisories at sundin ang mga anunsyo ng inyong barangay.*"
            )

    # CL3-4. How can I prevent dehydration?
    if any(k in clean_input for k in ["prevent dehydration", "avoid dehydration", "maiwasan ang dehydration", "hindi ma-dehydrate", "dehydration prevention", "iwasan ang uhaw", "how to stay hydrated", "manatiling hydrated"]):
        conversation_state["last_topic"] = "dehydration_prevention"
        return L(
            "💧 **How Can I Prevent Dehydration During El Niño?**\n\n"
            "Dehydration is one of the most common health risks during hot, dry El Niño months. Here's how to prevent it:\n\n"
            "🥤 **Drink Enough Water:**\n"
            "• Aim for **8–10 glasses (2 liters) of water per day** — more if working outdoors\n"
            "• Don't wait until you're thirsty — thirst is already a sign of mild dehydration\n"
            "• Keep a water bottle with you at all times\n\n"
            "🍉 **Eat Hydrating Foods:**\n"
            "• Fruits: watermelon, oranges, cucumber, pineapple\n"
            "• Vegetables: lettuce, celery, tomatoes\n"
            "• Soups and broths also help maintain fluid levels\n\n"
            "❌ **Avoid These:**\n"
            "• Alcohol and caffeinated drinks (coffee, energy drinks) — increase fluid loss\n"
            "• Salty and processed foods — they make the body retain less water\n"
            "• Skipping meals — your body loses water through digestion even when not drinking\n\n"
            "⚠️ **Signs you may already be dehydrated:**\n"
            "• Dark yellow urine (pale yellow = well-hydrated)\n"
            "• Dry mouth and lips\n"
            "• Headache or dizziness\n"
            "• Feeling unusually tired or irritable\n\n"
            "👶 **Extra care for vulnerable groups:**\n"
            "• Children need frequent water reminders — they may not notice thirst\n"
            "• Elderly people have reduced thirst sensation — schedule water intake\n"
            "• Outdoor workers should drink **one cup every 20 minutes** during heavy work",
            "💧 **Paano Maiwasan ang Dehydration sa Panahon ng El Niño?**\n\n"
            "Ang dehydration ay isa sa mga pinakakaraniwang health risks sa mainit at tuyong El Niño months. Narito kung paano ito maiwasan:\n\n"
            "🥤 **Uminom ng Sapat na Tubig:**\n"
            "• Target na **8–10 baso (2 litro) ng tubig bawat araw** — mas marami pa kapag nagtatrabaho sa labas\n"
            "• Huwag maghintay na mauhaw — ang pagkauhaw ay senyales na ng mild dehydration\n"
            "• Magdala ng water bottle sa lahat ng oras\n\n"
            "🍉 **Kumain ng mga Pagkaing may Mataas na Tubig:**\n"
            "• Prutas: pakwan, kahel, pipino, pinya\n"
            "• Gulay: litsugas, kintsay, kamatis\n"
            "• Ang mga sopas at sabaw ay nakakatulong din sa pagpapanatili ng fluid levels\n\n"
            "❌ **Iwasan Ito:**\n"
            "• Alak at caffeinated drinks (kape, energy drinks) — nagpapataas ng fluid loss\n"
            "• Maalat at processed na pagkain — kaya ng katawan na mag-retain ng mas kaunting tubig\n"
            "• Praning ng pagkain — nawawalan din ng tubig ang katawan sa digestion kahit hindi umiinom\n\n"
            "⚠️ **Mga senyales na maaaring dehydrated ka na:**\n"
            "• Matingkad na dilaw na ihi (maliwanag na dilaw = well-hydrated)\n"
            "• Tuyong bibig at labi\n"
            "• Sakit ng ulo o pagkahilo\n"
            "• Pag-feeling na unusually pagod o irritable\n\n"
            "👶 **Extra care para sa mga vulnerable:**\n"
            "• Ang mga bata ay kailangang paalalahanan madalas na uminom — maaaring hindi nila mapansin ang pagkauhaw\n"
            "• Ang mga matatanda ay may reduced thirst sensation — mag-iskedyul ng pag-inom ng tubig\n"
            "• Ang mga outdoor workers ay dapat uminom ng **isang baso bawat 20 minuto** sa mabigat na trabaho"
        )

    # CL3-5. What are the signs of heat exhaustion?/ What is heat exhaustion?
    if any(k in clean_input for k in ["signs of heat exhaustion", "symptoms of heat exhaustion", "heat exhaustion signs", "heat exhaustion symptoms", "senyales ng heat exhaustion", "sintomas ng heat exhaustion", "signs of heat stroke", "symptoms of heat stroke", "how do i know heat exhaustion", "recognize heat exhaustion", "heat exhaustion"]):
        conversation_state["last_topic"] = "heat_exhaustion_signs"
        return L(
            
            "**Heat exhaustion** is a serious condition that can develop quickly during extreme heat. Recognizing the signs early can prevent it from becoming a life-threatening **heat stroke**.\n\n"
            "⚠️ **Signs of Heat Exhaustion:**\n"
            "• Heavy, excessive sweating\n"
            "• Cool, pale, or clammy skin\n"
            "• Fast but weak pulse\n"
            "• Nausea or vomiting\n"
            "• Muscle cramps\n"
            "• Tiredness, weakness\n"
            "• Dizziness or fainting\n"
            "• Headache\n\n"
            "🚨 **When it becomes Heat Stroke (Medical Emergency):**\n"
            "If the person shows these signs, **call for medical help immediately:**\n"
            "• Body temperature above **40°C (104°F)**\n"
            "• Hot, red, dry, or damp skin\n"
            "• Rapid, strong pulse\n"
            "• Confusion, slurred speech, or loss of consciousness\n"
            "• Stopping sweating despite the heat\n\n"
            "💡 *Heat exhaustion can escalate to heat stroke within minutes. Do not ignore early signs.*\n\n",
            "🌡️ **Ano ang mga Senyales ng Heat Exhaustion?**\n\n"
            "Ang heat exhaustion ay isang seryosong kondisyon na mabilis na maaaring mangyari sa matinding init. Ang maagang pagkilala sa mga senyales nito ay makakapigil sa pagiging life-threatening na **heat stroke**.\n\n"
            "⚠️ **Mga Senyales ng Heat Exhaustion:**\n"
            "• Sobrang, excessive na pagpapawis\n"
            "• Malamig, maputla, o malagkit na balat\n"
            "• Mabilis ngunit mahinang pulso\n"
            "• Pagduduwal o pagsusuka\n"
            "• Muscle cramps\n"
            "• Pagod, panghihina\n"
            "• Pagkahilo o pagkamatay ng malay\n"
            "• Sakit ng ulo\n\n"
            "🚨 **Kapag naging Heat Stroke na (Medical Emergency):**\n"
            "Kung nagpapakita ng mga sumusunod na senyales ang tao, **humingi agad ng medikal na tulong:**\n"
            "• Body temperature na higit sa **40°C (104°F)**\n"
            "• Mainit, pula, tuyong o mamasa-masang balat\n"
            "• Mabilis, malakas na pulso\n"
            "• Pagkalito, hindi maliwanag na pagsasalita, o pagkawala ng malay\n"
            "• Paghinto ng pagpapawis kahit mainit pa\n\n"
            "💡 *Ang heat exhaustion ay maaaring lumala sa heat stroke sa loob ng ilang minuto. Huwag balewalain ang mga maagang senyales.*\n\n"
        )

    # CL-NEW-10. What is heatstroke?
    # 1. HEATSTROKE (Priority: Medical Emergency)
    if any(k in clean_input for k in [
        "heatstroke", "heat stroke", "may heatstroke", "first aid heatstroke",
        "first aid heat stroke", "treat heatstroke", "what to do if someone has heatstroke",
        "what to do heatstroke", "ano ang dapat gawin sa heatstroke", "gamot sa heatstroke"
    ]):
        return handle_heatstroke_emergency()

    # 2. HEAT EXHAUSTION (First Aid & Signs)
    if any(k in clean_input for k in [
        "heat exhaustion", "heat exhaustion first aid", "first aid heat exhaustion",
        "treat heat exhaustion", "someone has heat exhaustion", "nakakaranas ng heat exhaustion",
        "signs of heat exhaustion", "senyales ng heat exhaustion", "what to do heat exhaustion"
    ]):
        return handle_heat_exhaustion_first_aid()

    # 3. DEHYDRATION (First Aid & Treatment)
    if any(k in clean_input for k in [
        "dehydration", "dehydrated", "treat dehydration", "how do i treat dehydration",
        "dehydration first aid", "first aid dehydration", "what to do for dehydration",
        "gamot sa dehydration", "paano gamutin ang dehydration", "am i dehydrated",
        "signs of dehydration", "senyales ng dehydration"
    ]):
        return handle_dehydration_first_aid()

    # # CL-NEW-11. What are the signs of dehydration?
    # if any(k in clean_input for k in [
    #     "signs of dehydration", "symptoms of dehydration", "dehydration signs",
    #     "dehydration symptoms", "how do i know if i am dehydrated",
    #     "how to know dehydration", "recognize dehydration",
    #     "senyales ng dehydration", "sintomas ng dehydration",
    #     "signs that you are dehydrated", "am i dehydrated", "paano malaman kung dehydrated",
    #     "dehydration warning signs", "what does dehydration feel like"
    # ]):
    #     conversation_state["last_topic"] = "dehydration_signs"
    #     return L(
    #         "💧 **What Are the Signs of Dehydration?**\n\n"
    #         "Dehydration occurs when your body loses more fluids than it takes in. During El Niño's hot, dry conditions, it's one of the most common health risks — and it can sneak up on you.\n\n"
    #         "🔴 **Early signs (mild dehydration):**\n"
    #         "• Feeling **thirsty** — note: thirst already means you are mildly dehydrated\n"
    #         "• **Dark yellow urine** — pale/clear urine is a good sign; dark means drink more water\n"
    #         "• Dry mouth and lips\n"
    #         "• Feeling tired or sluggish\n"
    #         "• Decreased urination (fewer trips to the bathroom)\n\n"
    #         "🟠 **Moderate signs:**\n"
    #         "• Headache\n"
    #         "• Dizziness or lightheadedness, especially when standing up\n"
    #         "• Muscle cramps\n"
    #         "• Dry skin that doesn't bounce back when pinched\n\n"
    #         "🔴 **Severe signs (seek help immediately):**\n"
    #         "• Rapid heartbeat\n"
    #         "• Rapid breathing\n"
    #         "• Sunken eyes\n"
    #         "• Confusion or irritability\n"
    #         "• Fainting\n"
    #         "• Little to no urination for 8+ hours\n\n"
    #         "👶 **Special groups at higher risk:**\n"
    #         "• **Children** — smaller body mass means faster dehydration\n"
    #         "• **Elderly** — reduced thirst sensation means they may not feel thirsty even when dehydrated\n"
    #         "• **Outdoor workers** — losing fluids faster through sweat\n\n"
    #         "💡 *The best cure is prevention — drink 8–10 glasses of water daily during El Niño months, and more if you are working outdoors or feeling hot.*",
    #         "💧 **Ano ang mga Senyales ng Dehydration?**\n\n"
    #         "Ang dehydration ay nangyayari kapag mas maraming fluid ang nawawala sa katawan mo kaysa sa natatanggap nito. Sa mainit at tuyong kondisyon ng El Niño, ito ay isa sa mga pinakakaraniwang health risks — at maaari kang masorpresa.\n\n"
    #         "🔴 **Maagang senyales (mild dehydration):**\n"
    #         "• Pakiramdam na **nauuhaw** — tandaan: ang pagkauhaw ay senyales na ng mild dehydration\n"
    #         "• **Maitim na dilaw na ihi** — maliwanag/malinaw na ihi ang magandang tanda; maitim ay nangangahulugang uminom pa ng tubig\n"
    #         "• Tuyo ang bibig at labi\n"
    #         "• Pakiramdam ng pagod o hindi aktibo\n"
    #         "• Mas kaunting pag-ihi (mas kaunting beses sa banyo)\n\n"
    #         "🟠 **Katamtamang senyales:**\n"
    #         "• Sakit ng ulo\n"
    #         "• Pagkahilo, lalo na kapag tumayo\n"
    #         "• Pamumulikat ng kalamnan\n"
    #         "• Tuyong balat na hindi bumabalik kapag kinurot\n\n"
    #         "🔴 **Malubhang senyales (humingi agad ng tulong):**\n"
    #         "• Mabilis na tibok ng puso\n"
    #         "• Mabilis na paghinga\n"
    #         "• Malalim na nakita ang mga mata\n"
    #         "• Pagkalito o pagkainis\n"
    #         "• Pag-himatay\n"
    #         "• Halos walang ihi sa loob ng 8+ oras\n\n"
    #         "👶 **Mga grupo na mas madaling ma-dehydrate:**\n"
    #         "• **Mga bata** — mas maliit na katawan kaya mas mabilis mag-dehydrate\n"
    #         "• **Mga matatanda** — nabawasang pakiramdam ng pagkauhaw kaya maaaring hindi sila mauhaw kahit dehydrated na\n"
    #         "• **Mga outdoor worker** — mas mabilis nawawala ang fluids sa pagpapawis\n\n"
    #         "💡 *Ang pinakamabuting lunas ay pag-iwas — uminom ng 8–10 baso ng tubig araw-araw sa panahon ng El Niño months, at mas marami kung nagtatrabaho ka sa labas o mainit ka.*"
    #     )

    # CL3-6. What should I do if someone experiences heat exhaustion?
    if any(k in clean_input for k in ["if someone experiences heat exhaustion", "someone has heat exhaustion", "nakakaranas ng heat exhaustion", "may heat exhaustion", "heat exhaustion first aid", "first aid heat exhaustion", "treat heat exhaustion", "help someone heat exhaustion", "what to do heat exhaustion", "kung may heat exhaustion", "someone experiences heat", "person has heat exhaustion"]):
        conversation_state["last_topic"] = "heat_exhaustion_action"
        return L(
            "🚑 **What Should I Do If Someone Experiences Heat Exhaustion?**\n\n"
            "Act quickly — heat exhaustion can escalate to a deadly heat stroke within minutes.\n\n"
            "✅ **Step-by-Step First Aid:**\n\n"
            "1. **Move them to a cool place** — indoors with a fan or aircon, or at least to the shade\n"
            "2. **Loosen tight clothing** — unbutton collars, remove jackets or excess layers\n"
            "3. **Cool the body down** — apply cool (not ice-cold) damp cloths to the neck, armpits, and groin\n"
            "4. **Give water or electrolyte drinks** — if they are conscious and able to swallow\n"
            "   *(Do NOT give water if the person is unconscious or vomiting)*\n"
            "5. **Fan the person** — air circulation helps reduce body temperature\n"
            "6. **Have them rest** — lying down with legs slightly elevated helps circulation\n\n"
            "🚨 **Call for Medical Help Immediately If:**\n"
            "• Symptoms do not improve within **30 minutes**\n"
            "• The person becomes confused, stops sweating, or loses consciousness\n"
            "• Body temperature exceeds **40°C** — this is now a heat stroke\n\n"
            "⛔ **Do NOT:**\n"
            "• Give alcohol or caffeinated drinks\n"
            "• Leave the person alone\n"
            "• Use ice-cold water — it can cause shock\n\n"
            "📞 *For emergencies in Padre Garcia, contact your local barangay health center or MDRRMO.*",
            "🚑 **Ano ang Dapat Gawin Kung May Nakakaranas ng Heat Exhaustion?**\n\n"
            "Kumilos agad — ang heat exhaustion ay maaaring lumala sa deadly na heat stroke sa loob ng ilang minuto.\n\n"
            "✅ **Step-by-Step First Aid:**\n\n"
            "1. **Ilipat sila sa malamig na lugar** — sa loob ng bahay na may fan o aircon, o hindi man ay sa lilim\n"
            "2. **Luwagan ang mahigpit na damit** — i-unbutton ang kwelyo, tanggalin ang jacket o sobrang damit\n"
            "3. **Palamigin ang katawan** — mag-apply ng malamig (hindi yelo) na basang tela sa leeg, kilikili, at singit\n"
            "4. **Bigyan ng tubig o electrolyte drinks** — kung conscious sila at kayang lumunok\n"
            "   *(Huwag magbigay ng tubig kung hindi conscious ang tao o nagsusuka)*\n"
            "5. **Pagpaypay-paypayan ang tao** — nakakatulong ang air circulation sa pagbaba ng body temperature\n"
            "6. **Pahinugahin sila** — ang paghiga na may bahagyang nakataas na paa ay nakakatulong sa circulation\n\n"
            "🚨 **Humingi Agad ng Medikal na Tulong Kung:**\n"
            "• Hindi gumagaling ang symptoms sa loob ng **30 minuto**\n"
            "• Nalilito ang tao, tumitigil sa pagpapawis, o nawalan ng malay\n"
            "• Ang body temperature ay higit sa **40°C** — heat stroke na ito\n\n"
            "⛔ **Huwag:**\n"
            "• Bigyan ng alak o caffeinated drinks\n"
            "• Iwanan ang tao nang mag-isa\n"
            "• Gumamit ng sobrang malamig na tubig — maaaring magdulot ng shock\n\n"
            "📞 *Para sa mga emergency sa Padre Garcia, makipag-ugnayan sa inyong lokal na barangay health center o MDRRMO.*"
        )

    # CL3-7. What should farmers do during drought?
    if any(k in clean_input for k in ["farmers during drought", "what should farmers do", "farmers drought", "ano ang gagawin ng magsasaka", "mga magsasaka sa drought", "farmers el nino tips", "farmer tips drought", "farming during drought", "agricultural tips drought", "farming tips el nino"]):
        conversation_state["last_topic"] = "farmers_drought"
        return L(
            "🌾 **What Should Farmers Do During Drought / El Niño?**\n\n"
            "Farmers in Padre Garcia are among the most vulnerable during El Niño. Here are practical steps to protect livelihoods:\n\n"
            "🌱 **Crop Management:**\n"
            "• Switch to **drought-tolerant crop varieties** — drought-resistant rice strains (e.g. NSIC Rc480), corn, camote, and kangkong handle dry conditions better\n"
            "• **Adjust planting schedules** — plant before the dry season peaks so crops mature before the worst months\n"
            "• **Apply mulching** — cover soil with dried leaves, straw, or plastic to slow moisture evaporation\n"
            "• Reduce plant spacing to shade the soil and slow water loss\n\n"
            "💧 **Water Management:**\n"
            "• Switch to **drip or furrow irrigation** to use less water more efficiently\n"
            "• Water crops **early morning or after 5pm** — less evaporation during cool hours\n"
            "• Build **small water catchment areas** or ponds to store rainwater for irrigation\n"
            "• Prioritize water for the most critical crop growth stages\n\n"
            "🐄 **Livestock:**\n"
            "• Ensure animals have access to clean, cool water at all times\n"
            "• Provide shade for animals to reduce heat stress\n"
            "• Stock up on animal feed before prices rise during a drought\n\n"
            "📢 **Seek Support:**\n"
            "• Contact the **local agricultural office (DA / MAO)** for drought assistance programs\n"
            "• Check if your barangay or municipality has seed subsidy or livelihood programs\n"
            "• Consider crop insurance if available through PhilCrop or DA programs",
            "🌾 **Ano ang Dapat Gawin ng mga Magsasaka sa Panahon ng Drought / El Niño?**\n\n"
            "Ang mga magsasaka sa Padre Garcia ay kabilang sa mga pinaka-vulnerable sa panahon ng El Niño. Narito ang mga praktikal na hakbang para protektahan ang kabuhayan:\n\n"
            "🌱 **Crop Management:**\n"
            "• Lumipat sa **drought-tolerant na varieties ng pananim** — drought-resistant na strain ng palay (hal. NSIC Rc480), mais, camote, at kangkong ay mas kaya ng tuyong kondisyon\n"
            "• **I-adjust ang planting schedules** — magtanim bago mag-peak ang dry season para lumago ang pananim bago dumating ang pinaka-masamang buwan\n"
            "• **Mag-apply ng mulching** — takpan ang lupa ng tuyong dahon, dayami, o plastik para pabagalin ang evaporation ng moisture\n"
            "• Bawasan ang plant spacing para ma-shade ang lupa at mapabagal ang pagkawala ng tubig\n\n"
            "💧 **Water Management:**\n"
            "• Lumipat sa **drip o furrow irrigation** para mas mahusay na magamit ang mas kaunting tubig\n"
            "• Magdilig ng pananim **nang umaga o pagkatapos ng 5pm** — mas kaunting evaporation sa malamig na oras\n"
            "• Gumawa ng **maliit na water catchment areas** o pond para mag-store ng rainwater para sa irrigation\n"
            "• Unahin ang tubig para sa pinaka-kritikal na growth stages ng pananim\n\n"
            "🐄 **Livestock:**\n"
            "• Tiyaking may access ang mga hayop sa malinis, malamig na tubig sa lahat ng oras\n"
            "• Magbigay ng lilim para sa mga hayop para mabawasan ang heat stress\n"
            "• Mag-stock ng animal feed bago tumaas ang presyo sa panahon ng drought\n\n"
            "📢 **Humingi ng Suporta:**\n"
            "• Makipag-ugnayan sa **lokal na agricultural office (DA / MAO)** para sa drought assistance programs\n"
            "• Suriin kung ang inyong barangay o munisipyo ay may seed subsidy o livelihood programs\n"
            "• Isaalang-alang ang crop insurance kung available sa pamamagitan ng PhilCrop o DA programs"
        )

    # CL3-8. How can I protect plants and crops during dry periods / El Niño?
    if any(k in clean_input for k in ["protect plants", "protect crops", "crops during dry", "plants during el nino", "protect crops during", "panatilihing buhay ang halaman", "protektahan ang pananim", "pananim sa tagtuyot", "plants dry season", "crops dry season", "halaman sa el nino"]):
        conversation_state["last_topic"] = "protect_crops"
        return L(
            "🌱 **How Can I Protect Plants and Crops During Dry Periods / El Niño?**\n\n"
            "Whether you're a backyard gardener or a farmer, these steps help your plants survive the heat:\n\n"
            "🪣 **Water Wisely:**\n"
            "• Water **deeply but less frequently** — shallow daily watering encourages shallow roots; deep watering once every 2–3 days builds drought resilience\n"
            "• Water at the **base of plants**, not the leaves — reduces evaporation and disease\n"
            "• Water in the **early morning or evening** when evaporation is lowest\n\n"
            "🍂 **Mulching:**\n"
            "• Apply a 5–10cm layer of mulch (dried leaves, straw, rice hull) around plants\n"
            "• Mulch slows evaporation, moderates soil temperature, and reduces the need for watering\n\n"
            "🌿 **Plant Selection and Spacing:**\n"
            "• Choose **drought-tolerant varieties**: camote, cassava, kangkong, drought-resistant rice\n"
            "• Group plants closer together to shade the soil between them\n"
            "• Remove weeds — they compete with crops for limited water\n\n"
            "☀️ **Protect from Direct Sun:**\n"
            "• Use shade netting (50–70% shade) over seedlings and young plants\n"
            "• For backyard gardens, temporary bamboo or palm leaf covers work well\n\n"
            "🧪 **Soil Health:**\n"
            "• Add organic matter (compost) to improve soil's water-holding capacity\n"
            "• Avoid tilling too frequently — it speeds up moisture loss from soil",
            "🌱 **Paano Protektahan ang mga Halaman at Pananim sa Panahon ng Tagtuyot / El Niño?**\n\n"
            "Kahit backyard gardener ka man o magsasaka, nakakatulong ang mga hakbang na ito para mabuhay ang iyong mga halaman sa init:\n\n"
            "🪣 **Maingat na Pagdidilig:**\n"
            "• Magdilig nang **malalim ngunit mas madalang** — ang mababaw na pang-araw-araw na pagdidilig ay nagdudulot ng mababaw na ugat; ang malalim na pagdidilig bawat 2–3 araw ay nagpapalakas ng drought resilience\n"
            "• Magdilig sa **base ng halaman**, hindi sa dahon — nababawasan ang evaporation at sakit\n"
            "• Magdilig nang **umaga o gabi** kapag pinakamababa ang evaporation\n\n"
            "🍂 **Mulching:**\n"
            "• Mag-apply ng 5–10cm na layer ng mulch (tuyong dahon, dayami, ipa ng palay) sa paligid ng halaman\n"
            "• Nagpapabagal ng evaporation ang mulch, nagpapanatili ng soil temperature, at nagpapababa ng pangangailangan sa tubig\n\n"
            "🌿 **Pagpili at Paglalagay ng Halaman:**\n"
            "• Pumili ng **drought-tolerant varieties**: camote, cassava, kangkong, drought-resistant na palay\n"
            "• Maggrupo ng mga halaman nang mas malapit para ma-shade ang lupa sa pagitan ng mga ito\n"
            "• Tanggalin ang mga damo — nakikipagkumpetensya sila sa pananim para sa limitadong tubig\n\n"
            "☀️ **Protektahan mula sa Direktang Araw:**\n"
            "• Gumamit ng shade netting (50–70% shade) sa mga seedling at batang halaman\n"
            "• Para sa mga backyard garden, ang pansamantalang takip na gawa sa kawayan o dahon ng puno ay epektibo rin\n\n"
            "🧪 **Kalusugan ng Lupa:**\n"
            "• Magdagdag ng organic matter (compost) para mapabuti ang kakayahan ng lupa na mag-hold ng tubig\n"
            "• Iwasan ang madalas na paghuhukay — nagpapabilis ito ng pagkawala ng moisture mula sa lupa"
        )

    # CL3-9. How should I store water safely?
    if any(k in clean_input for k in ["store water safely", "store water", "water storage", "safe water storage", "paano mag-imbak ng tubig", "water properly", "how to store water", "safe drinking water", "pag-iimbak ng tubig", "mag-store ng tubig"]):
        conversation_state["last_topic"] = "water_storage"
        return L(
            "🪣 **How Should I Store Water Safely?**\n\n"
            "Storing water properly is essential during El Niño when supply can become irregular or cut off entirely.\n\n"
            "✅ **Safe Storage Steps:**\n\n"
            "1. **Use clean, food-grade containers** — sealed water drums, food-grade plastic containers, or water tanks\n"
            "   *(Avoid containers that previously held chemicals or non-food products)*\n"
            "2. **Wash containers thoroughly** before filling — use soap, rinse well, then rinse with a diluted bleach solution (1 tsp bleach per 1 liter water), let air dry\n"
            "3. **Fill from a safe water source** — tap water (when available), filtered water, or purchased drinking water\n"
            "4. **Keep containers tightly covered** — prevents contamination, mosquito breeding, and evaporation\n"
            "5. **Store in a cool, dark place** — away from direct sunlight which promotes bacterial growth\n"
            "6. **Label with the date filled** — stored water should be used or refreshed within **1–3 days** for drinking (longer for non-drinking use)\n\n"
            "💧 **For Drinking Water Specifically:**\n"
            "• If unsure of quality, **boil water for at least 1 minute** before drinking\n"
            "• You can also use **water purification tablets** (available at pharmacies)\n"
            "• Filter through clean cloth or a commercial filter before boiling if water looks cloudy\n\n"
            "⚠️ **Signs Water Has Gone Bad:**\n"
            "• Unusual smell or color\n"
            "• Cloudiness that doesn't settle\n"
            "• Presence of floating particles\n\n"
            "🏠 *Recommended minimum: store at least **3 days' worth of water** per household (roughly 4 liters per person per day).*",
            "🪣 **Paano Mag-imbak ng Tubig nang Ligtas?**\n\n"
            "Mahalaga ang tamang pag-iimbak ng tubig sa panahon ng El Niño kapag ang supply ay maaaring maging irregular o tuluyang mawala.\n\n"
            "✅ **Mga Hakbang sa Ligtas na Pag-iimbak:**\n\n"
            "1. **Gumamit ng malinis, food-grade na containers** — sealed water drums, food-grade plastic containers, o water tanks\n"
            "   *(Iwasan ang mga containers na ginamit dati sa chemicals o non-food products)*\n"
            "2. **Hugasan nang mabuti ang mga containers** bago punuin — gumamit ng sabon, banlawan nang mabuti, pagkatapos ay banlawan ng diluted bleach solution (1 kutsarita ng bleach sa 1 litro ng tubig), at hayaang matuyo sa hangin\n"
            "3. **Punan mula sa ligtas na water source** — tap water (kung available), filtered water, o biniling drinking water\n"
            "4. **Palaging takpan nang mahigpit ang mga containers** — pinipigilan ang kontaminasyon, pagpapalaki ng lamok, at evaporation\n"
            "5. **I-store sa malamig, madilim na lugar** — malayo sa direktang sikat ng araw na nagpo-promote ng bacterial growth\n"
            "6. **Lagyan ng label ng petsa ng pagpuno** — ang nakaimbak na tubig ay dapat gamitin o palitan sa loob ng **1–3 araw** para sa pag-inom (mas matagal para sa non-drinking use)\n\n"
            "💧 **Para sa Drinking Water Specifically:**\n"
            "• Kung hindi sigurado sa kalidad, **pakuluin ang tubig ng hindi bababa sa 1 minuto** bago inumin\n"
            "• Maaari ding gumamit ng **water purification tablets** (available sa mga parmasya)\n"
            "• Salain sa malinis na tela o commercial filter bago pakuluin kung mukhang malabo ang tubig\n\n"
            "⚠️ **Mga Senyales na Masama na ang Tubig:**\n"
            "• Di-karaniwang amoy o kulay\n"
            "• Kalabuan na hindi lumalipas\n"
            "• Mga floating particles\n\n"
            "🏠 *Inirerekomenda: mag-ipon ng hindi bababa sa **3 araw na suplay ng tubig** bawat sambahayan (mga 4 litro bawat tao bawat araw).*"
        )

    # CL3-10. What should I include in an emergency kit?
    if any(k in clean_input for k in ["emergency kit", "emergency bag", "go bag", "emergency supplies", "disaster kit", "emergency pack", "ano ang dapat nasa emergency kit", "emergency preparedness kit", "emergency checklist", "survival kit", "ano ang isasama sa emergency"]):
        conversation_state["last_topic"] = "emergency_kit"
        return L(
            "🎒 **What Should I Include in an Emergency Kit for El Niño / Extreme Heat?**\n\n"
            "Having a ready emergency kit ensures your family can cope during prolonged dry spells, water shortages, or heat emergencies.\n\n"
            "💧 **Water and Food:**\n"
            "• At least **3 days of drinking water** (4 liters per person per day)\n"
            "• Non-perishable food (canned goods, biscuits, dried food)\n"
            "• Manual can opener\n"
            "• Water purification tablets or small water filter\n\n"
            "🏥 **Health and First Aid:**\n"
            "• Basic first aid kit (bandages, antiseptic, gauze)\n"
            "• Oral rehydration salts (ORS) — for dehydration\n"
            "• Thermometer\n"
            "• Paracetamol / fever medicine\n"
            "• Any personal or prescription medications (1–2 week supply)\n"
            "• Sunscreen and lip balm\n\n"
            "🔦 **Tools and Essentials:**\n"
            "• Flashlight and extra batteries\n"
            "• Battery-powered or hand-crank radio (for PAGASA updates)\n"
            "• Whistle (to signal for help)\n"
            "• Multi-purpose tool or knife\n"
            "• Dust masks\n\n"
            "📄 **Important Documents (in a waterproof bag):**\n"
            "• ID cards, birth certificates\n"
            "• Insurance documents\n"
            "• Emergency contact numbers\n\n"
            "👕 **Clothing and Personal Items:**\n"
            "• Light, breathable clothing\n"
            "• Hat, sunglasses\n"
            "• Extra underwear and socks\n"
            "• Small towel\n\n"
            "📞 **Emergency Contacts to Include:**\n"
            "• Local barangay hall\n"
            "• MDRRMO (Municipal Disaster Risk Reduction and Management Office)\n"
            "• Nearest health center or hospital\n"
            "• PAGASA hotline: 1553",
            "🎒 **Ano ang Dapat Isama sa Emergency Kit para sa El Niño / Matinding Init?**\n\n"
            "Ang pagkakaroon ng handang emergency kit ay nagtitiyak na makakayanan ng inyong pamilya ang matagal na tagtuyot, water shortage, o heat emergency.\n\n"
            "💧 **Tubig at Pagkain:**\n"
            "• Hindi bababa sa **3 araw na drinking water** (4 litro bawat tao bawat araw)\n"
            "• Non-perishable na pagkain (canned goods, biskwit, dried food)\n"
            "• Manual na can opener\n"
            "• Water purification tablets o maliit na water filter\n\n"
            "🏥 **Kalusugan at First Aid:**\n"
            "• Basic first aid kit (bandage, antiseptic, gauze)\n"
            "• Oral rehydration salts (ORS) — para sa dehydration\n"
            "• Thermometer\n"
            "• Paracetamol / gamot sa lagnat\n"
            "• Anumang personal o prescription na gamot (1–2 linggong supply)\n"
            "• Sunscreen at lip balm\n\n"
            "🔦 **Mga Kagamitan at Essentials:**\n"
            "• Flashlight at extra batteries\n"
            "• Battery-powered o hand-crank na radyo (para sa PAGASA updates)\n"
            "• Pito (para mag-signal ng tulong)\n"
            "• Multi-purpose tool o kutsilyo\n"
            "• Dust masks\n\n"
            "📄 **Mahahalagang Dokumento (sa waterproof bag):**\n"
            "• ID cards, birth certificates\n"
            "• Insurance documents\n"
            "• Emergency contact numbers\n\n"
            "👕 **Damit at Personal na Gamit:**\n"
            "• Magaan, breathable na damit\n"
            "• Sombrero, salamin sa araw\n"
            "• Extra pang-loob at medyas\n"
            "• Maliit na tuwalya\n\n"
            "📞 **Mga Emergency Contact na Isasama:**\n"
            "• Lokal na barangay hall\n"
            "• MDRRMO (Municipal Disaster Risk Reduction and Management Office)\n"
            "• Pinakamalapit na health center o ospital\n"
            "• PAGASA hotline: 1553"
        )

    # CL3-11. When should I seek medical assistance during extreme heat?
    if any(k in clean_input for k in ["seek medical", "see a doctor", "go to hospital", "when to seek", "medical help heat", "medical assistance heat", "kumuha ng medikal", "pumunta sa doktor", "ospital heat", "when doctor heat", "medical attention heat", "when emergency heat"]):
        conversation_state["last_topic"] = "medical_seek_heat"
        return L(
            "🏥 **When Should I Seek Medical Assistance During Extreme Heat?**\n\n"
            "Most heat-related discomfort can be managed at home, but certain signs require **immediate professional medical attention**.\n\n"
            "🚨 **Go to the hospital or call for help IMMEDIATELY if:**\n"
            "• Body temperature is **above 40°C (104°F)**\n"
            "• The person **stops sweating** despite the heat (dangerous sign — the body's cooling system has failed)\n"
            "• **Loss of consciousness** or fainting that doesn't recover quickly\n"
            "• **Confusion, disorientation, or slurred speech**\n"
            "• **Seizures**\n"
            "• Skin that is **hot, red, and dry** (not sweating)\n"
            "• **Vomiting that won't stop**\n"
            "• Rapid, strong pulse with no improvement after cooling measures\n\n"
            "⚠️ **See a doctor within the day if:**\n"
            "• Heat exhaustion symptoms (heavy sweating, weakness, dizziness, nausea) **do not improve within 30 minutes** of rest and cooling\n"
            "• A child under 5 or elderly person shows any signs of heat illness\n"
            "• You have a chronic illness (heart disease, diabetes, hypertension) and are experiencing heat-related symptoms\n\n"
            "✅ **You can manage at home if:**\n"
            "• Mild dizziness or fatigue that improves quickly after moving to a cool area and drinking water\n"
            "• Light sweating and warmth that subsides with rest and cooling\n\n"
            "📞 **Emergency Contacts:**\n"
            "• Barangay health center / community health worker\n"
            "• MDRRMO (Municipal Disaster Risk Reduction and Management Office)\n"
            "• National Emergency Hotline: **911**\n"
            "• Red Cross: **143**",
            "🏥 **Kailan Dapat Humingi ng Medikal na Tulong sa Panahon ng Matinding Init?**\n\n"
            "Karamihan sa heat-related na discomfort ay maaaring hawakan sa bahay, ngunit ang ilang senyales ay nangangailangan ng **agarang propesyonal na medikal na atensyon**.\n\n"
            "🚨 **Pumunta sa ospital o tumawag ng tulong AGAD kung:**\n"
            "• Ang body temperature ay **higit sa 40°C (104°F)**\n"
            "• **Tumitigil ang pagpapawis** ng tao kahit mainit pa (mapanganib na senyales — nabigo na ang cooling system ng katawan)\n"
            "• **Pagkawala ng malay** o pagkahimatay na hindi agad bumabawi\n"
            "• **Pagkalito, disorientation, o hindi maliwanag na pagsasalita**\n"
            "• **Seizures (kombulsyon)**\n"
            "• Balat na **mainit, pula, at tuyo** (hindi nagpapawis)\n"
            "• **Pagsusuka na hindi tumitigil**\n"
            "• Mabilis, malakas na pulso na walang improvement pagkatapos ng cooling measures\n\n"
            "⚠️ **Pumunta sa doktor sa araw na iyon kung:**\n"
            "• Ang mga heat exhaustion symptoms (sobrang pawis, panghihina, pagkahilo, pagduduwal) ay **hindi gumagaling sa loob ng 30 minuto** ng pahinga at cooling\n"
            "• Ang isang batang wala pang 5 taon o matandang tao ay nagpapakita ng anumang senyales ng heat illness\n"
            "• Mayroon kang chronic illness (heart disease, diabetes, hypertension) at nakakaranas ng heat-related symptoms\n\n"
            "✅ **Maaaring hawakan sa bahay kung:**\n"
            "• Banayad na pagkahilo o pagod na mabilis na gumagaling pagkatapos lumipat sa malamig na lugar at uminom ng tubig\n"
            "• Magaang na pagpapawis at init na humihinto sa pahinga at cooling\n\n"
            "📞 **Mga Emergency Contact:**\n"
            "• Barangay health center / community health worker\n"
            "• MDRRMO (Municipal Disaster Risk Reduction and Management Office)\n"
            "• National Emergency Hotline: **911**\n"
            "• Red Cross: **143**"
        )
    

    # 7. Year/Month queries

    # Handle "last year" / "previous year" — resolve to 2025 (current year is 2026)
    if any(k in clean_input for k in [
        "last year", "previous year", "noong nakaraang taon", "nakaraang taon",
        "last yr", "yung nakaraang taon", "noong isang taon"
    ]):
        return handle_year_query("2025", padre_garcia_only, is_rain_specific=True)

    years_found = re.findall(r'\b(20[0-3]\d)\b', clean_input)

    # Detect all month names present in input
    months_found = []
    for month_str, month_num in MONTH_NAME_TO_INT.items():
        if re.search(r'\b' + month_str + r'\b', clean_input):
            if month_num not in months_found:
                months_found.append(month_num)

    # Two distinct months with no year → month comparison
    # If user asks about comparing months but names no specific months → explain capability
    if len(months_found) < 2 and any(k in clean_input for k in [
        "compare", "ihambing", "vs", "versus"
    ]) and any(k in clean_input for k in ["month", "months", "buwan", "monthly"]):
        return L(
            "📅 **Month Comparison — How to Ask**\n\n"
            "Yes! I can compare the rainfall of any two months using historical data (2010–2026).\n\n"
            "Just name the two months in your question. For example:\n"
            "• *'Compare June and December'*\n"
            "• *'June vs September rainfall'*\n"
            "• *'What is the difference between May and August rainfall?'*\n"
            "• *'Ihambing ang Marso at Hunyo'*\n\n"
            "I'll show the average rainfall, average temperature, and all-time records for each month at Padre Garcia and the other monitored stations.",
            "📅 **Pagkukumpara ng Buwan — Paano Magtanong**\n\n"
            "Oo! Maaari kong ihambing ang ulan ng dalawang buwan gamit ang historical data (2010–2026).\n\n"
            "Isama lang ang dalawang buwan sa iyong tanong. Halimbawa:\n"
            "• *'Ihambing ang Hunyo at Disyembre'*\n"
            "• *'Hunyo vs Setyembre ulan'*\n"
            "• *'Ano ang pagkakaiba ng ulan sa Mayo at Agosto?'*\n"
            "• *'Compare June and December'*\n\n"
            "Ipapakita ko ang average rainfall, average temperature, at all-time records para sa bawat buwan sa Padre Garcia at iba pang monitored na stations."
        )

    if len(months_found) >= 2 and not years_found:
        if any(k in clean_input for k in [
            "compare", "vs", "versus", "difference", "ihambing", "pagkakaiba",
            "contrast", "between", "pagkukumpara", "and", "at"
        ]):
            return handle_month_comparison(months_found[0], months_found[1], padre_garcia_only)

    for month_str, month_num in MONTH_NAME_TO_INT.items():
        if month_str in clean_input:
            if any(k in clean_input for k in ["predict", "forecast", "prediction", "hulaan", "prediksyon"]):
                target_yr = int(years_found[0]) if years_found else None
                return handle_ml_forecast_query(month_num, target_yr)
            elif len(years_found) == 1:
                return handle_month_year_query(month_num, int(years_found[0]), padre_garcia_only)

    if len(years_found) >= 2:
        return handle_year_comparison(years_found[0], years_found[1], padre_garcia_only)

    # Detect if user specifically asked about rain/weather
    is_rain_inquiry = any(k in clean_input for k in [
        "rain", "rainfall", "ulan", "pag-ulan", "weather", "panahon", 
        "what happened", "nangyari", "show me", "record", "summary"
    ])

    if len(years_found) == 1:
        return handle_year_query(years_found[0], padre_garcia_only, is_rain_specific=is_rain_inquiry)
    # 8. ML prediction — broad catch-all (skip if a specific prediction-topic intent matches)
    if any(k in clean_input for k in ["predict", "forecast", "machine learning", "ml model", "hulaan", "prediksyon", "prediction"]):
        if not any(k in clean_input for k in [
            "what is rainfall prediction", "what is rain prediction", "what is a rainfall prediction",
            "ano ang rainfall prediction", "explain rainfall prediction", "ibig sabihin ng rainfall prediction",
            "can you predict rainfall", "can you predict rain", "do you predict rainfall",
            "kaya mo bang hulaan ang ulan", "maari mo bang hulaan",
            "how accurate is", "accuracy of prediction", "how reliable is the prediction",
            "prediction accuracy", "how good is the prediction", "gaano katumpak",
            "tumpak ba ang prediction", "reliable ba ang prediction",
            "what data is used for prediction", "what data is used", "data used for prediction",
            "what data does the model use", "what inputs does the model",
            "anong data ang ginagamit", "saan galing ang data",
            "difference between historical and predicted", "historical vs predicted",
            "predicted vs historical", "pagkakaiba ng historical at predicted",
            "what ml model", "what machine learning model", "which ml model", "which model is used",
            "what model is used", "what model does", "what algorithm", "what ai model",
            "random forest", "ano ang ml model", "anong modelo", "anong ml model",
            "accurate is the production", "rainfall production",
        ]):
            return L(
                "🤖 **ML Model Ready!**\n"
                "YES! You can ask for a monthly rainfall forecast, for example:\n"
                "• *'Predict rain for May'*\n"
                "• *'Forecast rainfall for December'*",
                "🤖 **Handa na ang ML Model!**\n"
                "OO! Maaari kang mag-ask about monthly rainfall forecast, halimbawa:\n"
                "• *'Hulaan ang ulan para sa May'*\n"
                "• *'Forecast ng rainfall para sa December'*"
            )

    # CL-11. How does El Niño affect the Philippines?
    if any(k in clean_input for k in ["affect the philippines", "el nino philippines", "philippines el nino", "epekto sa pilipinas", "pilipinas el nino", "how el nino affects", "nakakaapekto sa pilipinas", "effect in the philippines", "impact in the philippines", "nakakaapekto ang el nino sa pilipinas"]):
        conversation_state["last_topic"] = "el_nino_philippines"
        return L(
            "🇵🇭 **How Does El Niño Affect the Philippines?**\n\n"
            "The Philippines is one of the most vulnerable countries to El Niño because it sits in a region directly affected by Pacific Ocean temperature shifts.\n\n"
            "🌧️ **Rainfall:**\n"
            "• Below-normal rainfall, especially from **December to May**\n"
            "• Prolonged dry spells and droughts, particularly in Luzon and Visayas\n"
            "• Reduced river flow and reservoir levels\n\n"
            "🌡️ **Temperature:**\n"
            "• Above-normal temperatures, increasing heat stress on people and crops\n"
            "• Higher risk of heat-related illnesses\n\n"
            "🌾 **Agriculture:**\n"
            "• Crop failures — especially rice and corn — due to water shortages\n"
            "• Reduced irrigation supply forces farmers to delay or skip planting\n"
            "• Livestock suffer from heat and lack of water\n\n"
            "💧 **Water Supply:**\n"
            "• Domestic water shortages in communities and cities\n"
            "• Hydroelectric power output drops as water levels fall\n\n"
            "📍 **Locally in Padre Garcia, Batangas:**\n"
            "Our data (2010–2026) shows that during El Niño months, rainfall drops significantly and temperatures rise above the monthly average — directly impacting local farmers and households.",
            "🇵🇭 **Paano Nakakaapekto ang El Niño sa Pilipinas?**\n\n"
            "Ang Pilipinas ay isa sa mga pinakamahinang bansa sa El Niño dahil nasa region ito na direktang apektado ng mga changes sa Pacific Ocean temperature.\n\n"
            "🌧️ **Rainfall:**\n"
            "• Below-normal rainfall, lalo na mula **Disyembre hanggang Mayo**\n"
            "• Mahahabang dry spells at drought, partikular sa Luzon at Visayas\n"
            "• Nababawasang river flow at reservoir levels\n\n"
            "🌡️ **Temperature:**\n"
            "• Above-normal temperatures, nagdudulot ng heat stress sa mga tao at pananim\n"
            "• Mas mataas na risk ng heat-related illnesses\n\n"
            "🌾 **Agriculture:**\n"
            "• Mga crop failures — lalo na ang palay at mais — dahil sa water shortages\n"
            "• Ang reduced irrigation supply ay nagpipilit sa mga farmers na mag-delay o lumaktaw ng planting\n"
            "• Naghihirap ang mga hayop dahil sa init at kakulangan ng tubig\n\n"
            "💧 **Water Supply:**\n"
            "• Domestic water shortages sa mga komunidad at lungsod\n"
            "• Bumababa ang hydroelectric power output habang bumababa ang water levels\n\n"
            "📍 **Locally sa Padre Garcia, Batangas:**\n"
            "Ang aming data (2010–2026) ay nagpapakita na sa panahon ng El Niño months, malaki ang pagbaba ng rainfall at tumataas ang temperature nang higit sa monthly average — direktang nakakaapekto sa mga local farmers at households."
        )

    # CL-12. Does El Niño happen every year?
    if any(k in clean_input for k in ["every year", "bawat taon", "does el nino happen every year", "how often el nino", "how frequent", "gaano kadalas", "frequency of el nino", "regular ba ang el nino"]):
        conversation_state["last_topic"] = "el_nino_frequency"
        return L(
            "📅 **Does El Niño Happen Every Year?**\n\n"
            "No — El Niño does **not** happen every year. It is an irregular cycle that occurs roughly every **2 to 7 years**, though the timing is not perfectly predictable.\n\n"
            "🔁 **Historical frequency:**\n"
            "• In some decades, El Niño events occur closer together (every 2–3 years)\n"
            "• In others, they are spaced further apart (5–7 years)\n"
            "• A single event typically lasts 9–18 months\n\n"
            "📊 **From our dataset (2010–2026), notable El Niño periods included:**\n"
            "• 2015–2016 — Very Strong El Niño (one of the strongest on record globally)\n"
            "• 2018–2019 — Weak/Moderate El Niño\n"
            "• 2023–2024 — Strong El Niño\n\n"
            "🔄 Between events, the ENSO cycle is either in a **Neutral** phase or a **La Niña** phase — both of which bring different conditions to the Philippines.\n\n"
            "💡 *Even in non-El Niño years, it is worth monitoring ONI values and PAGASA seasonal forecasts to stay prepared.*",
            "📅 **Nagkakaroon Ba ng El Niño Tuwing Taon?**\n\n"
            "Hindi — ang El Niño ay **hindi** nangyayari tuwing taon. Ito ay isang irregular cycle na nangyayari nang humigit-kumulang bawat **2 hanggang 7 taon**, kahit hindi perfectly predictable ang timing.\n\n"
            "🔁 **Historical frequency:**\n"
            "• Sa ilang dekada, ang mga El Niño events ay mas malapit sa isa't isa (bawat 2–3 taon)\n"
            "• Sa iba, mas malayo ang agwat (5–7 taon)\n"
            "• Ang isang event ay karaniwang tumatagal ng 9–18 buwan\n\n"
            "📊 **Mula sa aming dataset (2010–2026), mga notable na El Niño periods:**\n"
            "• 2015–2016 — Very Strong El Niño (isa sa pinaka-malakas na naitala globally)\n"
            "• 2018–2019 — Weak/Moderate El Niño\n"
            "• 2023–2024 — Strong El Niño\n\n"
            "🔄 Sa pagitan ng mga events, ang ENSO cycle ay nasa **Neutral** o **La Niña** phase — parehong nagdudulot ng iba't ibang kondisyon sa Pilipinas.\n\n"
            "💡 *Kahit sa mga non-El Niño years, sulit pa rin ang pag-monitor ng ONI values at PAGASA seasonal forecasts para laging handa.*"
        )
         # CL-13. Signs of El Nino?
    if any(k in clean_input for k in ["every year", "bawat taon", "does el nino happen every year", "how often el nino", "how frequent", "gaano kadalas", "frequency of el nino", "regular ba ang el nino"]):
        conversation_state["last_topic"] = "el_nino_frequency"
        return L(
            "📅 **Does El Niño Happen Every Year?**\n\n"
            "No — El Niño does **not** happen every year. It is an irregular cycle that occurs roughly every **2 to 7 years**, though the timing is not perfectly predictable.\n\n"
            "🔁 **Historical frequency:**\n"
            "• In some decades, El Niño events occur closer together (every 2–3 years)\n"
            "• In others, they are spaced further apart (5–7 years)\n"
            "• A single event typically lasts 9–18 months\n\n"
            "📊 **From our dataset (2010–2026), notable El Niño periods included:**\n"
            "• 2015–2016 — Very Strong El Niño (one of the strongest on record globally)\n"
            "• 2018–2019 — Weak/Moderate El Niño\n"
            "• 2023–2024 — Strong El Niño\n\n"
            "🔄 Between events, the ENSO cycle is either in a **Neutral** phase or a **La Niña** phase — both of which bring different conditions to the Philippines.\n\n"
            "💡 *Even in non-El Niño years, it is worth monitoring ONI values and PAGASA seasonal forecasts to stay prepared.*",
            "📅 **Nagkakaroon Ba ng El Niño Tuwing Taon?**\n\n"
            "Hindi — ang El Niño ay **hindi** nangyayari tuwing taon. Ito ay isang irregular cycle na nangyayari nang humigit-kumulang bawat **2 hanggang 7 taon**, kahit hindi perfectly predictable ang timing.\n\n"
            "🔁 **Historical frequency:**\n"
            "• Sa ilang dekada, ang mga El Niño events ay mas malapit sa isa't isa (bawat 2–3 taon)\n"
            "• Sa iba, mas malayo ang agwat (5–7 taon)\n"
            "• Ang isang event ay karaniwang tumatagal ng 9–18 buwan\n\n"
            "📊 **Mula sa aming dataset (2010–2026), mga notable na El Niño periods:**\n"
            "• 2015–2016 — Very Strong El Niño (isa sa pinaka-malakas na naitala globally)\n"
            "• 2018–2019 — Weak/Moderate El Niño\n"
            "• 2023–2024 — Strong El Niño\n\n"
            "🔄 Sa pagitan ng mga events, ang ENSO cycle ay nasa **Neutral** o **La Niña** phase — parehong nagdudulot ng iba't ibang kondisyon sa Pilipinas.\n\n"
            "💡 *Kahit sa mga non-El Niño years, sulit pa rin ang pag-monitor ng ONI values at PAGASA seasonal forecasts para laging handa.*"
        )
    # ==========================================
    # CHECKLIST 4 — Community Contacts & LGU Services
    # ==========================================

    # CL-NEW-12. Rainfall prediction explainer questions
    if any(k in clean_input for k in [
        "what is rainfall prediction", "what is rain prediction", "what is a rainfall prediction",
        "ano ang rainfall prediction", "ano ang rain prediction", "explain rainfall prediction",
        "ipaliwanag ang rainfall prediction", "what does rainfall prediction mean",
        "ibig sabihin ng rainfall prediction", "what is prediction", "what is climate prediction"
    ]):
        conversation_state["last_topic"] = "prediction_explainer"
        return L(
            "🤖 **What Is Rainfall Prediction?**\n\n"
            "**Rainfall prediction** (or rainfall forecasting) is the use of data and mathematical models to estimate how much rain is likely to fall in a specific place and time period — before it actually happens.\n\n"
            "In this chatbot, rainfall prediction is done using a **Machine Learning (ML) model** trained on historical weather and climate data from Padre Garcia, Batangas (2010–2026).\n\n"
            "📊 **How it works here:**\n"
            "1. Historical climate data is collected — monthly temperature, regional rainfall, and ONI (El Niño/La Niña indicator) values\n"
            "2. A **Random Forest Regressor** model is trained to find patterns between those inputs and past rainfall\n"
            "3. When you ask for a prediction, the model uses those learned patterns to estimate rainfall for the requested month\n\n"
            "💡 *You can ask for a prediction like: 'Predict rainfall for May' or 'What is the predicted rainfall for January 2027?'*\n\n"
            "⚠️ *Predictions are estimates — not guarantees. Always check PAGASA for official forecasts.*",
            "🤖 **Ano ang Rainfall Prediction?**\n\n"
            "Ang **rainfall prediction** (o rainfall forecasting) ay ang paggamit ng data at mathematical models para tantiyahin kung gaano karaming ulan ang malamang na mahuhulog sa isang partikular na lugar at panahon — bago pa ito mangyari.\n\n"
            "Sa chatbot na ito, ginagawa ang rainfall prediction gamit ang isang **Machine Learning (ML) model** na sinanay sa historical weather at climate data mula Padre Garcia, Batangas (2010–2026).\n\n"
            "📊 **Paano ito gumagana dito:**\n"
            "1. Kinokolekta ang historical climate data — monthly temperature, regional rainfall, at ONI (El Niño/La Niña indicator) values\n"
            "2. Ang isang **Random Forest Regressor** model ay sinanay para mahanap ang mga pattern sa pagitan ng mga input na iyon at nakaraan na rainfall\n"
            "3. Kapag humingi ka ng prediction, ginagamit ng model ang mga natutunan na pattern para i-estimate ang rainfall para sa hiniling na buwan\n\n"
            "💡 *Maaari kang humingi ng prediction tulad ng: 'Hulaan ang ulan para sa Mayo' o 'Ano ang predicted na ulan para Enero 2027?'*\n\n"
            "⚠️ *Ang mga prediction ay mga tantiya — hindi garantiya. Laging suriin ang PAGASA para sa opisyal na forecast.*"
        )

    if any(k in clean_input for k in [
        "can you predict rainfall", "can you predict rain", "do you predict rainfall",
        "predict rainfall", "can this chatbot predict", "can readykomunidad predict",
        "kaya mo bang hulaan ang ulan", "maari mo bang hulaan ang ulan",
        "pwede bang hulaan ang ulan", "prediction ng ulan", "mayroon prediction"
    ]):
        # Only intercept if NOT already a specific month/year prediction query
        if not any(k in clean_input for k in ["january", "february", "march", "april", "may", "june",
                                               "july", "august", "september", "october", "november", "december",
                                               "enero", "pebrero", "marso", "abril", "hunyo", "hulyo",
                                               "agosto", "setyembre", "oktubre", "nobyembre", "disyembre"]):
            conversation_state["last_topic"] = "prediction_capability"
            return L(
                "🤖 **Can I Predict Rainfall?**\n\n"
                "Yes! This chatbot uses a **Machine Learning model** to predict monthly rainfall for **Padre Garcia, Batangas**.\n\n"
                "📅 **How to ask for a prediction:**\n"
                "• *'Predict rainfall for May'*\n"
                "• *'What is the predicted rainfall for January 2027?'*\n"
                "• *'How much rain is expected in August?'*\n\n"
                "📊 **What the model uses:**\n"
                "• Month of the year\n"
                "• Historical temperature averages\n"
                "• Regional rainfall data from nearby stations\n"
                "• ONI (El Niño/La Niña indicator) values\n\n"
                "⚠️ *Predictions are ML-based estimates using historical patterns — they are not official weather forecasts. For real-time forecasts, check **PAGASA** (www.pagasa.dost.gov.ph).*",
                "🤖 **Maari Ba Akong Mag-Predict ng Ulan?**\n\n"
                "Oo! Gumagamit ang chatbot na ito ng **Machine Learning model** para mahulaan ang monthly rainfall para sa **Padre Garcia, Batangas**.\n\n"
                "📅 **Paano humingi ng prediction:**\n"
                "• *'Hulaan ang ulan para sa Mayo'*\n"
                "• *'Ano ang predicted na ulan para sa Enero 2027?'*\n"
                "• *'Gaano karaming ulan ang inaasahan sa Agosto?'*\n\n"
                "📊 **Ginagamit ng model:**\n"
                "• Buwan ng taon\n"
                "• Historical temperature averages\n"
                "• Regional rainfall data mula sa mga kalapit na estasyon\n"
                "• ONI (El Niño/La Niña indicator) values\n\n"
                "⚠️ *Ang mga prediction ay ML-based na tantiya gamit ang historical patterns — hindi ito opisyal na weather forecast. Para sa real-time na forecast, suriin ang **PAGASA** (www.pagasa.dost.gov.ph).*"
            )

    # CL-NEW-ML-MODEL. What ML model is used in this chatbot?
    if any(k in clean_input for k in [
        "what ml model", "what machine learning model", "which ml model", "which model is used",
        "what model is used", "what model does this use", "what model does this chatbot use",
        "what algorithm", "what ai model", "random forest", "what is the model",
        "ano ang ml model", "anong modelo", "anong ml model", "anong machine learning model",
        "anong algorithm", "anong modelo ang ginagamit", "modelo ng chatbot"
    ]):
        conversation_state["last_topic"] = "ml_model_used"
        return L(
            "🤖 **What ML Model Is Used in This Chatbot?**\n\n"
            "This chatbot uses a **Random Forest Regressor** — a machine learning algorithm from the `scikit-learn` Python library — to predict monthly rainfall for Padre Garcia, Batangas.\n\n"
            "🌳 **What is a Random Forest?**\n"
            "A Random Forest is an ensemble learning method that builds many decision trees during training and combines their outputs to produce a more accurate and stable prediction. It works well with tabular climate data and handles non-linear patterns effectively.\n\n"
            "📊 **Model details:**\n"
            "• **Type:** Random Forest Regressor (supervised regression)\n"
            "• **Task:** Predict monthly rainfall in millimeters (mm)\n"
            "• **Training data:** Climate records from Padre Garcia, Batangas (2010–2026)\n"
            "• **Input features:** Month, average temperature, regional rainfall (Quezon/Bicol), ONI value\n"
            "• **Output:** Estimated monthly rainfall for Padre Garcia\n"
            "• **Saved as:** `climate_rain_model.pkl` (via `joblib`)\n\n"
            "💡 *You can ask: 'How accurate is the model?' or 'What data is used for prediction?' to learn more.*",
            "🤖 **Anong ML Model ang Ginagamit sa Chatbot na Ito?**\n\n"
            "Gumagamit ang chatbot na ito ng **Random Forest Regressor** — isang machine learning algorithm mula sa `scikit-learn` Python library — para mahulaan ang monthly rainfall para sa Padre Garcia, Batangas.\n\n"
            "🌳 **Ano ang Random Forest?**\n"
            "Ang Random Forest ay isang ensemble learning method na nagtatayo ng maraming decision trees sa panahon ng training at pinagsasama ang kanilang mga output para makagawa ng mas tumpak at matatag na prediction. Epektibo ito sa tabular climate data at kaya nitong harapin ang mga non-linear na pattern.\n\n"
            "📊 **Mga detalye ng modelo:**\n"
            "• **Uri:** Random Forest Regressor (supervised regression)\n"
            "• **Gawain:** Hulaan ang monthly rainfall sa millimeters (mm)\n"
            "• **Training data:** Climate records mula Padre Garcia, Batangas (2010–2026)\n"
            "• **Mga input features:** Buwan, average temperature, regional rainfall (Quezon/Bicol), ONI value\n"
            "• **Output:** Tinatayang monthly rainfall para sa Padre Garcia\n"
            "• **Na-save bilang:** `climate_rain_model.pkl` (gamit ang `joblib`)\n\n"
            "💡 *Maaari kang magtanong: 'Gaano katumpak ang modelo?' o 'Anong data ang ginagamit para sa prediction?' para matuto pa.*"
        )

    if any(k in clean_input for k in [
        "how accurate is the prediction", "how accurate is rainfall prediction",
        "how accurate is the rainfall", "how accurate is rainfall",
        "accuracy of prediction", "prediction accuracy", "how reliable is the prediction",
        "how accurate is this chatbot", "is the prediction accurate",
        "how accurate is the model", "how accurate is the ml", "model accuracy",
        "accurate is the production", "rainfall production", "accurate production",
        "gaano katumpak ang prediction", "tumpak ba ang prediction",
        "reliable ba ang prediction", "accuracy ng modelo", "how good is the prediction"
    ]):
        conversation_state["last_topic"] = "prediction_accuracy"
        return L(
            "📏 **How Accurate Is the Rainfall Prediction?**\n\n"
            "The ML model used here is a **Random Forest Regressor** trained on climate data from Padre Garcia, Batangas (2010–2026). Here's an honest assessment:\n\n"
            "✅ **What it does well:**\n"
            "• Captures seasonal patterns — months with historically high/low rainfall are generally predicted well\n"
            "• Accounts for El Niño/La Niña signals through ONI values\n"
            "• Trained on multi-station data for better regional context\n\n"
            "⚠️ **Limitations to know:**\n"
            "• It cannot predict sudden or extreme weather events (typhoons, unusual storms)\n"
            "• Accuracy decreases further into the future — predictions for next month are more reliable than 2 years out\n"
            "• It does not account for climate change trends over time\n"
            "• It is a local model — useful for Padre Garcia, but not for other regions\n\n"
            "📊 *The model is best used as a general guide and planning tool — not as a precise day-to-day forecast.*\n\n"
            "🌐 *For official, high-accuracy forecasts, always refer to **PAGASA** (Philippine Atmospheric, Geophysical and Astronomical Services Administration) at www.pagasa.dost.gov.ph.*",
            "📏 **Gaano Katumpak ang Rainfall Prediction?**\n\n"
            "Ang ML model na ginagamit dito ay isang **Random Forest Regressor** na sinanay sa climate data mula Padre Garcia, Batangas (2010–2026). Narito ang tapat na pagsusuri:\n\n"
            "✅ **Kung ano ang magaling nito:**\n"
            "• Nakukuha ang mga seasonal patterns — ang mga buwan na may historically mataas/mababang rainfall ay karaniwang tama ang prediction\n"
            "• Isinasaalang-alang ang mga El Niño/La Niña signals sa pamamagitan ng ONI values\n"
            "• Sinanay sa multi-station data para sa mas mahusay na regional context\n\n"
            "⚠️ **Mga limitasyon na dapat malaman:**\n"
            "• Hindi nito mahuhulaan ang biglaang o extreme na weather events (bagyo, hindi karaniwang bagyo)\n"
            "• Bumababa ang accuracy habang mas malayo sa hinaharap — ang predictions para sa susunod na buwan ay mas maaasahan kaysa sa 2 taon mula ngayon\n"
            "• Hindi nito isinasaalang-alang ang mga climate change trends sa paglipas ng panahon\n"
            "• Ito ay lokal na modelo — kapaki-pakinabang para sa Padre Garcia, ngunit hindi para sa ibang rehiyon\n\n"
            "📊 *Pinakamainam gamitin ang modelo bilang pangkalahatang gabay at kasangkapan sa pagpaplano — hindi bilang tumpak na araw-araw na forecast.*\n\n"
            "🌐 *Para sa opisyal, mataas na katumpak na forecast, laging sumangguni sa **PAGASA** (Philippine Atmospheric, Geophysical and Astronomical Services Administration) sa www.pagasa.dost.gov.ph.*"
        )

    if any(k in clean_input for k in [
        "what data is used for prediction", "what data is used", "data used for prediction",
        "what data does the model use", "what inputs does the model use",
        "where does the data come from", "data source prediction", "model data",
        "anong data ang ginagamit", "saan galing ang data", "data ng prediction",
        "what is the data behind", "what climate data is used"
    ]):
        conversation_state["last_topic"] = "prediction_data"
        return L(
            "📊 **What Data Is Used for Rainfall Prediction?**\n\n"
            "The ML rainfall prediction model was trained using climate data from **three weather stations** covering the years **2010–2026**.\n\n"
            "📍 **Data sources:**\n"
            "• **Padre Garcia, Batangas** — the primary station and prediction target\n"
            "• **Quezon / Bicol Region** — regional rainfall for broader context\n"
            "• **Basco, Batanes** — northern Philippines reference station\n"
            "• **ENSO/ONI dataset** — monthly Oceanic Niño Index values (from NOAA)\n\n"
            "🔢 **Features used by the model (inputs):**\n"
            "1. **Month** — the calendar month (1–12) captures seasonal patterns\n"
            "2. **Temperature** — average monthly temperature at Padre Garcia\n"
            "3. **Regional rainfall** — rainfall from the Quezon/Bicol station as a regional proxy\n"
            "4. **ONI value** — the current ENSO phase indicator (El Niño, Neutral, or La Niña)\n\n"
            "🎯 **Target (what the model predicts):**\n"
            "• Monthly rainfall in **millimeters (mm)** for Padre Garcia, Batangas\n\n"
            "🛠️ *All data was cleaned, merged, and saved in `chatbot_master_climate_data.csv` for use by the model.*",
            "📊 **Anong Data ang Ginagamit para sa Rainfall Prediction?**\n\n"
            "Ang ML rainfall prediction model ay sinanay gamit ang climate data mula sa **tatlong weather station** na sumasaklaw sa mga taong **2010–2026**.\n\n"
            "📍 **Mga pinagkukunan ng data:**\n"
            "• **Padre Garcia, Batangas** — ang pangunahing estasyon at target ng prediction\n"
            "• **Quezon / Bicol Region** — regional rainfall para sa mas malawak na konteksto\n"
            "• **Basco, Batanes** — northern Philippines reference station\n"
            "• **ENSO/ONI dataset** — monthly Oceanic Niño Index values (mula sa NOAA)\n\n"
            "🔢 **Mga features na ginagamit ng model (inputs):**\n"
            "1. **Buwan** — ang calendar month (1–12) ay kumukuha ng seasonal patterns\n"
            "2. **Temperature** — average monthly temperature sa Padre Garcia\n"
            "3. **Regional rainfall** — ulan mula sa Quezon/Bicol station bilang regional proxy\n"
            "4. **ONI value** — ang kasalukuyang ENSO phase indicator (El Niño, Neutral, o La Niña)\n\n"
            "🎯 **Target (ang hinuhulaan ng model):**\n"
            "• Monthly rainfall sa **millimeters (mm)** para sa Padre Garcia, Batangas\n\n"
            "🛠️ *Ang lahat ng data ay na-clean, na-merge, at na-save sa `chatbot_master_climate_data.csv` para gamitin ng model.*"
        )

    if any(k in clean_input for k in [
        "difference between historical and predicted", "historical vs predicted",
        "predicted vs historical", "difference between historical rainfall and predicted",
        "historical data vs prediction", "what is the difference between historical and predicted",
        "pagkakaiba ng historical at predicted", "historical at predicted rainfall",
        "ano ang pagkakaiba ng historical at predicted"
    ]):
        conversation_state["last_topic"] = "historical_vs_predicted"
        return L(
            "📊 **What Is the Difference Between Historical and Predicted Rainfall?**\n\n"
            "Great question — these two types of data serve different purposes:\n\n"
            "🗂️ **Historical Rainfall:**\n"
            "• Data that was **actually recorded** by weather stations in the past\n"
            "• Covers **2010–2026** for Padre Garcia, Batangas\n"
            "• 100% accurate — it reflects what really happened\n"
            "• Used to see trends, compare years, and identify El Niño/La Niña periods\n"
            "• Example: *'What was the rainfall in June 2015?'*\n\n"
            "🤖 **Predicted Rainfall:**\n"
            "• An **estimate** generated by the ML model based on learned patterns\n"
            "• Used for **future months or years** where no recorded data exists yet\n"
            "• Based on historical patterns, temperature, regional data, and ONI\n"
            "• Not guaranteed — it's a mathematical estimate, not a certainty\n"
            "• Example: *'Predict rainfall for May 2027'*\n\n"
            "🔑 **Key difference:**\n"
            "• Historical = what **happened** → exact values\n"
            "• Predicted = what **might happen** → an informed estimate\n\n"
            "💡 *Use historical data to understand the past and identify patterns. Use predictions as a planning guide for the future — and always complement with official PAGASA forecasts.*",
            "📊 **Ano ang Pagkakaiba ng Historical at Predicted Rainfall?**\n\n"
            "Magandang tanong — ang dalawang uri ng data na ito ay may iba't ibang layunin:\n\n"
            "🗂️ **Historical Rainfall:**\n"
            "• Data na **talagang naitala** ng mga weather station sa nakaraan\n"
            "• Sumasaklaw sa **2010–2026** para sa Padre Garcia, Batangas\n"
            "• 100% tumpak — repleksyon ng talagang nangyari\n"
            "• Ginagamit para makita ang mga trend, ihambing ang mga taon, at matukoy ang mga El Niño/La Niña period\n"
            "• Halimbawa: *'Ano ang ulan sa Hunyo 2015?'*\n\n"
            "🤖 **Predicted Rainfall:**\n"
            "• Isang **tantiya** na ginawa ng ML model batay sa mga natutunan na pattern\n"
            "• Ginagamit para sa **mga darating na buwan o taon** kung saan wala pang naitala na data\n"
            "• Batay sa historical patterns, temperature, regional data, at ONI\n"
            "• Hindi garantisado — ito ay mathematical na tantiya, hindi katiyakan\n"
            "• Halimbawa: *'Hulaan ang ulan para sa Mayo 2027'*\n\n"
            "🔑 **Pangunahing pagkakaiba:**\n"
            "• Historical = kung ano ang **nangyari** → eksaktong values\n"
            "• Predicted = kung ano ang **maaaring mangyari** → isang may batayan na tantiya\n\n"
            "💡 *Gamitin ang historical data para maunawaan ang nakaraan at matukoy ang mga pattern. Gamitin ang mga prediction bilang gabay sa pagpaplano para sa hinaharap — at laging dagdagan ng opisyal na PAGASA forecast.*"
        )

    # CL4-1. Where can I ask for assistance during El Niño?
    if any(k in clean_input for k in [
        "ask for assistance", "where can i ask", "where to ask", "saan humingi ng tulong",
        "humingi ng tulong", "seek assistance", "where to get help", "saan makakakuha ng tulong",
        "assistance during el nino", "tulong sa el nino", "saan tulong", "maghanap ng tulong",
    ]):
        # Don't intercept farmer-specific assistance questions
        if any(k in clean_input for k in ["farmers", "farmer", "magsasaka", "agricultural", "farm"]):
            pass
        else:
            conversation_state["last_topic"] = "where_assistance"
            return L(
                "🆘 **Where Can I Ask for Assistance During El Niño?**\n\n"
            "Several local and national offices can help during El Niño. Here are your key points of contact in **Padre Garcia, Batangas**:\n\n"
            "🏛️ **Local Government (LGU) Offices:**\n"
            "• **Barangay Hall** — first point of contact for community-level concerns; can connect you to relief goods, water supply, or evacuation\n"
            "• **MDRRMO** (Municipal Disaster Risk Reduction and Management Office) — coordinates disaster response, water distribution, and community preparedness\n"
            "• **MAO** (Municipal Agricultural Office) — assists farmers with drought relief, seeds, and livelihood programs\n\n"
            "🏥 **Health Concerns:**\n"
            "• **RHU** (Rural Health Unit) — contact for heat-related illnesses, dehydration, and health advisories\n\n"
            "💧 **Water Concerns:**\n"
            "• **MWD / Local Waterworks** — report water supply issues or request water delivery\n\n"
            "📢 **National Agencies:**\n"
            "• **PAGASA** — seasonal forecasts and weather advisories. Hotline: **1553**\n"
            "• **DSWD** (Department of Social Welfare and Development) — livelihood and relief assistance\n"
            "• **DA** (Department of Agriculture) — drought assistance for farmers\n\n"
            "📞 **Emergency:** National Emergency Hotline **911** | Red Cross **143**",
            "🆘 **Saan Humingi ng Tulong Sa Panahon ng El Niño?**\n\n"
            "Maraming lokal at pambansang opisina ang maaaring tumulong sa panahon ng El Niño. Narito ang mga pangunahing contact sa **Padre Garcia, Batangas**:\n\n"
            "🏛️ **Mga Opisina ng Local Government (LGU):**\n"
            "• **Barangay Hall** — unang contact para sa community-level na concerns; makakatulong sa relief goods, water supply, o evacuation\n"
            "• **MDRRMO** (Municipal Disaster Risk Reduction and Management Office) — nag-co-coordinate ng disaster response, water distribution, at community preparedness\n"
            "• **MAO** (Municipal Agricultural Office) — tumutulong sa mga farmers sa drought relief, binhi, at livelihood programs\n\n"
            "🏥 **Para sa Kalusugan:**\n"
            "• **RHU** (Rural Health Unit) — contact para sa heat-related illnesses, dehydration, at health advisories\n\n"
            "💧 **Para sa Tubig:**\n"
            "• **MWD / Local Waterworks** — mag-ulat ng water supply issues o humiling ng water delivery\n\n"
            "📢 **Pambansang Ahensya:**\n"
            "• **PAGASA** — seasonal forecasts at weather advisories. Hotline: **1553**\n"
            "• **DSWD** — livelihood at relief assistance\n"
            "• **DA** (Department of Agriculture) — drought assistance para sa mga farmers\n\n"
            "📞 **Emergency:** National Emergency Hotline **911** | Red Cross **143**"
        )

    # CL4-2. What is the MDRRMO hotline number?
    if any(k in clean_input for k in [
        "mdrrmo hotline", "mdrrmo number", "mdrrmo contact", "mdrrmo phone",
        "contact mdrrmo", "call mdrrmo", "mdrrmo telephone", "numero ng mdrrmo",
        "mdrrmo hotline number", "mdrrmo", "how to contact mdrrmo",
    ]):
        conversation_state["last_topic"] = "mdrrmo_contact"
        return L(
            "🛡️ **MDRRMO Contact — Padre Garcia, Batangas**\n\n"
            "The **MDRRMO** (Municipal Disaster Risk Reduction and Management Office) is your primary contact for disaster preparedness, emergency response, and El Niño-related concerns.\n\n"
            "📞 **MDRRMO Hotline:** **((043) 515-7424)**\n\n"
            "🏛️ **Office Location:** Municipal Hall, Padre Garcia, Batangas\n\n"
            "📋 **What MDRRMO can help with:**\n"
            "• El Niño and drought preparedness coordination\n"
            "• Emergency water distribution\n"
            "• Relief goods and community assistance\n"
            "• Evacuation center management\n"
            "• Connecting residents to relevant government agencies\n\n"
            "📢 *Hotline will be updated once confirmed with the LGU. In the meantime, visit your **Barangay Hall** for immediate assistance.*\n\n"
            "📞 **Other emergency numbers:**\n"
            "• National Emergency Hotline: **911**\n"
            "• Red Cross: **143**\n"
            "• PAGASA: **1553**",
            "🛡️ **MDRRMO Contact — Padre Garcia, Batangas**\n\n"
            "Ang **MDRRMO** (Municipal Disaster Risk Reduction and Management Office) ang inyong pangunahing contact para sa disaster preparedness, emergency response, at mga concerns na may kaugnayan sa El Niño.\n\n"
            "📞 **MDRRMO Hotline:** **((043) 515-7424)**\n\n"
            "🏛️ **Lokasyon ng Opisina:** Municipal Hall, Padre Garcia, Batangas\n\n"
            "📋 **Ano ang maitutulong ng MDRRMO:**\n"
            "• Coordination ng El Niño at drought preparedness\n"
            "• Emergency water distribution\n"
            "• Relief goods at community assistance\n"
            "• Pamamahala ng evacuation center\n"
            "• Pagkonekta ng mga residente sa mga kaugnay na ahensya ng gobyerno\n\n"
            "📢 *Ia-update ang hotline pagkatapos makumpirma sa LGU. Sa ngayon, bisitahin ang inyong **Barangay Hall** para sa agarang tulong.*\n\n"
            "📞 **Iba pang emergency numbers:**\n"
            "• National Emergency Hotline: **911**\n"
            "• Red Cross: **143**\n"
            "• PAGASA: **1553**"
        )

    # CL4-3. How can I contact the RHU?
    if any(k in clean_input for k in [
        "contact the rhu", "contact rhu", "rhu contact", "rhu hotline", "rhu number",
        "rural health unit", "rhu phone", "how to reach rhu", "call rhu",
        "numero ng rhu", "makipag-ugnayan sa rhu", "rhu padre garcia",
    ]):
        conversation_state["last_topic"] = "rhu_contact"
        return L(
            "🏥 **How Can I Contact the RHU?**\n\n"
            "The **RHU** (Rural Health Unit) of Padre Garcia, Batangas is your local public health resource for heat-related illnesses, dehydration, and other health concerns during El Niño.\n\n"
            "📞 **RHU Contact:** **((043) 515-9209)**\n\n"
            "🏛️ **Location:** Padre Garcia, Batangas (near the Municipal Hall)\n\n"
            "🕐 **Operating Hours:** Monday – Friday, 8:00 AM – 5:00 PM\n"
            "*(For after-hours emergencies, proceed to the nearest hospital or call 911)*\n\n"
            "🩺 **What the RHU can help with during El Niño:**\n"
            "• Treatment of heat exhaustion and heat stroke\n"
            "• Oral rehydration therapy for dehydration\n"
            "• Health advisories during extreme heat\n"
            "• Referrals to hospitals for severe cases\n"
            "• Free or subsidized medicines for eligible residents\n\n"
            "📞 **Emergency numbers:**\n"
            "• National Emergency Hotline: **911**\n"
            "• Red Cross: **143**",
            "🏥 **Paano Makipag-ugnayan sa RHU?**\n\n"
            "Ang **RHU** (Rural Health Unit) ng Padre Garcia, Batangas ang inyong lokal na pampublikong health resource para sa heat-related illnesses, dehydration, at iba pang health concerns sa panahon ng El Niño.\n\n"
            "📞 **RHU Contact:** **((043) 515-9209)**\n\n"
            "🏛️ **Lokasyon:** Padre Garcia, Batangas (malapit sa Municipal Hall)\n\n"
            "🕐 **Operating Hours:** Lunes – Biyernes, 8:00 AM – 5:00 PM\n"
            "*(Para sa after-hours emergencies, pumunta sa pinakamalapit na ospital o tumawag sa 911)*\n\n"
            "🩺 **Ano ang maitutulong ng RHU sa panahon ng El Niño:**\n"
            "• Paggamot ng heat exhaustion at heat stroke\n"
            "• Oral rehydration therapy para sa dehydration\n"
            "• Health advisories sa panahon ng matinding init\n"
            "• Referral sa ospital para sa mga malubhang kaso\n"
            "• Libre o subsidized na gamot para sa mga eligible na residente\n\n"
            "📞 **Mga emergency number:**\n"
            "• National Emergency Hotline: **911**\n"
            "• Red Cross: **143**"
        )

    # CL4-4 & CL4-6. Where to report water shortages / what to do during water shortages
    if any(k in clean_input for k in [
        "report water shortage", "report water", "where to report water",
        "saan mag-ulat ng water shortage", "mag-ulat ng kulang na tubig",
        "what to do during water shortage", "water shortage what to do",
        "kapag kulang ang tubig", "kapag walang tubig", "kung wala nang tubig",
        "water shortage tips", "during water shortage", "water rationing",
        "tubig kulang", "how to cope water shortage", "water shortages",
        "what should i do during water shortage", "what should i do during water shortages",
    ]):
        conversation_state["last_topic"] = "water_shortage_action"
        return L(
            "💧 **Where to Report & What to Do During Water Shortages**\n\n"
            "**📣 Where to Report Water Shortages:**\n"
            "• **Barangay Hall** — report to your barangay captain or tanod immediately\n"
            "• **Municipal Waterworks / MWD** — report supply interruptions or pipe issues\n"
            "• **MDRRMO** — for large-scale or emergency water supply concerns\n"
            "• **Mayor's Office** — if the issue affects a wide area\n\n"
            "✅ **What to Do While Waiting for Supply to Return:**\n"
            "• Use stored water for drinking and cooking first — prioritize safe drinking water\n"
            "• Switch to sponge baths to reduce water use\n"
            "• Reuse water where safe (e.g. rinse water for flushing)\n"
            "• Ask your barangay if emergency water trucks or refilling stations are available\n"
            "• Coordinate with neighbors — share water if someone has excess stored supply\n\n"
            "🚰 **For Drinking Water:**\n"
            "• Boil water for at least 1 minute before drinking if unsure of quality\n"
            "• Use water purification tablets as backup\n"
            "• Check if your LGU is distributing water during the shortage\n\n"
            "📞 **MDRRMO Hotline:** (043) 515-7424 | **National Emergency:** 911",
            "💧 **Saan Mag-ulat at Ano ang Dapat Gawin sa Panahon ng Water Shortage**\n\n"
            "**📣 Saan Mag-ulat ng Water Shortage:**\n"
            "• **Barangay Hall** — iulat agad sa inyong barangay captain o tanod\n"
            "• **Municipal Waterworks / MWD** — iulat ang supply interruptions o problema sa pipes\n"
            "• **MDRRMO** — para sa malakihan o emergency na water supply concerns\n"
            "• **Mayor's Office** — kung nakakaapekto ang isyu sa malawak na lugar\n\n"
            "✅ **Ano ang Gagawin Habang Hinihintay ang Pagbabalik ng Supply:**\n"
            "• Gamitin muna ang nakaimbak na tubig para sa pag-inom at pagluluto — unahin ang ligtas na drinking water\n"
            "• Lumipat sa sponge bath para mabawasan ang paggamit ng tubig\n"
            "• I-reuse ang tubig kung ligtas (hal. rinse water para sa pag-flush)\n"
            "• Tanungin ang inyong barangay kung may emergency water trucks o refilling stations\n"
            "• Makipagtulungan sa mga kapitbahay — magbahaginan ng tubig kung may sobra\n\n"
            "🚰 **Para sa Drinking Water:**\n"
            "• Pakuluin ang tubig ng hindi bababa sa 1 minuto bago inumin kung hindi sigurado sa kalidad\n"
            "• Gumamit ng water purification tablets bilang backup\n"
            "• Suriin kung namamahagi ang inyong LGU ng tubig sa panahon ng shortage\n\n"
            "📞 **MDRRMO Hotline:** (043) 515-7424 | **National Emergency:** 911"
        )

    # CL4-5. Where can farmers ask for assistance?
    if any(k in clean_input for k in [
        "farmers ask for assistance", "farmers seek help", "where can farmers",
        "farmer assistance", "agricultural assistance", "saan humingi ng tulong ang magsasaka",
        "tulong para sa magsasaka", "farm assistance", "farmer help drought",
        "agricultural support", "DA assistance", "MAO assistance",
    ]):
        conversation_state["last_topic"] = "farmer_assistance"
        return L(
            "🌾 **Where Can Farmers Ask for Assistance During El Niño / Drought?**\n\n"
            "Farmers in Padre Garcia have several options for government and community support:\n\n"
            "🏛️ **Local Offices:**\n"
            "• **MAO** (Municipal Agricultural Office), Padre Garcia — first point of contact for all agricultural concerns. They can provide seeds, fertilizer subsidies, and drought advisories.\n"
            "• **Barangay Hall** — can connect farmers to relief programs and local livelihood support\n"
            "• **MDRRMO** — coordinates relief assistance including food and water for affected farming communities\n\n"
            "🌱 **National Agencies:**\n"
            "• **DA** (Department of Agriculture) — drought assistance, crop insurance (PhilCrop), and seed distribution programs\n"
            "• **DSWD** — livelihood and food assistance for severely affected farmer households\n"
            "• **NIA** (National Irrigation Administration) — for irrigation water access concerns\n\n"
            "📋 **Programs to Ask About:**\n"
            "• **PhilCrop** — crop insurance program for losses due to El Niño or drought\n"
            "• **SURE Aid** (DA) — immediate cash assistance for farmers affected by calamities\n"
            "• **Seed Subsidy Programs** — drought-tolerant seed varieties distributed during El Niño\n\n"
            "📞 **DA Hotline:** (02) 8844-0000 | **MDRRMO:**  (043) 515-7424",
            "🌾 **Saan Humingi ng Tulong ang mga Magsasaka sa Panahon ng El Niño / Drought?**\n\n"
            "Maraming opsyon ang mga magsasaka sa Padre Garcia para sa suporta ng gobyerno at komunidad:\n\n"
            "🏛️ **Mga Lokal na Opisina:**\n"
            "• **MAO** (Municipal Agricultural Office), Padre Garcia — unang contact para sa lahat ng agricultural concerns. Makapagbibigay ng mga binhi, fertilizer subsidies, at drought advisories.\n"
            "• **Barangay Hall** — makakakonekta sa mga farmers sa relief programs at lokal na livelihood support\n"
            "• **MDRRMO** — nag-co-coordinate ng relief assistance kasama na ang pagkain at tubig para sa mga affected na farming communities\n\n"
            "🌱 **Pambansang Ahensya:**\n"
            "• **DA** (Department of Agriculture) — drought assistance, crop insurance (PhilCrop), at seed distribution programs\n"
            "• **DSWD** — livelihood at food assistance para sa malubhang affected na farmer households\n"
            "• **NIA** (National Irrigation Administration) — para sa mga concerns tungkol sa access sa irrigation water\n\n"
            "📋 **Mga Programa na Maaaring Itanong:**\n"
            "• **PhilCrop** — crop insurance program para sa mga pagkalugi dahil sa El Niño o drought\n"
            "• **SURE Aid** (DA) — agarang cash assistance para sa mga farmers na apektado ng kalamidad\n"
            "• **Seed Subsidy Programs** — drought-tolerant na varieties ng binhi na ibinibigay sa panahon ng El Niño\n\n"
            "📞 **DA Hotline:** (02) 8844-0000 | **MDRRMO:**  (043) 515-7424"
        )

    # CL4-7. Where can I get info about local preparedness programs?
    if any(k in clean_input for k in [
        "local preparedness programs", "preparedness programs", "government programs el nino",
        "community programs", "local programs", "where to get information",
        "saan makakakuha ng impormasyon", "lokal na programa", "preparedness information",
        "community preparedness", "where can i learn", "saan makakaalam",
        "information about programs", "lgu programs",
    ]):
        conversation_state["last_topic"] = "local_programs"
        return L(
            "📋 **Where Can I Get Information About Local El Niño Preparedness Programs?**\n\n"
            "Several channels provide updates on preparedness programs in Padre Garcia and Batangas:\n\n"
            "🏛️ **Visit or Contact These Offices:**\n"
            "• **Barangay Hall** — your barangay regularly receives and distributes information on relief and preparedness programs\n"
            "• **MDRRMO** (Municipal Disaster Risk Reduction and Management Office) — manages and announces all local disaster preparedness activities. Hotline: **(043) 515-7424**\n"
            "• **MAO** (Municipal Agricultural Office) — for farming-specific programs like seed subsidies and drought support\n"
            "• **RHU** (Rural Health Unit) — for health-related advisories and programs during El Niño\n\n"
            "📢 **Stay Informed Through:**\n"
            "• **Barangay public announcements / tanod** — local PA systems and community boards\n"
            "• **Municipal Facebook page / official social media** — many LGUs post advisories and program announcements online\n"
            "• **PAGASA website and hotline (1553)** — national weather advisories and seasonal outlooks\n"
            "• **NDRRMC** (National Disaster Risk Reduction and Management Council) — national-level preparedness updates\n\n"
            "💡 *Don't wait for emergencies — ask your barangay captain about upcoming El Niño preparedness activities in your area.*",
            "📋 **Saan Makakakuha ng Impormasyon Tungkol sa Lokal na El Niño Preparedness Programs?**\n\n"
            "Maraming channel ang nagbibigay ng updates tungkol sa preparedness programs sa Padre Garcia at Batangas:\n\n"
            "🏛️ **Bisitahin o Makipag-ugnayan sa mga Opisinang Ito:**\n"
            "• **Barangay Hall** — regular na tumatanggap at nagbabahagi ng impormasyon tungkol sa relief at preparedness programs ang inyong barangay\n"
            "• **MDRRMO** — namamahala at nag-a-announce ng lahat ng lokal na disaster preparedness activities. Hotline: **(043) 515-7424**\n"
            "• **MAO** — para sa farming-specific programs tulad ng seed subsidies at drought support\n"
            "• **RHU** — para sa health-related advisories at programs sa panahon ng El Niño\n\n"
            "📢 **Manatiling Informed Sa Pamamagitan ng:**\n"
            "• **Barangay public announcements / tanod** — lokal na PA systems at community boards\n"
            "• **Municipal Facebook page / official social media** — maraming LGU ang nagpo-post ng advisories at program announcements online\n"
            "• **PAGASA website at hotline (1553)** — pambansang weather advisories at seasonal outlooks\n"
            "• **NDRRMC** — pambansang preparedness updates\n\n"
            "💡 *Huwag maghintay ng emergency — tanungin na ngayon ang inyong barangay captain tungkol sa upcoming El Niño preparedness activities sa inyong lugar.*"
        )

    # CL4-8. Who should I contact during an emergency related to El Niño?
    if any(k in clean_input for k in [
        "contact during emergency", "who to contact emergency", "contact during el nino emergency", "contact during an el nino emergency",
        "who should i call", "emergency contact el nino", "sino ang tatawagin",
        "contact for emergency", "emergency number el nino", "who to call emergency",
        "contact in case of emergency", "help during emergency", "emergency el nino contact",
        "who should i contact during", "sino ang dapat tawagan",
    ]):

        conversation_state["last_topic"] = "emergency_contacts"
        return L(
            "📞 **Who Should I Contact During an El Niño-Related Emergency?**\n\n"
            "For emergencies in Padre Garcia, Batangas, contact the appropriate local office below. Save these numbers on your phone for quick access.\n\n"

            "🚨 **LOCAL EMERGENCY HOTLINES – PADRE GARCIA**\n\n"

            "🛡️ **Municipal Disaster Risk Reduction and Management Office (MDRRMO)**\n"
            "• Mobile: **0977-801-1557 / 0919-611-6180**\n"
            "• Landline: **(043) 515-7424**\n"
            "• Facebook: MDRRMO Padre Garcia\n"
            "• Location: Barangay Poblacion, Padre Garcia, Batangas\n"
            "• Contact Persons: Hon. Celsa Braga-Rivera (Municipal Mayor and MDRRMO Chairperson); Sir Abuel Mendoza (MDRRMO Head)\n\n"

            "🚒 **Padre Garcia Fire Station (BFP)**\n"
            "• Emergency Hotline: **177**\n"
            "• Mobile: **0966-871-5388**\n"
            "• Landline: **(043) 515-1177**\n"
            "• Facebook: BFP Riva Padre Garcia FS\n"
            "• Location: Barangay San Miguel, Padre Garcia, Batangas\n"
            "• Contact Person: SFO4 Rachel S. Nicasio, Acting Municipal Fire Marshal\n\n"

            "🚔 **Padre Garcia Police Station (PNP)**\n"
            "• Emergency Hotline: **166**\n"
            "• Mobile: **0928-663-7762**\n"
            "• Landline: **(043) 515-9209**\n"
            "• Facebook: PNP Padre Garcia Municipal Police Station\n"
            "• Location: Barangay San Miguel, Padre Garcia, Batangas\n"
            "• Contact Person: PMAJ. Frank Davis R. Tuddao, COP\n\n"

            "🏥 **Municipal Rural Health Unit (RHU)**\n"
            "• Mobile: **0917-188-3838**\n"
            "• Landline: **(043) 515-9209**\n"
            "• Facebook: Padre Garcia Rural Health Unit\n"
            "• Location: Municipal Compound, Barangay Poblacion, Padre Garcia, Batangas\n"
            "• Contact Person: Dr. Anne Kristine Vallejo\n\n"

            "🏥 **New Padre Garcia Hospital**\n"
            "• Landline: **(043) 772-0437**\n"
            "• Facebook: New Padre Garcia Hospital\n"
            "• Location: LGU Compound, Barangay Castillo, Padre Garcia, Batangas\n"
            "• Contact Person: Dr. Dianne M. Cuevas\n\n"

            "💙 **Padre Garcia Dialysis Center**\n"
            "• Mobile: **0954-406-8373**\n"
            "• Facebook: Padre Garcia Dialysis Center\n"
            "• Location: LGU Compound, Barangay Castillo, Padre Garcia, Batangas\n"
            "• Contact Person: Sir Nilo Atienza\n\n"

            "🤝 **Municipal Social Welfare and Development Office (MSWDO)**\n"
            "• Mobile: **0933-879-3128**\n"
            "• Landline: **(043) 515-9341**\n"
            "• Facebook: MSWDO Padre Garcia\n"
            "• Location: Municipal Compound, Barangay Poblacion, Padre Garcia, Batangas\n"
            "• Contact Person: Ms. Wilma Polancos Galela\n\n"

            "🌍 **OTHER IMPORTANT NATIONAL HOTLINES**\n"
            "• National Emergency Hotline: **911**\n"
            "• Philippine Red Cross: **143**\n"
            "• NDRRMC Operations Center: **(02) 8911-1406**\n"
            "• DOH Emergency Hotline: **1555**\n"
            "• DA Hotline: **(02) 8844-0000**\n\n"

            "💧 **Water Supply Problems:** Contact your Barangay Hall or the MDRRMO for assistance.\n"
            "🌾 **Agricultural Concerns:** Contact the Municipal Agricultural Office (MAO) through the Municipal Hall.\n"
            "🌤️ **Weather Updates:** Follow official PAGASA announcements and advisories.\n\n"

            "⚠️ **IMPORTANT REMINDERS:**\n"
            "• Call the MDRRMO for disaster-related concerns, drought impacts, and emergency assistance.\n"
            "• Contact the RHU or hospital for health-related emergencies.\n"
            "• Call the appropriate emergency responders when immediate assistance is needed.\n"
            "• Follow verified LGU and government announcements for updated information.\n\n"
            "📢 **Stay informed, stay prepared, and keep these emergency contacts accessible!**",

            "📞 **Sino ang Dapat Tawagan sa Panahon ng Emergency na May Kaugnayan sa El Niño?**\n\n"
            "Para sa mga emergency sa Padre Garcia, Batangas, makipag-ugnayan sa tamang lokal na tanggapan gamit ang mga numerong nakalista sa ibaba. I-save ang mga ito sa inyong cellphone para madaling matawagan.\n\n"

            "🚨 **MGA LOKAL NA EMERGENCY HOTLINE – PADRE GARCIA**\n\n"

            "🛡️ **Municipal Disaster Risk Reduction and Management Office (MDRRMO)**\n"
            "• Mobile: **0977-801-1557 / 0919-611-6180**\n"
            "• Landline: **(043) 515-7424**\n"
            "• Facebook: MDRRMO Padre Garcia\n"
            "• Lokasyon: Barangay Poblacion, Padre Garcia, Batangas\n"
            "• Mga Contact Person: Hon. Celsa Braga-Rivera (Municipal Mayor at MDRRMO Chairperson); Sir Abuel Mendoza (MDRRMO Head)\n\n"

            "🚒 **Padre Garcia Fire Station (BFP)**\n"
            "• Emergency Hotline: **177**\n"
            "• Mobile: **0966-871-5388**\n"
            "• Landline: **(043) 515-1177**\n"
            "• Facebook: BFP Riva Padre Garcia FS\n"
            "• Lokasyon: Barangay San Miguel, Padre Garcia, Batangas\n"
            "• Contact Person: SFO4 Rachel S. Nicasio, Acting Municipal Fire Marshal\n\n"

            "🚔 **Padre Garcia Police Station (PNP)**\n"
            "• Emergency Hotline: **166**\n"
            "• Mobile: **0928-663-7762**\n"
            "• Landline: **(043) 515-9209**\n"
            "• Facebook: PNP Padre Garcia Municipal Police Station\n"
            "• Lokasyon: Barangay San Miguel, Padre Garcia, Batangas\n"
            "• Contact Person: PMAJ. Frank Davis R. Tuddao, COP\n\n"

            "🏥 **Municipal Rural Health Unit (RHU)**\n"
            "• Mobile: **0917-188-3838**\n"
            "• Landline: **(043) 515-9209**\n"
            "• Facebook: Padre Garcia Rural Health Unit\n"
            "• Lokasyon: Municipal Compound, Barangay Poblacion, Padre Garcia, Batangas\n"
            "• Contact Person: Dr. Anne Kristine Vallejo\n\n"

            "🏥 **New Padre Garcia Hospital**\n"
            "• Landline: **(043) 772-0437**\n"
            "• Facebook: New Padre Garcia Hospital\n"
            "• Lokasyon: LGU Compound, Barangay Castillo, Padre Garcia, Batangas\n"
            "• Contact Person: Dr. Dianne M. Cuevas\n\n"

            "💙 **Padre Garcia Dialysis Center**\n"
            "• Mobile: **0954-406-8373**\n"
            "• Facebook: Padre Garcia Dialysis Center\n"
            "• Lokasyon: LGU Compound, Barangay Castillo, Padre Garcia, Batangas\n"
            "• Contact Person: Sir Nilo Atienza\n\n"

            "🤝 **Municipal Social Welfare and Development Office (MSWDO)**\n"
            "• Mobile: **0933-879-3128**\n"
            "• Landline: **(043) 515-9341**\n"
            "• Facebook: MSWDO Padre Garcia\n"
            "• Lokasyon: Municipal Compound, Barangay Poblacion, Padre Garcia, Batangas\n"
            "• Contact Person: Ms. Wilma Polancos Galela\n\n"

            "🌍 **IBA PANG MAHALAGANG NATIONAL HOTLINES**\n"
            "• National Emergency Hotline: **911**\n"
            "• Philippine Red Cross: **143**\n"
            "• NDRRMC Operations Center: **(02) 8911-1406**\n"
            "• DOH Emergency Hotline: **1555**\n"
            "• DA Hotline: **(02) 8844-0000**\n\n"

            "💧 **Problema sa Suplay ng Tubig:** Makipag-ugnayan sa inyong Barangay Hall o MDRRMO para sa tulong.\n"
            "🌾 **Problema sa Agrikultura:** Makipag-ugnayan sa Municipal Agricultural Office (MAO) sa pamamagitan ng Municipal Hall.\n"
            "🌤️ **Update sa Panahon:** Sundan ang mga opisyal na anunsyo at weather advisory ng PAGASA.\n\n"

            "⚠️ **MAHAHALAGANG PAALALA:**\n"
            "• Tawagan ang MDRRMO para sa mga problemang may kaugnayan sa sakuna, tagtuyot, at emergency assistance.\n"
            "• Makipag-ugnayan sa RHU o ospital para sa mga emergency na may kinalaman sa kalusugan.\n"
            "• Tawagan ang tamang emergency responders kung kinakailangan ng agarang tulong.\n"
            "• Sundin ang mga beripikadong anunsyo mula sa LGU at mga opisyal na ahensya ng gobyerno.\n\n"
            "📢 **Maging alerto, laging handa, at panatilihing accessible ang mga emergency contact number na ito!**"
        )

    # CL4-9. What local government services are available during El Niño?
    if any(k in clean_input for k in [
        "local government services", "lgu services", "government services el nino",
        "government help el nino", "what services available", "what does the lgu",
        "what can the government", "government support el nino", "serbisyo ng lgu",
        "lokal na serbisyo", "serbisyo ng pamahalaan", "anong serbisyo", "lgu during el nino",
        "local services el nino", "government services available",
    ]):
        conversation_state["last_topic"] = "lgu_services"
        return L(
            "🏛️ **What Local Government Services Are Available During El Niño?**\n\n"
            "The LGU of Padre Garcia, along with barangay governments, activates several services during El Niño:\n\n"
            "💧 **Water Services:**\n"
            "• Emergency water trucking to affected communities\n"
            "• Coordination with waterworks for priority supply restoration\n"
            "• Water refilling station referrals\n\n"
            "🌾 **Agricultural Support:**\n"
            "• Seed and fertilizer subsidies for affected farmers (via MAO)\n"
            "• Drought advisories and crop guidance\n"
            "• Referral to DA programs (PhilCrop insurance, SURE Aid)\n\n"
            "🏥 **Health Services:**\n"
            "• Free treatment for heat-related illnesses at the RHU\n"
            "• Distribution of oral rehydration salts (ORS)\n"
            "• Community health advisories during extreme heat\n\n"
            "🍚 **Food & Livelihood:**\n"
            "• Food packs for severely affected households (via DSWD and MDRRMO)\n"
            "• Livelihood programs for families whose income is disrupted\n\n"
            "📢 **Information & Early Warning:**\n"
            "• Barangay-level announcements and advisories\n"
            "• MDRRMO coordinates with PAGASA for local weather updates\n"
            "• Community preparedness meetings and drills\n\n"
            "🏠 **How to Access These Services:**\n"
            "• Visit or call your **Barangay Hall** first — they will direct you to the right office\n"
            "• Contact **MDRRMO** at (043) 515-7424 for emergency coordination\n"
            "• Check the **Municipal Facebook page** for service announcements",
            "🏛️ **Anong mga Serbisyo ng Local Government ang Available sa Panahon ng El Niño?**\n\n"
            "Ang LGU ng Padre Garcia, kasama ang mga barangay government, ay nag-a-activate ng ilang serbisyo sa panahon ng El Niño:\n\n"
            "💧 **Serbisyo sa Tubig:**\n"
            "• Emergency water trucking sa mga apektadong komunidad\n"
            "• Koordinasyon sa waterworks para sa priority supply restoration\n"
            "• Referral sa mga water refilling station\n\n"
            "🌾 **Suporta sa Agrikultura:**\n"
            "• Seed at fertilizer subsidies para sa mga apektadong farmers (sa pamamagitan ng MAO)\n"
            "• Drought advisories at crop guidance\n"
            "• Referral sa mga DA programs (PhilCrop insurance, SURE Aid)\n\n"
            "🏥 **Serbisyo sa Kalusugan:**\n"
            "• Libreng paggamot ng heat-related illnesses sa RHU\n"
            "• Pamamahagi ng oral rehydration salts (ORS)\n"
            "• Community health advisories sa panahon ng matinding init\n\n"
            "🍚 **Pagkain at Kabuhayan:**\n"
            "• Food packs para sa malubhang apektadong mga pamilya (sa pamamagitan ng DSWD at MDRRMO)\n"
            "• Livelihood programs para sa mga pamilyang naputol ang kita\n\n"
            "📢 **Impormasyon at Early Warning:**\n"
            "• Mga anunsyo at advisory sa antas ng barangay\n"
            "• Nakikipagtulungan ang MDRRMO sa PAGASA para sa lokal na weather updates\n"
            "• Mga community preparedness meetings at drills\n\n"
            "🏠 **Paano Ma-access ang mga Serbisyong Ito:**\n"
            "• Bisitahin o tawagan muna ang inyong **Barangay Hall** — ire-redirect nila kayo sa tamang opisina\n"
            "• Makipag-ugnayan sa **MDRRMO** sa (043) 515-7424 para sa emergency coordination\n"
            "• Suriin ang **Municipal Facebook page** para sa mga service announcements"
        )

    # Drought definition — must be before section 9 so "drought" alone doesn't hit el nino handler
    if any(k in clean_input for k in ["what is drought", "define drought", "ano ang drought", "meaning of drought", "drought definition", "what does drought mean", "drought mean", "tagtuyot ano", "ano ang tagtuyot"]):
        conversation_state["last_topic"] = "drought_definition"
        return L(
            "🏜️ **What is Drought?**\n\n"
            "**Drought** is a prolonged period of below-normal rainfall that causes serious water shortages for people, crops, and the environment.\n\n"
            "It is not just one dry day — drought develops gradually over weeks or months when rainfall is consistently far below average.\n\n"
            "💧 **Types of drought:**\n"
            "• **Meteorological drought** — Rainfall is significantly below normal for an extended period\n"
            "• **Agricultural drought** — Soil moisture is too low to support crop growth\n"
            "• **Hydrological drought** — Rivers, reservoirs, and groundwater levels drop critically\n\n"
            "🇵🇭 **In the Philippines:**\n"
            "Drought is one of the most damaging natural hazards, especially during **El Niño** years. It reduces water supply for drinking and irrigation, causes crop failures, and increases wildfire risk.\n\n"
            "📍 *In Padre Garcia, Batangas, drought conditions are most likely during El Niño months when local rainfall drops significantly below its monthly average.*",
            "🏜️ **Ano ang Drought?**\n\n"
            "Ang **drought** (tagtuyot) ay isang mahabang panahon ng below-normal na rainfall na nagdudulot ng malubhang water shortages para sa mga tao, pananim, at kalikasan.\n\n"
            "Hindi lang ito isang araw na walang ulan — unti-unting nagde-develop ang drought sa loob ng mga linggo o buwan kapag ang rainfall ay patuloy na mas mababa sa average.\n\n"
            "💧 **Mga uri ng drought:**\n"
            "• **Meteorological drought** — Ang rainfall ay malayo sa normal sa mahabang panahon\n"
            "• **Agricultural drought** — Ang soil moisture ay masyadong mababa para suportahan ang paglaki ng pananim\n"
            "• **Hydrological drought** — Critically bumababa ang levels ng mga ilog, reservoir, at groundwater\n\n"
            "🇵🇭 **Sa Pilipinas:**\n"
            "Ang drought ay isa sa mga pinaka-mapanira na natural hazards, lalo na sa panahon ng **El Niño**. Binabawasan nito ang water supply para sa inumin at irrigation, nagdudulot ng crop failures, at nagpapataas ng wildfire risk.\n\n"
            "📍 *Sa Padre Garcia, Batangas, ang drought conditions ay pinakamalamang sa panahon ng El Niño months kapag ang local rainfall ay malaki ang pagbaba mula sa monthly average.*"
        )

    # 9. ENSO + station matchers
    if any(k in clean_input for k in ["el nino", "drought", "dry", "tagtuyot", "tagtuyo", "tuyo"]):
        # Don't intercept ENSO–El Niño relationship questions — handled above
        if any(k in clean_input for k in [
            "enso", "related to el nino", "how is enso", "enso and el nino",
            "el nino and enso", "enso vs el nino", "el nino vs enso",
            "relationship between enso", "relationship between el nino and enso",
            "how does enso", "enso relate", "enso connection", "enso link",
        ]):
            pass
        # Don't intercept reduce-rainfall questions — handled above
        elif any(k in clean_input for k in [
            "reduce rainfall", "reduces rainfall", "less rainfall", "less rain",
            "lower rainfall", "rainfall decrease", "decrease rainfall",
            "bawasan ang ulan", "nagbabawas ng ulan", "kaunting ulan",
            "bumababa ang ulan", "pagbaba ng ulan",
        ]):
            pass
        # Don't intercept "signs" questions — handled above
        elif any(k in clean_input for k in [
            "signs", "sign", "palatandaan", "tanda", "how to know",
            "paano malaman", "paano malalaman", "nagsimula na ba",
        ]):
            pass
        # Don't intercept "prepare/preparation" questions — handled above
        elif any(k in clean_input for k in [
            "prepare", "preparation", "ihanda", "paghahanda", "maghanda",
            "what should i prepare", "ano ang dapat ihanda",
        ]):
            pass
        # Don't intercept "emergency contact" questions — handled below
        elif any(k in clean_input for k in [
            "contact", "call", "tawagan", "emergency", "emerhen", "hotline",
            "who should i contact", "who to contact", "sino ang tatawagin",
        ]):
            pass
        elif padre_garcia_only:
            return handle_el_nino_padre_garcia()
        else:
            # If Station Comparison Mode is active, show all 3 stations
            if conversation_state.get("station_mode", False):
                return handle_el_nino_general()
            # Otherwise, default to Padre Garcia
            return handle_el_nino_padre_garcia()

    if any(k in clean_input for k in ["la nina", "flood", "pagbaha"]) or re.search(r'\bbaha\b', clean_input):
        return handle_la_nina_general()

    if any(k in clean_input for k in ["rain", "precipitation", "rainfall", "wettest", "ulan", "pag-ulan", "pinakamalakas na ulan"]):
        # Don't intercept rainwater harvesting questions — let them fall to KB
        if any(k in clean_input for k in ["harvest", "rainwater", "collect", "store water", "impok"]):
            pass
        # Don't intercept month comparison questions
        elif any(k in clean_input for k in ["compare", "vs", "versus", "ihambing", "pagkakaiba", "difference between"]) and any(k in clean_input for k in ["month", "months", "buwan"]):
            pass
        # Don't intercept "last year" / "previous year" — let them fall to year query
        elif any(k in clean_input for k in ["last year", "previous year", "noong nakaraang taon", "nakaraang taon", "last yr", "nakaraang taon"]):
            pass
        # Don't intercept prediction/forecast questions
        elif any(k in clean_input for k in ["predict", "forecast", "hulaan", "prediksyon"]):
            pass
        else:
            return handle_rain_query(padre_garcia_only)

    if any(k in clean_input for k in ["temp", "temperature", "hot", "heat", "warmest", "highest temperature", "temperatura", "pinakamainit"]):
        return handle_temp_query(padre_garcia_only)

    if any(k in clean_input for k in ["stations", "station", "what are the 3", "three stations", "monitored stations", "mga istasyon", "tatlong istasyon", "istasyon"]):
        return handle_stations_overview()

    # 3-station El Niño comparison (explicit request for all stations)
    if any(k in clean_input for k in [
        "el nino all stations", "el nino 3 stations", "el nino across stations",
        "el nino sa lahat ng stations", "el nino sa tatlong stations",
        "compare el nino across stations", "el nino comparison stations",
        "show all stations el nino", "3 station el nino"
    ]):
        return handle_el_nino_general()

    # 10. Conversational memory follow-up
    if conversation_state["last_topic"] in ["el_nino", "el_nino_definition"] and any(w in clean_input for w in ["affect", "impact", "environment", "consequence", "result", "why", "epekto", "pinsala", "bakit", "resulta"]):
        return L(
            "⚠️ **El Niño Impacts on Padre Garcia:**\n"
            "• **Ecosystem Strain:** Drought conditions dry out forest cover, elevating wildfire risks.\n"
            "• **Water Depletion:** Low precipitation reduces river discharge and reservoir levels.\n"
            "• **Livelihood Disruption:** Farmers face reduced crop productivity — early preparedness is key.",
            "⚠️ **Mga Epekto ng El Niño sa Padre Garcia:**\n"
            "• **Ecosystem Strain:** Ang drought conditions ay nagpapatuyo sa forest cover, nagpapataas ng wildfire risk.\n"
            "• **Water Depletion:** Ang low rainfall ay nagbabawas sa river discharge at reservoir levels.\n"
            "• **Livelihood Disruption:** Nababawasan ang crop productivity ng mga farmers — ang early preparedness ay mahalaga."
        )

    # 11. Knowledge-base fallback: if none of the rules above matched,
    # search the SQLite knowledge base before giving up.
    try:
        matches = db.search_knowledge_base(clean_input, limit=1)
    except Exception:
        matches = []
    if matches:
        kb = matches[0]
        conversation_state["last_topic"] = f"kb:{kb['category']}"
        if conversation_state.get("lang") == "tl" and kb.get("answer_tl"):
            return kb["answer_tl"]
        return kb["answer_en"]

    # Fire emergency handler — out of scope but important safety response
    if any(k in clean_input for k in [
        "house on fire", "fire", "sunog", "nasusunog", "nasunog", "may sunog",
        "burning", "nakikita ko ang usok", "smoke", "bfp", "bureau of fire",
        "fire station", "fire truck", "pemapadpad", "apoy",
    ]):
        return L(
            "🚒 **Fire Emergency — Call for Help Immediately!**\n\n"
            "This chatbot focuses on climate and El Niño topics, but for a fire emergency:\n\n"
            "📞 **Call these numbers right away:**\n"
            "• **BFP Emergency Hotline: 177** *(Bureau of Fire Protection)*\n"
            "• **Padre Garcia Fire Station: 0966-871-5388** or **(043) 515-1177**\n"
            "• **National Emergency Hotline: 911**\n\n"
            "🏃 **While waiting for help:**\n"
            "• Get everyone out of the building immediately\n"
            "• Do not go back inside for any belongings\n"
            "• Alert your neighbors\n"
            "• Meet at a safe distance away from the building\n\n"
            "📍 *Padre Garcia Fire Station is located at Barangay San Miguel, Padre Garcia, Batangas.*",
            "🚒 **Emergency sa Sunog — Humingi ng Tulong Agad!**\n\n"
            "Ang chatbot na ito ay nakatuon sa climate at El Niño na mga paksa, ngunit para sa emergency na sunog:\n\n"
            "📞 **Tumawag agad sa mga numerong ito:**\n"
            "• **BFP Emergency Hotline: 177** *(Bureau of Fire Protection)*\n"
            "• **Padre Garcia Fire Station: 0966-871-5388** o **(043) 515-1177**\n"
            "• **National Emergency Hotline: 911**\n\n"
            "🏃 **Habang hinihintay ang tulong:**\n"
            "• Palabasin agad ang lahat mula sa gusali\n"
            "• Huwag bumalik sa loob para sa anumang bagay\n"
            "• Abisuhan ang inyong mga kapitbahay\n"
            "• Magtipun-tipon sa ligtas na distansya mula sa gusali\n\n"
            "📍 *Ang Padre Garcia Fire Station ay matatagpuan sa Barangay San Miguel, Padre Garcia, Batangas.*"
        )

    return L(
        "I'm not sure I understood that because it's out of my knowledge. Here are some things you can ask me:\n"
        "• *'What is El Niño?'*\n"
        "• *'What are the effects of El Niño?'*\n"
        "• *'What is the difference between El Niño and La Niña?'*\n"
        "• *'What is ReadyKomunidad?'*\n"
        "• *'How was the weather in 2015?'*\n"
        "• *'Compare 2015 and 2020'*\n"
        "• *'Predict rain for May'*",
        "Hindi ko naintindihan yun kase hindi na s'ya abot ng aking kakayanan. Narito ang ilang bagay na maaari mong itanong:\n"
        "• *'Ano ang El Niño?'*\n"
        "• *'Ano ang mga epekto ng El Niño?'*\n"
        "• *'Ano ang pagkakaiba ng El Niño at La Niña?'*\n"
        "• *'Ano ang ReadyKomunidad?'*\n"
        "• *'Kumusta ang weather noong 2015?'*\n"
        "• *'Ihambing ang 2015 at 2020'*\n"
        "• *'Hulaan ang ulan para sa May'*"
    )

@app.route('/')
def home():
    if os.path.exists('readykomunidad.html'):
        with open('readykomunidad.html', 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "Chatbot Web Interface Ready"
        
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    message = data.get('message', '')
    session_id = data.get('session_id') or str(uuid.uuid4())

    # 1. Generate normal chatbot reply
    reply = chatbot_event_router(message)

    # 2. Mirror user's emojis
    is_tl = conversation_state.get('lang') == 'tl'
    reply = apply_emoji_mirroring(reply, message, is_tl=is_tl)

    # 3. Append related questions automatically based on topic/keywords
    topic = conversation_state.get('last_topic', '')
    related_chips = generate_related_questions(topic, message, reply, is_tl=is_tl)
    if related_chips:
        reply = reply + "\n" + related_chips

    # Persist both sides to SQLite
    try:
        db.log_message(session_id, 'user', message,
                       lang=conversation_state.get('lang'))
        db.log_message(session_id, 'bot', reply,
                       topic=conversation_state.get('last_topic'))
    except Exception as e:
        print(f"[chatlog] could not save message: {e}")

    return jsonify({'reply': reply, 'session_id': session_id})
    
@app.route('/api/history', methods=['GET'])
def history():
    """Return the saved chatlog for a session so the UI can restore it
    after a page refresh. Maps DB rows to the frontend's {sender, text, time}."""
    session_id = request.args.get('session_id') or ''
    if not session_id:
        return jsonify({'messages': []})
    try:
        rows = db.get_chatlog(session_id)
    except Exception as e:
        print(f"[history] error: {e}")
        return jsonify({'messages': []})
    return jsonify({
        'messages': [
            {'sender': r['role'], 'text': r['message'], 'time': r['created_at']}
            for r in rows
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting Web Chatbot Server on http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
