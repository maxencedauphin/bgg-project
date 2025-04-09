import streamlit as st
import plotly.graph_objects as go
from streamlit.components.v1 import html

# Importer les fonctions de prédiction depuis le nouveau fichier
from prediction import (
    # load_prediction_model,
    predict_game_rating,
    prepare_input_data,
    GAME_MECHANICS,
    GAME_DOMAINS, load_prediction_from_remote
)

# Function to add the Le Wagon logo to the sidebar
def add_logo_to_sidebar():
    st.sidebar.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-image: url("https://raw.githubusercontent.com/lewagon/fullstack-images/master/uikit/logo.png");
            background-repeat: no-repeat;
            background-position: 15px 20px;
            background-size: 150px auto;
            padding-top: 120px;
        }
        </style>
    """, unsafe_allow_html=True)

def create_home_page_css():
    return """
    <style>
    /* Main container for the home page */
    .home-container {
        background-image: url("https://thumbs.dreamstime.com/z/board-games-hand-draw-doodle-background-vector-illustration-147131823.jpg?ct=jpeg");
        background-size: 114%;  /* Zoomed to 150% of original size */
        background-position: center;  /* Keep centered */
        padding: 10rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        position: relative;
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Overlay to make text more readable */
    .home-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.3);  /* Semi-transparent black overlay - darkened more */
        border-radius: 15px;
    }
    
    /* Title styling */
    .home-title {
        position: relative;
        z-index: 1;
        color: #ffffff !important;  /* Force white color with !important */
        text-align: center;
        font-size: 3.5rem;  /* Increased font size */
        font-weight: 800;  /* Extra bold */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 15px;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.9), 
                     0px 0px 30px rgba(255, 255, 255, 0.5);  /* Enhanced glow effect */
        letter-spacing: 1px;
    }
    
    /* Title text specific styling */
    .title-text {
        color: #ffffff !important;  /* Force white color with !important */
        font-family: 'Arial', sans-serif;  /* More readable font */
        -webkit-text-fill-color: white;  /* For webkit browsers */
    }
    
    /* Animation to make the dice bounce */
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-15px) rotate(10deg);
        }
    }
    
    /* Animation to make the dice spin */
    @keyframes spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
    
    /* Combined animation for the dice */
    .animated-dice {
        display: inline-block;
        font-size: 4.5rem;  /* Larger icon */
        filter: drop-shadow(2px 2px 5px rgba(0, 0, 0, 0.7));
        margin-bottom: 15px;  /* More space below icon */
        animation: bounce 2s infinite ease-in-out;
        transform-origin: center;
    }
    
    /* Content box styling */
    .content-box {
        background-color: white;
        border-radius: 15px;
        padding: 2rem;
        color: #333;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
        margin: 1rem auto;
        max-width: 800px;
    }
    
    /* Magical story box styling */
    .magical-story-box {
        background: linear-gradient(135deg, #1a0033, #3a0066, #1a0033);
        border-radius: 15px;
        padding: 2.5rem;
        color: #fff;
        box-shadow: 0 8px 32px rgba(78, 0, 146, 0.5), 
                   inset 0 0 80px rgba(180, 120, 255, 0.2);
        margin: 1rem auto;
        max-width: 800px;
        position: relative;
        overflow: hidden;
        font-family: 'Cinzel', serif;
        letter-spacing: 0.5px;
        line-height: 1.8;
    }
    
    /* Add magical particles effect */
    .magical-story-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('https://www.transparenttextures.com/patterns/stardust.png'), 
                          url('https://www.transparenttextures.com/patterns/asfalt-light.png');
        background-blend-mode: screen;
        opacity: 0.15;
        z-index: 0;
        animation: backgroundShift 20s infinite alternate;
    }
    
    @keyframes backgroundShift {
        0% { background-position: 0% 0%; }
        100% { background-position: 100% 100%; }
    }
    
    /* Style for magical story title */
    .magical-title {
        font-family: 'Cinzel Decorative', 'Cinzel', serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        background: linear-gradient(120deg, #e6c0ff, #ffffff, #c9a0ff);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 10px rgba(180, 120, 255, 0.7);
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    
    /* Style for magical story content */
    .magical-content {
        position: relative;
        z-index: 1;
        font-size: 1.15rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        font-weight: 400;
    }
    
    /* Style for magical story paragraphs */
    .magical-content p {
        margin-bottom: 1.2rem;
    }
    
    /* Style for magical icons */
    .magical-icon {
        display: inline-block;
        margin: 0 5px;
        font-size: 1.4rem;
        animation: magicPulse 2s infinite;
        filter: drop-shadow(0 0 5px rgba(180, 120, 255, 0.8));
        vertical-align: middle;
    }
    
    @keyframes magicPulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    /* Style for magical list */
    .magical-list {
        list-style-type: none;
        padding-left: 1rem;
        margin: 1.5rem 0;
    }
    
    .magical-list-item {
        margin-bottom: 1rem;
        position: relative;
        padding-left: 2rem;
    }
    
    .magical-list-item::before {
        content: "✦";
        position: absolute;
        left: 0;
        color: #b380ff;
        font-size: 1.2rem;
        animation: starTwinkle 3s infinite;
    }
    
    @keyframes starTwinkle {
        0% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0.5; transform: scale(1); }
    }
    
    /* Style for magical highlights */
    .magical-highlight {
        background: linear-gradient(120deg, rgba(180, 120, 255, 0.2), rgba(140, 80, 255, 0.3));
        padding: 0 5px;
        border-radius: 4px;
        font-weight: 600;
        color: #f0e6ff;
    }
    
    /* Add floating sparkles */
    .sparkle {
        position: absolute;
        width: 3px;
        height: 3px;
        border-radius: 50%;
        background-color: white;
        box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8);
        animation: float 6s infinite;
        z-index: 2;
    }
    
    .sparkle:nth-child(1) {
        top: 20%;
        left: 10%;
        animation-delay: 0s;
    }
    
    .sparkle:nth-child(2) {
        top: 30%;
        left: 85%;
        animation-delay: 1s;
    }
    
    .sparkle:nth-child(3) {
        top: 70%;
        left: 20%;
        animation-delay: 2s;
    }
    
    .sparkle:nth-child(4) {
        top: 80%;
        left: 75%;
        animation-delay: 3s;
    }
    
    .sparkle:nth-child(5) {
        top: 40%;
        left: 50%;
        animation-delay: 4s;
    }
    
    @keyframes float {
        0% { transform: translateY(0) scale(1); opacity: 0; }
        25% { transform: translateY(-20px) scale(1.2); opacity: 1; }
        50% { transform: translateY(-40px) scale(1); opacity: 0.6; }
        75% { transform: translateY(-60px) scale(1.2); opacity: 0.3; }
        100% { transform: translateY(-80px) scale(1); opacity: 0; }
    }
    
    /* Animation for color change from dark red to orange to purple to black */
    @keyframes colorChange {
        0% { background-color: #8B0000; }  /* Dark Red */
        33% { background-color: #FF8C00; }  /* Dark Orange */
        66% { background-color: #800080; }  /* Purple */
        100% { background-color: #000000; }  /* Black */
    }
      
    /* Animation for globe rotation */
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Animation for pulsation effect */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Animation for rainbow text effect */
    @keyframes rainbowText {
        0% { color: #8B0000; }  /* Dark Red */
        33% { color: #FF8C00; }  /* Dark Orange */
        66% { color: #800080; }  /* Purple */
        100% { color: #000000; }  /* Black */
    }
    
    /* Style for custom Streamlit button - Globe style with custom color animation */
    div[data-testid="stButton"] > button:first-child {
        background: #8B0000;  /* Start with dark red */
        color: white;
        font-weight: bold;
        border-radius: 50%;
        width: 90px;
        height: 90px;
        padding: 0;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5),
                    inset 0 0 20px rgba(255, 255, 255, 0.5);
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
        animation: colorChange 4s infinite alternate;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
    }
    
    /* Animated text inside button */
    div[data-testid="stButton"] > button:first-child span {
        position: relative;
        z-index: 2;
        animation: rainbowText 4s infinite alternate;
    }
    
    /* Create shadow effect for globe */
    div[data-testid="stButton"] > button:first-child::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, 
                    rgba(255, 255, 255, 0.4) 0%, 
                    rgba(255, 255, 255, 0.1) 30%, 
                    rgba(0, 0, 0, 0.1) 70%,
                    rgba(0, 0, 0, 0.4) 100%);
        border-radius: 50%;
        z-index: 1;
    }
    
    /* Create a rotating border effect with custom colors */
    div[data-testid="stButton"] > button:first-child::before {
        content: "";
        position: absolute;
        top: -4px;
        left: -4px;
        right: -4px;
        bottom: -4px;
        background: linear-gradient(45deg, 
            #8B0000, #FF8C00, #800080, #000000);
        background-size: 400% 400%;
        border-radius: 50%;
        z-index: -1;
        animation: rotate 3s linear infinite, colorChange 4s infinite alternate;
        filter: blur(4px);
    }
    
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-5px) rotate(5deg);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.6),
                    inset 0 0 30px rgba(255, 255, 255, 0.5);
        cursor: pointer;
        animation-play-state: paused;
    }
    
    div[data-testid="stButton"] > button:hover::before {
        animation-play-state: running;
        animation-duration: 1s;
    }
    
    /* Numbered list styling */
    ol {
        padding-left: 1.5rem;
    }
    
    ol li {
        margin-bottom: 0.5rem;
    }
    
    /* Style to center the gauge chart */
    .gauge-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto;
        max-width: 400px;
    }
    
    /* Game Summary box styling */
    .summary-box {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border-left: 5px solid #4CAF50;
    }
    
    /* Info box styling for term explanations */
    .info-box {
        background-color: #e8f4fd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0d6efd;
    }
    
    .info-box h4 {
        color: #0d6efd;
        margin-top: 0;
    }
    
    /* Custom styles for Streamlit elements to make them match our design */
    div.stAlert > div:first-child {
        padding: 1.5rem;
        border-radius: 10px;
    }
    
    /* Adventure story box styling */
    .adventure-box {
        background: linear-gradient(135deg, #2c3e50, #4a69bd);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4), 
                   inset 0 0 60px rgba(255, 255, 255, 0.1);
        margin: 1.5rem auto;
        position: relative;
        overflow: hidden;
    }
    
    .adventure-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('https://img.freepik.com/free-vector/hand-drawn-mystical-background_23-2149389405.jpg?w=1380&t=st=1680541276~exp=1680541876~hmac=a3b3ef0c2b3c7f5d7b508e6bcf4c29ca6e8f58a8a92f5f88b8a57240d6f49692');
        background-size: cover;
        background-position: center;
        opacity: 0.2;
        z-index: 0;
    }
    
    .adventure-title {
        position: relative;
        z-index: 1;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .adventure-content {
        position: relative;
        z-index: 1;
        font-size: 1.1rem;
        line-height: 1.6;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    .magic-icon {
        display: inline-block;
        margin: 0 5px;
        animation: pulse 2s infinite;
    }
    
    .power-list {
        margin-top: 1rem;
        padding-left: 1rem;
    }
    
    .power-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.8rem;
        font-weight: 500;
    }
    
    .power-icon {
        display: inline-block;
        margin-right: 10px;
        font-size: 1.3rem;
        animation: pulse 2s infinite;
    }
    
    .highlight {
        background: linear-gradient(120deg, rgba(255,215,0,0.2), rgba(255,215,0,0.3));
        padding: 0 5px;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .magic-sparkle {
        position: absolute;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background-color: white;
        box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8);
        animation: sparkle 4s infinite;
        z-index: 2;
    }
    
    @keyframes sparkle {
        0% { opacity: 0; transform: scale(0); }
        50% { opacity: 1; transform: scale(1); }
        100% { opacity: 0; transform: scale(0); }
    }
    
    /* Import Google fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Cinzel+Decorative:wght@700&display=swap');
    </style>
    """

# Function to create an rating gauge visualization
def create_improved_rating_gauge(rating):
    # Define color scheme for the gauge
    colors = {
        'poor': '#FF5252',          # Bright red
        'below_average': '#FFA726',  # Orange
        'average': '#FFEB3B',        # Yellow
        'good': '#66BB6A',           # Light green
        'excellent': '#2E7D32'       # Dark green
    }
    
    # Create a more appealing gauge chart
    fig = go.Figure()
    
    # Add background for the full scale (0-10)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=rating,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 2, 'tickcolor': "darkgrey"},
            'bar': {'color': "rgba(0,0,0,0)"},  # Transparent bar
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 3], 'color': colors['poor']},
                {'range': [3, 5], 'color': colors['below_average']},
                {'range': [5, 7], 'color': colors['average']},
                {'range': [7, 8.5], 'color': colors['good']},
                {'range': [8.5, 10], 'color': colors['excellent']}
            ],
            'threshold': {
                'line': {'color': "darkblue", 'width': 4},
                'thickness': 0.8,
                'value': rating
            }
        },
        number={
            'font': {'size': 40, 'color': '#1F2937', 'family': 'Arial'},  # Reduced size
            'suffix': '/10',
            'valueformat': '.2f'
        },
        title={
            'text': "Predicted Rating",
            'font': {'size': 19, 'color': '#1F2937', 'family': 'Arial'}  # Reduced size
        }
    ))
    
    # Add rating category text based on the predicted value
    rating_category = ""
    if rating < 3:
        rating_category = "Poor"
    elif rating < 5:
        rating_category = "Below Average"
    elif rating < 7:
        rating_category = "Average"
    elif rating < 8.5:
        rating_category = "Good"
    else:
        rating_category = "Excellent"
    
    # Add annotation for the rating category
    fig.add_annotation(
        x=0.5,
        y=0.25,
        text=f"Rating Category: <b>{rating_category}</b>",
        showarrow=False,
        font=dict(size=16)  # Reduced size
    )
    
    # Update layout with reduced height
    fig.update_layout(
        height=300,  # Reduced height (was 400)
        margin=dict(l=20, r=20, t=30, b=20),  # Reduced margins
        paper_bgcolor="white",
        font={'color': "#1F2937", 'family': "Arial"}
    )
    
    return fig

# Initialize the session variables if they don't exist
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Initialize variables to store game characteristics and prediction
if 'game_characteristics' not in st.session_state:
    st.session_state.game_characteristics = {}

if 'predicted_rating' not in st.session_state:
    st.session_state.predicted_rating = None

# Function to generate a hash for game characteristics to detect changes
def get_characteristics_hash(characteristics):
    import hashlib
    import json
    # Convert dict to a sorted string representation for consistent hashing
    sorted_chars = json.dumps(characteristics, sort_keys=True)
    return hashlib.md5(sorted_chars.encode()).hexdigest()

# Navigation management via the sidebar
add_logo_to_sidebar()

# Main function that runs the application
def main():
    # Define column names for prediction
    min_players_col = 'min_players'
    max_players_col = 'max_players'
    play_time_col = 'play_time'
    complexity_col = 'complexity_average'
    min_age_col = 'min_age'
    
    # Load model
    #model = load_prediction_model()

    # HOME PAGE
    if st.session_state.page == "Home":
        # Hide the sidebar on home page for cleaner look
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
        
        # Add CSS for home page styling
        st.markdown(create_home_page_css(), unsafe_allow_html=True)
        
        # Content - Banner and title
        st.markdown("""
        <div class="home-container">
            <h1 class="home-title">
                <div class="animated-dice">🎲</div>
                <span class="title-text">Board Game Success Predictor</span>
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        magical_story_html = """
<div style="background: linear-gradient(135deg, #1a0033, #3a0066, #1a0033); border-radius: 15px; padding: 2.5rem; color: #fff; box-shadow: 0 8px 32px rgba(78, 0, 146, 0.5), inset 0 0 80px rgba(180, 120, 255, 0.2); margin: 1rem auto; max-width: 850px; position: relative; overflow: hidden; font-family: 'Cinzel', serif; letter-spacing: 0.5px; line-height: 1.7;">
    <div style="position: absolute; width: 3px; height: 3px; border-radius: 50%; background-color: white; box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8); top: 20%; left: 10%; animation: float 6s infinite;"></div>
    <div style="position: absolute; width: 3px; height: 3px; border-radius: 50%; background-color: white; box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8); top: 30%; left: 85%; animation: float 6s infinite 1s;"></div>
    <div style="position: absolute; width: 3px; height: 3px; border-radius: 50%; background-color: white; box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8); top: 70%; left: 20%; animation: float 6s infinite 2s;"></div>
    <div style="position: absolute; width: 3px; height: 3px; border-radius: 50%; background-color: white; box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8); top: 80%; left: 75%; animation: float 6s infinite 3s;"></div>
    <div style="position: absolute; width: 3px; height: 3px; border-radius: 50%; background-color: white; box-shadow: 0 0 10px 2px rgba(255, 255, 255, 0.8); top: 40%; left: 50%; animation: float 6s infinite 4s;"></div>
    
    <h2 style="font-family: 'Cinzel', serif; font-size: 1.7rem; font-weight: 700; margin-bottom: 1.2rem; text-align: center; position: relative; z-index: 1; letter-spacing: 1px; color: white; white-space: nowrap;">
        <span style="display: inline-block; margin-right: 10px; font-size: 1.3rem;">🧙‍♂️</span> The Quest for the Legendary Game
    </h2>
    
    <div style="position: relative; z-index: 1; font-size: 1rem; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5); font-weight: 400;">
        <p>In a kingdom where board games reign supreme, you are the <span style="background: linear-gradient(120deg, rgba(180, 120, 255, 0.2), rgba(140, 80, 255, 0.3)); padding: 0 5px; border-radius: 4px; font-weight: 600; color: #f0e6ff;">Archmage of Gaming</span>, the greatest game creator of all time. The King has entrusted you with a crucial mission: create a game so extraordinary that it will unite all the peoples of the kingdom.</p>
        
        <p>But beware! Many paths lie before you, and only one combination of mechanics, complexity, and theme will lead you to success. Fortunately, you possess a magical artifact: <span style="display: inline-block; margin: 0 5px; font-size: 1.2rem;">🔮</span> the <span style="background: linear-gradient(120deg, rgba(180, 120, 255, 0.2), rgba(140, 80, 255, 0.3)); padding: 0 5px; border-radius: 4px; font-weight: 600; color: #f0e6ff;">Orb of Prediction</span> <span style="display: inline-block; margin: 0 5px; font-size: 1.2rem;">✨</span></p>
        
        <p>With this mystical orb, you can:</p>
        
        <ul style="list-style-type: none; padding-left: 0; margin: 1.2rem 0;">
            <li style="margin-bottom: 0.8rem; display: flex; align-items: flex-start;">
                <span style="display: inline-block; margin-right: 10px; font-size: 1.2rem; flex-shrink: 0;">🧪</span> 
                <span>Test different magical formulas before spending your precious development resources</span>
            </li>
            <li style="margin-bottom: 0.8rem; display: flex; align-items: flex-start;">
                <span style="display: inline-block; margin-right: 10px; font-size: 1.2rem; flex-shrink: 0;">⚔️</span> 
                <span>Optimize your creation's characteristics to maximize its appeal to players</span>
            </li>
            <li style="margin-bottom: 0.8rem; display: flex; align-items: flex-start;">
                <span style="display: inline-block; margin-right: 10px; font-size: 1.2rem; flex-shrink: 0;">🏆</span> 
                <span>Compare your game to market expectations and existing legends</span>
            </li>
        </ul>
    </div>
</div>
<style>
@keyframes float {
    0% { transform: translateY(0) scale(1); opacity: 0; }
    25% { transform: translateY(-20px) scale(1.2); opacity: 1; }
    50% { transform: translateY(-40px) scale(1); opacity: 0.6; }
    75% { transform: translateY(-60px) scale(1.2); opacity: 0.3; }
    100% { transform: translateY(-80px) scale(1); opacity: 0; }
}
</style>
"""
        
        # Use the html component to render the magical story
        html(magical_story_html, height=600)
        
        # How to use section
        st.markdown("""
        <div class="content-box">
            <h3>How to use:</h3>
            <ol>
                <li>Click the button below to go to the prediction page</li>
                <li>Adjust the sliders to match your game's characteristics</li>
                <li>Select the game domain and mechanics</li>
                <li>Get an instant prediction of your game's rating!</li>
            </ol>
            <div style="text-align: center;">
                <!-- The button will be added via Streamlit -->
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a Streamlit button that works with Streamlit navigation
        col1, col2, col3 = st.columns([1.5, 1, 1.5])
        with col2:
            if st.button("TRY ME!", key="start_predicting_button", 
                        type="primary", use_container_width=True):
                # Set the page to "Predictive Analysis" in the session state
                st.session_state.page = "Predictive Analysis"
                # Force page reload with st.rerun()
                st.rerun()
    # PREDICTIVE ANALYSIS PAGE
    elif st.session_state.page == "Predictive Analysis":
        st.header("🔮 Predictive Analysis")
        
        # Add term explanations
        with st.expander("📚 Understanding Key Terms", expanded=False):
            st.markdown("""
            <div class="info-box">
                <h4>What is a BoardGameGeek Rating?</h4>
                <p>BoardGameGeek (BGG) is the world's largest board game database and community. The BGG rating is a score from 1-10 that reflects how much players enjoy a game:</p>
                <ul>
                    <li><strong>1-3:</strong> Poor games that players generally dislike</li>
                    <li><strong>4-5:</strong> Below average games with significant flaws</li>
                    <li><strong>6-7:</strong> Average to good games that most players enjoy</li>
                    <li><strong>7-8.5:</strong> Very good games that are highly recommended</li>
                    <li><strong>8.5-10:</strong> Excellent games considered among the best</li>
                </ul>
                <p>A higher rating generally means better commercial success and player satisfaction.</p>
            </div>
            
            <div class="info-box">
                <h4>What are Game Mechanics?</h4>
                <p>Game mechanics are the rules and methods designed for interaction with the game state. They're the core systems that make a game function and create the gameplay experience. Examples include:</p>
                <ul>
                    <li><strong>Dice Rolling:</strong> Using dice to determine outcomes (e.g., Yahtzee)</li>
                    <li><strong>Worker Placement:</strong> Placing tokens to select actions (e.g., Agricola)</li>
                    <li><strong>Card Drafting:</strong> Selecting cards from a limited pool (e.g., 7 Wonders)</li>
                    <li><strong>Area Control:</strong> Competing for control of territories (e.g., Risk)</li>
                </ul>
                <p>Different mechanics appeal to different player types and can significantly impact a game's rating.</p>
            </div>
            
            <div class="info-box">
                <h4>What is Game Complexity?</h4>
                <p>Complexity (or weight) measures how difficult a game is to learn and play, rated from 1-5:</p>
                <ul>
                    <li><strong>1:</strong> Very simple (e.g., Uno, Candy Land)</li>
                    <li><strong>2:</strong> Easy to learn (e.g., Ticket to Ride, Carcassonne)</li>
                    <li><strong>3:</strong> Moderate complexity (e.g., Catan, Pandemic)</li>
                    <li><strong>4:</strong> Complex (e.g., Terraforming Mars, Scythe)</li>
                    <li><strong>5:</strong> Very complex (e.g., Twilight Imperium, Gloomhaven)</li>
                </ul>
                <p>The right complexity level depends on your target audience. Family games tend to be 1-2.5, while strategy games are often 3-4.5.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create form for user inputs
        with st.form("prediction_form"):
            st.write("Enter game characteristics to predict its rating:")
            
            # Create two columns for the form
            col1, col2 = st.columns(2)
            
            # Define default values for sliders
            min_players_min, min_players_max = 1, 8
            max_players_min, max_players_max = 2, 20
            play_time_min, play_time_max = 1, 240
            
            # First column inputs
            with col1:
                # Modified Min Players slider with more appropriate range and default
                min_players = st.slider("Min Players", min_players_min, min_players_max, 2, 
                                       help="Minimum number of players required for the game")
                max_players = st.slider("Max Players", max_players_min, max_players_max, 4,
                                       help="Maximum number of players supported by the game")
                play_time = st.slider("Play Time (minutes)", play_time_min, play_time_max, 60, step=1,
                                     help="Average time to complete one game")
            
            # Second column inputs
            with col2:
                # Changed "Min Age" to "Age" and range from 0-18 to 10-70
                min_age = st.slider("Age", 10, 70, 0,
                                   help="Minimum recommended age for players")
                complexity = st.slider("Complexity Average", 1.0, 5.0, 2.5, step=0.1,
                                      help="How complex the game is (1=simple, 5=complex)")
            
            # Game domain selection
            selected_domain = st.selectbox("Game Domain", GAME_DOMAINS,
                                          help="Primary category of the game")
            
            # Game mechanics selection (multi-select)
            st.subheader("Game Mechanics")
            st.write("Select the mechanics that apply to your game:")
            
            # List of common mechanics for the multi-select
            common_mechanics = GAME_MECHANICS
            
            selected_mechanics = st.multiselect(
                "Common Game Mechanics",
                options=common_mechanics,
                #default=['dice rolling', 'card drafting'],
                help="Select multiple mechanics that apply to your game"
            )            
                        
            # Submit button
            submitted = st.form_submit_button("Predict Rating")
        
        # Process form submission
        if submitted:
            # Collect current game characteristics
            current_characteristics = {
                min_players_col: min_players,
                max_players_col: max_players,
                play_time_col: play_time,
                min_age_col: min_age,
                complexity_col: complexity,
                'selected_domain': selected_domain,
                'selected_mechanics': sorted(selected_mechanics)  # Sort for consistent comparison
            }
            
            # Always generate a new prediction when characteristics change
            # Store the new characteristics
            st.session_state.game_characteristics = current_characteristics
            
            # Prepare input data for prediction
            input_data = prepare_input_data(
                min_players, max_players, play_time, min_age,
                complexity, selected_domain, selected_mechanics, common_mechanics
            )

            predicted_rating = load_prediction_from_remote(min_players, max_players, play_time, min_age, complexity, selected_domain, selected_mechanics)
            
            # Make prediction
            #predicted_rating = predict_game_rating(model, input_data)
            
            # Store the prediction in session state
            if predicted_rating is not None:
                st.session_state.predicted_rating = predicted_rating
            
            # Use the stored prediction
            if st.session_state.predicted_rating is not None:
                # Create a container to center the gauge chart
                st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
                
                # Create gauge chart
                gauge_fig = create_improved_rating_gauge(st.session_state.predicted_rating)
                st.plotly_chart(gauge_fig, use_container_width=False, config={'displayModeBar': False})
                
                # Close the container
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display summary in a shaded box
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.subheader("Game Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    # Changed "Min Age" to "Age"
                    st.write(f"**Age:** {min_age} years")
                    st.write(f"**Players:** {min_players} to {max_players} players")
                    st.write(f"**Domain:** {selected_domain}")
                
                with summary_col2:
                    st.write(f"**Complexity:** {complexity}/5")
                    st.write(f"**Play Time:** {play_time} minutes")
                    st.write(f"**Mechanics:** {', '.join(selected_mechanics[:3])}{'...' if len(selected_mechanics) > 3 else ''}")
                
                # Add market potential analysis based on rating
                st.markdown("### Market Potential Analysis")
                if st.session_state.predicted_rating >= 8.0:
                    st.success("⭐ **High Market Potential**: This game concept shows excellent potential for commercial success. Games with ratings above 8.0 often become bestsellers and receive strong community support.")
                elif st.session_state.predicted_rating >= 7.0:
                    st.info("✅ **Good Market Potential**: This game concept shows good potential. With some refinement, it could become quite successful in the market.")
                elif st.session_state.predicted_rating >= 6.0:
                    st.warning("⚠️ **Moderate Market Potential**: This game concept has average appeal. Consider enhancing certain aspects to improve its market potential.")
                else:
                    st.error("❌ **Limited Market Potential**: This game concept may struggle to find an audience. Consider significant revisions to core mechanics or complexity.")
                
                # Add design recommendations based on inputs
                st.markdown("### Design Recommendations")
                recommendations = []
                
                if complexity > 3.5 and min_age < 14:
                    recommendations.append("Consider increasing the recommended age or reducing complexity for better alignment")
                
                if play_time > 120 and complexity < 3.0:
                    recommendations.append("Long play time with low complexity might lead to player boredom - consider adding more strategic depth")
                
                if max_players > 6 and 'worker placement' in selected_mechanics:
                    recommendations.append("Worker placement games with many players can suffer from downtime - consider adding simultaneous play elements")
                
                if len(selected_mechanics) < 3:
                    recommendations.append("Adding more complementary mechanics could enhance gameplay variety")
                
                if selected_domain == 'strategy games' and complexity < 2.5:
                    recommendations.append("Strategy game fans typically prefer higher complexity - consider adding more strategic depth")
                
                if not recommendations:
                    recommendations.append("Your game design appears well-balanced for its target audience")
                
                for i, rec in enumerate(recommendations):
                    st.write(f"{i+1}. {rec}")
                
                # Close the summary box div
                st.markdown('</div>', unsafe_allow_html=True)
               
    # ABOUT PAGE
    elif st.session_state.page == "About":
        st.header("ℹ️ About the Project")
        st.write("""
        ## Board Game Success Predictor
        This project analyzes BoardGameGeek data to help game designers, publishers, and enthusiasts understand what makes board games successful.

        ### Why This Matters
        The board game industry has grown significantly, with global sales exceeding $13 billion in 2021. Understanding what factors contribute to a game's success can help:
        
        - **Game Designers**: Create more appealing games
        - **Publishers**: Make informed production decisions
        - **Retailers**: Stock games likely to sell well
        - **Consumers**: Find games they'll enjoy

        ### Technologies Used:
        - Python 🐍
        - Pandas & NumPy for data analysis 📊
        - Streamlit for the user interface 🌊
        - Plotly for visualizations 📈
        - Scikit-learn for machine learning 🤖

        ### Data Sources:
        - Data collected from BoardGameGeek database 🎲
        - Over 20,000 games analyzed with 100+ features
        
        #### Lead TA 👨‍🏫
        - Cynthia Siew-Tu
        
        #### Team members 👨‍💼👩‍💼
        - Maxence Dauphin
        - Mónica Costa
        - Tahar Guenfoud
        - Bernhard Riemer
        - Konstantin
        """)

# Navigation buttons in the sidebar
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()
with col2:
    if st.button("🔮 Predict", use_container_width=True):
        st.session_state.page = "Predictive Analysis"
        st.rerun()
with col3:
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "About"
        st.rerun()

# Run the main function
main()