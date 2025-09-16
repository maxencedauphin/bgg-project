# Import necessary libraries for the application
import streamlit as st
import requests

from prediction import (GAME_DOMAINS, GAME_MECHANICS)

# Configure the page appearance and layout
st.set_page_config(page_title="BGG Project", page_icon="🎲", layout="wide")

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

# Function to predict rating based on user inputs
def predict_game_rating(model, input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    return prediction

# Main function that runs the application
def main():
    add_logo_to_sidebar()
    st.title("🎲 BGG Project - Board Game Analysis")

    # Navigation sidebar
    page = st.sidebar.radio("Choose a section:",
                            ["Predictive Analysis", "About"])


    # PREDICTIVE ANALYSIS PAGE
    if page == "Predictive Analysis":
        st.header("🔮 Predictive Analysis")



        # Predict rating for a new game
        st.subheader("Predict Rating for a New Game")

        # Create form for user inputs
        with st.form("prediction_form"):
            st.write("Enter game characteristics to predict its rating:")

            # Create two columns for the form
            col1, col2 = st.columns(2)

            # First column inputs
            with col1:
                min_players = st.slider("Min Players", 1, 20, 1)
                max_players = st.slider("Max Players", 2, 20, 2)


            # Second column inputs
            with col2:
                min_age = st.slider("Min Age", 1, 99, 1)
                complexity = st.slider("Complexity Average", 1, 5, 2, step=1)
                #year = st.slider("Year Published", 1950, 2025, 2020)

            play_time = st.slider("Play Time (minutes)", 1, 260, 30, step=10)

            # Full width inputs
            common_mechanics = GAME_MECHANICS

            selected_mechanics = st.multiselect(
                "Common Game Mechanics",
                options=common_mechanics,
                #default=['dice rolling', 'card drafting'],
                help="Select multiple mechanics that apply to your game"
            )
            mechanics = sorted(selected_mechanics) #st.text_input("Mechanics (comma-separated)", "Dice Rolling, Card Drafting")


            #unique_domains = ["Strategy", "Family", "Party", "Abstract", "Thematic", "War", "Customizable"]

            selected_domain = st.selectbox("Domain", GAME_DOMAINS)

            # Submit button
            submitted = st.form_submit_button("Predict Rating")

        # Process form submission
        if submitted:

            prediction = load_prediction_from_remote(min_players, max_players, play_time, min_age, complexity, selected_domain, mechanics)

            if prediction is not None:
                st.success(f"Predicted Rating: {prediction:.2f}/10")
            if max_players < min_players:
                st.success("Please enter a greater number of max_players (compared to min_players)")


    # ABOUT PAGE
    else:
        st.header("ℹ️ About the Project")
        st.write("""
        ## BGG Project
        This project analyzes BoardGameGeek data to better understand the board game industry.

        ### Technologies Used:
        - Python 🐍
        - Pandas & NumPy for data analysis 📊
        - Streamlit for the user interface 🌊
        - Plotly & Matplotlib for visualizations 📈
        - Scikit-learn for predictive models 🤖

        ### Data Sources:
        - Data collected from BoardGameGeek database 🎲


        #### Lead TA 👨‍🏫
        - Cynthia Siew-Tu

        #### Team members 👨‍💼👩‍💼
        - Maxence Dauphin
        - Mónica Costa
        - Tahar Guenfoud
        - Konstantin
        - Bernhard Riemer
        """)

def load_prediction_from_remote(player_min: int, player_max: int, play_time_min: int, age_min: int, complexity: float, domains: str, mechanics: list[str]) :
    params={
        #'year_published': year_published,
        'min_players': player_min,
        'max_players': player_max,
        'play_time': play_time_min,
        'min_age': age_min,
        'complexity': complexity,
        'mechanics': mechanics,
        'domains': domains
    }
    url=st.secrets['my_url']

    try:
        response = requests.get(url=url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()['prediction']
        elif response.status_code == 422:
            print("Request failed:", response.status_code)
            print(response.json())
            return None
        else:
            print("Request failed:", response.status_code)
            print(response.text)
            return None

    except requests.exceptions.ConnectionError:
        st.error("Prediction service is not available. Please try again later.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request to prediction service timed out.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# Run the application
if __name__ == "__main__":
    main()
